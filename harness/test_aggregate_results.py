"""Offline tests for aggregate_results.

Each test writes small results CSVs and routing configs into a temp folder,
shaped like the files experiment_runner, run_sql_matrix and run_qa_matrix
produce, so no model calls, datasets or network are needed.

From the repository root:

    python -m unittest discover -s harness -t . -p "test_*.py"
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import aggregate_results as agg

# Column layouts as written by the runner on main (step rows, no config
# column) and by the K8 QA matrix (question rows with config, f1, em).
MAIN_COLUMNS = [
    "question_id", "pipeline", "difficulty", "step", "model", "tokens_in",
    "tokens_out", "cost", "time_sec", "raw_output", "final_answer",
    "is_correct", "error_propagated",
]
K8_COLUMNS = ["config"] + MAIN_COLUMNS[:-1] + ["f1", "em", "error_propagated"]

SQL_STEPS = ["sql.selector", "sql.decomposer", "sql.refiner"]


def write_csv(path: Path, columns: list, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return path


def sql_step_rows(qid, models, is_correct, cost_per_step, propagated="",
                  config=None, raw_outputs=None):
    """Step rows for one SQL question; models maps step -> model."""
    rows = []
    for i, step in enumerate(SQL_STEPS):
        row = {
            "question_id": str(qid), "pipeline": "sql", "difficulty": "simple",
            "step": step, "model": models[step], "tokens_in": 100, "tokens_out": 10,
            "cost": f"{cost_per_step:.8f}", "time_sec": "0.5000",
            "raw_output": (raw_outputs or {}).get(step, f"out-{step}"),
            "final_answer": "SELECT 1", "is_correct": is_correct,
            "error_propagated": propagated,
        }
        if config is not None:
            row["config"] = config
        rows.append(row)
    return rows


def qa_question_row(config, qid, is_correct, f1, em, cost):
    return {
        "config": config, "question_id": str(qid), "pipeline": "qa",
        "difficulty": "bridge", "step": "", "model": "", "tokens_in": 300,
        "tokens_out": 30, "cost": f"{cost:.8f}", "time_sec": "2.0000",
        "raw_output": "", "final_answer": "Paris", "is_correct": is_correct,
        "f1": f"{f1:.4f}", "em": em, "error_propagated": "",
    }


def write_configs(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "all-flash": "flash",
        "cheap-selector": {"sql": {"sql.selector": "flash-lite", "sql.decomposer": "flash",
                                   "sql.refiner": "flash"}},
        "qwen-refiner": {"sql": {"sql.selector": "flash", "sql.decomposer": "flash",
                                 "sql.refiner": "qwen"}},
        "qa-all-strong": {"qa": "flash"},
        "qa-cheap-decomposer": {"qa": {"qa.decomposer": "flash-lite", "qa.hop_reader": "flash",
                                       "qa.composer": "flash", "qa.formatter": "flash"}},
        "qa-qwen-decomposer": {"qa": {"qa.decomposer": "qwen", "qa.hop_reader": "flash",
                                      "qa.composer": "flash", "qa.formatter": "flash"}},
    }
    for name, body in configs.items():
        (config_dir / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")


ALL_FLASH = {s: "flash" for s in SQL_STEPS}
CHEAP_SEL = dict(ALL_FLASH, **{"sql.selector": "flash-lite"})
QWEN_REF = dict(ALL_FLASH, **{"sql.refiner": "qwen"})


def build_fixture(root: Path) -> dict:
    """Write a small SQL + QA experiment into root and return the paths."""
    write_configs(root / "configs")
    sql_dir = root / "sql"
    # all-flash: q1 right, q2 right, q3 wrong. Main schema, config from file name.
    baseline = write_csv(sql_dir / "results-all-flash.csv", MAIN_COLUMNS,
                         sql_step_rows(1, ALL_FLASH, "True", 0.004)
                         + sql_step_rows(2, ALL_FLASH, "True", 0.004)
                         + sql_step_rows(3, ALL_FLASH, "False", 0.004))
    # cheap-selector: q2 flips right -> wrong with propagation; accuracy 1/3.
    cheap = write_csv(sql_dir / "results-cheap-selector.csv", MAIN_COLUMNS,
                      sql_step_rows(1, CHEAP_SEL, "True", 0.003, "False")
                      + sql_step_rows(2, CHEAP_SEL, "False", 0.003, "True")
                      + sql_step_rows(3, CHEAP_SEL, "False", 0.003, "False"))
    # qwen-refiner: same answers as baseline, cheaper. K8 schema with config column.
    qwen = write_csv(sql_dir / "qwen-refiner.csv", K8_COLUMNS,
                     sql_step_rows(1, QWEN_REF, "True", 0.002, "False", config="qwen-refiner")
                     + sql_step_rows(2, QWEN_REF, "True", 0.002, "False", config="qwen-refiner")
                     + sql_step_rows(3, QWEN_REF, "False", 0.002, "False", config="qwen-refiner"))
    # QA matrix: one file, question rows, three configs.
    qa = write_csv(root / "qa_results.csv", K8_COLUMNS, [
        qa_question_row("qa-all-strong", 1, "True", 1.0, "True", 0.020),
        qa_question_row("qa-all-strong", 2, "True", 0.9, "False", 0.020),
        qa_question_row("qa-all-strong", 3, "False", 0.2, "False", 0.020),
        qa_question_row("qa-cheap-decomposer", 1, "True", 1.0, "True", 0.015),
        qa_question_row("qa-cheap-decomposer", 2, "True", 0.85, "False", 0.015),
        qa_question_row("qa-cheap-decomposer", 3, "False", 0.1, "False", 0.015),
        qa_question_row("qa-qwen-decomposer", 1, "True", 1.0, "True", 0.010),
        qa_question_row("qa-qwen-decomposer", 2, "False", 0.3, "False", 0.010),
        qa_question_row("qa-qwen-decomposer", 3, "False", 0.0, "False", 0.010),
    ])
    return {"config_dir": root / "configs", "inputs": [baseline, cheap, qwen, qa],
            "out_dir": root / "out"}


def by_config(result: dict) -> dict:
    return {(s["pipeline"], s["config"]): s for s in result["configs"]}


def by_option(result: dict) -> dict:
    return {(o["pipeline"], o["step"], o["model"]): o for o in result["steps"]}


class AggregateResultsTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fx = build_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_aggregate(self, **kwargs):
        return agg.aggregate(self.fx["inputs"], self.fx["out_dir"],
                             config_dir=self.fx["config_dir"], **kwargs)

    # -- reading ---------------------------------------------------------

    def test_config_from_filename_strips_results_prefix(self):
        self.assertEqual(agg.config_from_filename(Path("x/results-all-flash.csv")), "all-flash")
        self.assertEqual(agg.config_from_filename(Path("x/qa_results.csv")), "qa_results")
        self.assertEqual(agg.config_from_filename(Path("x/qwen-refiner.csv")), "qwen-refiner")

    def test_read_fills_missing_columns_and_config(self):
        rows = agg.read_results([self.fx["inputs"][0]])
        self.assertEqual(len(rows), 9)
        self.assertEqual(set(rows[0]), set(agg.ALL_COLUMNS))
        self.assertEqual({r["config"] for r in rows}, {"all-flash"})
        self.assertEqual(rows[0]["f1"], "")

    def test_read_keeps_config_column_when_present(self):
        rows = agg.read_results([self.fx["inputs"][2]])
        self.assertEqual({r["config"] for r in rows}, {"qwen-refiner"})

    def test_expand_inputs_skips_own_outputs(self):
        write_csv(self.root / "sql" / "summary.csv", ["question_id"], [])
        found = agg.expand_inputs([str(self.root / "sql" / "*.csv")])
        self.assertEqual(sorted(p.name for p in found),
                         ["qwen-refiner.csv", "results-all-flash.csv", "results-cheap-selector.csv"])

    # -- collapsing ------------------------------------------------------

    def test_question_rows_sum_cost_and_keep_scores(self):
        rows = agg.read_results([self.fx["inputs"][1]])
        qrows = agg.question_rows(rows)
        self.assertEqual(len(qrows), 3)
        q2 = next(q for q in qrows if q["question_id"] == "2")
        self.assertAlmostEqual(q2["cost"], 0.009)
        self.assertEqual(q2["tokens_in"], 300)
        self.assertEqual(q2["is_correct"], "False")
        self.assertEqual(q2["error_propagated"], "True")
        self.assertEqual(q2["steps"]["sql.selector"], "flash-lite")

    def test_question_rows_pass_through_qa_summary_rows(self):
        rows = agg.read_results([self.fx["inputs"][3]])
        qrows = agg.question_rows(rows)
        self.assertEqual(len(qrows), 9)
        self.assertEqual(qrows[0]["steps"], {})
        self.assertEqual(qrows[0]["f1"], "1.0000")

    # -- weak step -------------------------------------------------------

    def test_weak_step_from_plan(self):
        plan = {"sql.selector": "flash-lite", "sql.decomposer": "flash", "sql.refiner": "flash"}
        self.assertEqual(agg.weak_step_for(plan, {}, "flash"), ("sql.selector", "flash-lite"))
        self.assertEqual(agg.weak_step_for("flash", {}, "flash"), ("", ""))
        self.assertEqual(agg.weak_step_for("qwen", {}, "flash"), ("all", "qwen"))

    def test_weak_step_falls_back_to_step_rows_without_config_json(self):
        seen = {"sql.selector": "flash", "sql.decomposer": "qwen", "sql.refiner": "flash"}
        self.assertEqual(agg.weak_step_for(None, seen, "flash"), ("sql.decomposer", "qwen"))
        self.assertEqual(agg.weak_step_for(None, {}, "flash"), ("unknown", ""))
        two = {"sql.selector": "qwen", "sql.decomposer": "qwen", "sql.refiner": "flash"}
        self.assertEqual(agg.weak_step_for(None, two, "flash")[0], "multiple")

    # -- per-config summary ---------------------------------------------

    def test_config_summary_accuracy_cost_and_propagation(self):
        s = by_config(self.run_aggregate())
        base = s[("sql", "all-flash")]
        self.assertTrue(base["is_baseline"])
        self.assertEqual((base["n_questions"], base["n_scored"], base["n_correct"]), (3, 3, 2))
        self.assertAlmostEqual(base["accuracy"], 2 / 3)
        self.assertAlmostEqual(base["cost_per_question"], 0.012)
        self.assertIsNone(base["propagation_rate"])

        cheap = s[("sql", "cheap-selector")]
        self.assertEqual((cheap["weak_step"], cheap["weak_model"]), ("sql.selector", "flash-lite"))
        self.assertAlmostEqual(cheap["accuracy"], 1 / 3)
        self.assertEqual((cheap["n_propagation_judged"], cheap["n_propagated"]), (3, 1))
        self.assertAlmostEqual(cheap["propagation_rate"], 1 / 3)

    def test_right_to_wrong_against_baseline(self):
        s = by_config(self.run_aggregate())
        cheap = s[("sql", "cheap-selector")]
        self.assertEqual(cheap["n_paired"], 3)
        self.assertEqual(cheap["n_baseline_right"], 2)
        self.assertEqual(cheap["n_right_to_wrong"], 1)
        self.assertAlmostEqual(cheap["right_to_wrong_rate"], 0.5)
        self.assertAlmostEqual(cheap["delta_accuracy_pts"], -100 / 3)
        self.assertAlmostEqual(cheap["delta_cost_pct"], -25.0)
        qwen = s[("sql", "qwen-refiner")]
        self.assertEqual(qwen["n_right_to_wrong"], 0)
        self.assertAlmostEqual(qwen["delta_accuracy_pts"], 0.0)

    def test_qa_summary_has_f1_and_em(self):
        s = by_config(self.run_aggregate())
        base = s[("qa", "qa-all-strong")]
        self.assertTrue(base["is_baseline"])
        self.assertAlmostEqual(base["mean_f1"], (1.0 + 0.9 + 0.2) / 3)
        self.assertAlmostEqual(base["em_rate"], 1 / 3)
        self.assertEqual(base["n_propagation_judged"], 0)
        cheap = s[("qa", "qa-cheap-decomposer")]
        self.assertEqual(cheap["weak_step"], "qa.decomposer")
        self.assertAlmostEqual(cheap["right_to_wrong_rate"], 0.0)

    # -- per-step summary and recommendation ----------------------------

    def test_recommends_cheaper_option_within_tolerance(self):
        o = by_option(self.run_aggregate())
        self.assertTrue(o[("sql", "sql.refiner", "qwen")]["recommended"])
        self.assertFalse(o[("sql", "sql.refiner", "flash")]["recommended"])
        self.assertTrue(o[("qa", "qa.decomposer", "flash-lite")]["recommended"])
        self.assertFalse(o[("qa", "qa.decomposer", "qwen")]["recommended"])
        self.assertIn("beyond", o[("qa", "qa.decomposer", "qwen")]["reason"])

    def test_keeps_strong_when_drop_exceeds_tolerance(self):
        o = by_option(self.run_aggregate())
        self.assertTrue(o[("sql", "sql.selector", "flash")]["recommended"])
        self.assertFalse(o[("sql", "sql.selector", "flash-lite")]["recommended"])
        self.assertIn("beyond", o[("sql", "sql.selector", "flash-lite")]["reason"])

    def test_tolerance_is_applied(self):
        o = by_option(self.run_aggregate(tolerance=50.0))
        self.assertTrue(o[("sql", "sql.selector", "flash-lite")]["recommended"])

    def test_exactly_one_recommendation_per_step(self):
        result = self.run_aggregate()
        counts = {}
        for o in result["steps"]:
            counts[(o["pipeline"], o["step"])] = counts.get((o["pipeline"], o["step"]), 0) + o["recommended"]
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(set(counts), {("sql", "sql.selector"), ("sql", "sql.refiner"),
                                       ("qa", "qa.decomposer")})

    def test_without_baseline_no_recommendation(self):
        inputs = [self.fx["inputs"][1]]  # cheap-selector only
        result = agg.aggregate(inputs, self.fx["out_dir"], config_dir=self.fx["config_dir"])
        self.assertEqual(len(result["steps"]), 1)
        self.assertFalse(result["steps"][0]["recommended"])
        self.assertIn("no all-strong baseline", result["steps"][0]["reason"])

    # -- files -----------------------------------------------------------

    def test_writes_three_files_with_expected_columns(self):
        self.run_aggregate()
        out = self.fx["out_dir"]
        for name, columns in (("all_results.csv", agg.ALL_COLUMNS),
                              ("config_summary.csv", agg.CONFIG_SUMMARY_COLUMNS),
                              ("summary.csv", agg.SUMMARY_COLUMNS)):
            with (out / name).open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, columns, name)
                rows = list(reader)
            self.assertTrue(rows, name)
        with (out / "all_results.csv").open(encoding="utf-8", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 27 + 9)
        with (out / "summary.csv").open(encoding="utf-8", newline="") as f:
            recommended = [r for r in csv.DictReader(f) if r["recommended"] == "True"]
        self.assertEqual(len(recommended), 3)

    def test_main_reports_missing_inputs(self):
        code = agg.main(["--inputs", str(self.root / "nothing" / "*.csv"),
                         "--out-dir", str(self.fx["out_dir"])])
        self.assertEqual(code, 1)

    def test_main_end_to_end(self):
        code = agg.main(["--inputs", str(self.root / "sql" / "*.csv"),
                         str(self.root / "qa_results.csv"),
                         "--out-dir", str(self.fx["out_dir"]),
                         "--config-dir", str(self.fx["config_dir"])])
        self.assertEqual(code, 0)
        self.assertTrue((self.fx["out_dir"] / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
