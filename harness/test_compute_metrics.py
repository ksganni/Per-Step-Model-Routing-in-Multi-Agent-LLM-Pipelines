"""Offline tests for compute_metrics.

The statistics helpers are checked against values worked out by hand; the
end-to-end tests write small CSVs shaped like the SQL step rows and the QA
question rows, so no model calls or datasets are needed.

From the repository root:

    python -m unittest discover -s harness -t . -p "test_*.py"
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import compute_metrics as cm
from harness.test_aggregate_results import (
    ALL_FLASH, CHEAP_SEL, K8_COLUMNS, MAIN_COLUMNS, qa_question_row, sql_step_rows,
    write_configs, write_csv,
)


class StatsHelperTests(unittest.TestCase):

    def test_wilson_interval_matches_hand_calculation(self):
        lo, hi = cm.wilson_ci(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=4)
        self.assertAlmostEqual(hi, 0.9433, places=4)
        self.assertEqual(cm.wilson_ci(0, 0), (None, None))
        lo, hi = cm.wilson_ci(0, 5)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_mcnemar_exact(self):
        self.assertAlmostEqual(cm.mcnemar_exact(5, 0), 0.0625)
        self.assertEqual(cm.mcnemar_exact(1, 1), 1.0)
        self.assertEqual(cm.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(cm.mcnemar_exact(10, 2), 2 * sum(
            __import__("math").comb(12, i) for i in range(3)) / 2 ** 12)

    def test_bonett_price_interval(self):
        lo, hi = cm.bonett_price_ci(0, 0, 25)
        self.assertAlmostEqual(hi, 1.959963984540054 * (2 / 27 / 27) ** 0.5, places=6)
        self.assertAlmostEqual(lo, -hi, places=9)
        lo, hi = cm.bonett_price_ci(20, 0, 35)
        self.assertAlmostEqual(lo, -0.7177, places=3)
        self.assertAlmostEqual(hi, -0.3633, places=3)
        lo, hi = cm.bonett_price_ci(0, 0, 3)
        self.assertLess(lo, -0.5)
        self.assertEqual(cm.bonett_price_ci(0, 0, 0), (None, None))

    def test_holm_adjustment(self):
        self.assertEqual(cm.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(cm.holm([None, 0.02]), [None, 0.02])
        self.assertEqual(cm.holm([0.5, 0.6]), [1.0, 1.0])

    def test_percentile_interpolates(self):
        self.assertEqual(cm.percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertEqual(cm.percentile([7.0], 0.9), 7.0)

    def test_paired_bootstrap_identical_and_opposite(self):
        rng = random.Random(1)
        same = cm.paired_bootstrap([0.0] * 30, 200, rng)
        self.assertEqual((same["mean"], same["ci_low"], same["ci_high"], same["p"]), (0.0, 0.0, 0.0, 1.0))
        worse = cm.paired_bootstrap([-1.0] * 25 + [0.0] * 5, 2000, rng)
        self.assertAlmostEqual(worse["mean"], -25 / 30)
        self.assertLess(worse["ci_high"], 0)
        self.assertLess(worse["p"], 0.01)
        noise = cm.paired_bootstrap([1.0, -1.0] * 20, 2000, rng)
        self.assertGreater(noise["p"], 0.5)
        self.assertLess(noise["ci_low"], 0)
        self.assertGreater(noise["ci_high"], 0)

    def test_bootstrap_is_reproducible_with_seed(self):
        source = random.Random(3)
        values = [source.random() for _ in range(40)]
        a = cm.bootstrap_mean_ci(values, 500, random.Random(7))
        b = cm.bootstrap_mean_ci(values, 500, random.Random(7))
        self.assertEqual(a, b)
        self.assertLess(a[0], sum(values) / len(values))
        self.assertGreater(a[1], sum(values) / len(values))

    def test_pareto_front(self):
        points = [
            {"config": "A", "q": 0.80, "c": 1.0},
            {"config": "B", "q": 0.90, "c": 2.0},
            {"config": "C", "q": 0.70, "c": 2.0},
            {"config": "D", "q": 0.85, "c": 3.0},
            {"config": "E", "q": None, "c": 0.5},
        ]
        front = cm.pareto_front(points, "q", "c")
        self.assertEqual(front["A"], [])
        self.assertEqual(front["B"], [])
        self.assertEqual(front["C"], ["A", "B"])
        self.assertEqual(front["D"], ["B"])
        self.assertNotIn("E", front)


def qa_rows(config, qids, f1_of, cost):
    rows = []
    for q in qids:
        f1 = f1_of(q)
        rows.append(qa_question_row(config, q, "True" if f1 >= 0.8 else "False", f1,
                                    "True" if f1 == 1.0 else "False", cost))
    return rows


class EndToEndTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_configs(self.root / "configs")
        qids = list(range(1, 41))
        # Baseline: F1 1.0 on q1..30, 0.2 on q31..40.  cheap-decomposer: same answers,
        # cheaper.  qwen-decomposer: loses q1..20.
        base = qa_rows("qa-all-strong", qids, lambda q: 1.0 if q <= 30 else 0.2, 0.020)
        cheap = qa_rows("qa-cheap-decomposer", qids, lambda q: 1.0 if q <= 30 else 0.2, 0.012)
        qwen = qa_rows("qa-qwen-decomposer", qids[:35], lambda q: 0.0 if q <= 20 else (1.0 if q <= 30 else 0.2),
                       0.008)
        self.qa = write_csv(self.root / "qa_results.csv", K8_COLUMNS, base + cheap + qwen)
        sql_rows = []
        for q in range(1, 6):
            sql_rows += sql_step_rows(q, ALL_FLASH, "True" if q <= 4 else "False", 0.004)
        self.sql_base = write_csv(self.root / "sql" / "all-flash.csv", MAIN_COLUMNS, sql_rows)
        cheap_rows = []
        for q in range(1, 6):
            cheap_rows += sql_step_rows(q, CHEAP_SEL, "True" if q <= 2 else "False", 0.003,
                                        "True" if q in (3, 4) else "False")
        self.sql_cheap = write_csv(self.root / "sql" / "cheap-selector.csv", MAIN_COLUMNS, cheap_rows)
        self.inputs = [self.qa, self.sql_base, self.sql_cheap]

    def tearDown(self):
        self.tmp.cleanup()

    def run_compute(self, **kwargs):
        kwargs.setdefault("resamples", 1000)
        return cm.compute(self.inputs, config_dir=self.root / "configs", **kwargs)

    def by_config(self, res, pipeline):
        return {c["config"]: c for c in res["pipelines"][pipeline]["configs"]}

    def comparison(self, res, pipeline, config):
        return next(c for c in res["pipelines"][pipeline]["comparisons"] if c["config"] == config)

    def test_config_quality_and_primary_metric(self):
        res = self.run_compute()
        qa = res["pipelines"]["qa"]
        self.assertEqual(qa["primary_metric"], "mean_f1")
        self.assertEqual(qa["baseline"], "qa-all-strong")
        base = self.by_config(res, "qa")["qa-all-strong"]
        self.assertEqual(base["n_questions"], 40)
        self.assertAlmostEqual(base["accuracy"], 0.75)
        self.assertAlmostEqual(base["mean_f1"], (30 * 1.0 + 10 * 0.2) / 40)
        self.assertLess(base["accuracy_ci"][0], 0.75)
        self.assertGreater(base["accuracy_ci"][1], 0.75)
        self.assertAlmostEqual(base["cost_per_question"], 0.020)
        self.assertEqual(res["pipelines"]["sql"]["primary_metric"], "accuracy")

    def test_paired_comparison_counts_and_tests(self):
        res = self.run_compute()
        qwen = self.comparison(res, "qa", "qa-qwen-decomposer")
        self.assertEqual(qwen["n_paired"], 35)
        self.assertEqual(qwen["baseline_right_config_wrong"], 20)
        self.assertEqual(qwen["baseline_wrong_config_right"], 0)
        self.assertEqual(qwen["n_baseline_right"], 30)
        self.assertAlmostEqual(qwen["right_to_wrong_rate"], 20 / 30)
        self.assertAlmostEqual(qwen["p_accuracy_mcnemar"], 2 / 2 ** 20)
        self.assertLess(qwen["delta_f1_ci"][1], 0)
        self.assertLess(qwen["p_f1_bootstrap_holm"], 0.05)
        self.assertAlmostEqual(qwen["cost_ratio"], 0.008 / 0.020)
        cheap = self.comparison(res, "qa", "qa-cheap-decomposer")
        self.assertEqual(cheap["delta_f1"], 0.0)
        self.assertEqual(cheap["p_accuracy_mcnemar"], 1.0)
        self.assertAlmostEqual(cheap["cost_ratio"], 0.6)

    def test_step_findings_verdicts(self):
        res = self.run_compute()
        verdicts = {(f["config"]): f["verdict"] for f in res["pipelines"]["qa"]["step_findings"]}
        self.assertEqual(verdicts["qa-cheap-decomposer"], "non-inferior")
        self.assertEqual(verdicts["qa-qwen-decomposer"], "worse")
        sql = res["pipelines"]["sql"]["step_findings"]
        self.assertEqual([f["verdict"] for f in sql], ["too few"])

    def test_no_change_on_few_questions_is_not_non_inferior(self):
        res = self.run_compute()
        cheap = self.comparison(res, "qa", "qa-cheap-decomposer")
        lo, hi = cheap["delta_accuracy_ci"]
        self.assertLess(lo, -0.05)
        self.assertGreater(hi, 0.05)
        sql = self.comparison(res, "sql", "cheap-selector")
        self.assertLess(sql["delta_accuracy_ci"][0], -0.4)

    def test_sql_propagation_from_error_propagated_column(self):
        res = self.run_compute()
        cheap = self.comparison(res, "sql", "cheap-selector")
        self.assertEqual(cheap["n_judged"], 5)
        self.assertEqual(cheap["n_propagated"], 2)
        self.assertAlmostEqual(cheap["propagation_rate"], 0.4)
        self.assertEqual(cheap["baseline_right_config_wrong"], 2)
        qa_cheap = self.comparison(res, "qa", "qa-cheap-decomposer")
        self.assertIsNone(qa_cheap["propagation_rate"])

    def test_pareto_on_own_and_common_questions(self):
        res = self.run_compute()
        par = res["pipelines"]["qa"]["pareto"]
        own = {p["config"]: p for p in par["own_questions"]}
        self.assertTrue(own["qa-cheap-decomposer"]["on_frontier"])
        self.assertFalse(own["qa-all-strong"]["on_frontier"])
        self.assertEqual(own["qa-all-strong"]["dominated_by"], ["qa-cheap-decomposer"])
        self.assertTrue(own["qa-qwen-decomposer"]["on_frontier"])
        self.assertTrue(par["common_questions"]["used"])
        self.assertEqual(par["common_questions"]["n_common"], 35)

    def test_duplicate_file_rows_are_not_double_counted(self):
        copy = write_csv(self.root / "results-all-flash.csv", MAIN_COLUMNS,
                         [r for q in range(1, 6) for r in sql_step_rows(q, ALL_FLASH, "True", 0.004)])
        res = cm.compute(self.inputs + [copy], config_dir=self.root / "configs", resamples=200)
        base = self.by_config(res, "sql")["all-flash"]
        self.assertAlmostEqual(base["cost_per_question"], 0.012)
        self.assertTrue(any("already read" in w for w in res["data_quality"]["warnings"]))

    def test_warnings_for_small_samples_and_missing_baseline(self):
        res = cm.compute([self.sql_cheap], config_dir=self.root / "configs", resamples=200)
        warnings = " ".join(res["data_quality"]["warnings"])
        self.assertIn("no all-strong baseline", warnings)
        self.assertIn("fewer than", warnings)
        self.assertEqual(res["pipelines"]["sql"]["comparisons"], [])

    def test_reproducible_with_seed(self):
        a = self.run_compute(seed=11)
        b = self.run_compute(seed=11)
        a["meta"].pop("generated_at")
        b["meta"].pop("generated_at")
        self.assertEqual(a, b)

    def test_writes_json_csv_and_markdown(self):
        out = self.root / "reports"
        code = cm.main(["--inputs", str(self.qa), str(self.root / "sql" / "*.csv"),
                        "--out-dir", str(out), "--config-dir", str(self.root / "configs"),
                        "--resamples", "300"])
        self.assertEqual(code, 0)
        data = json.loads((out / "metrics.json").read_text())
        self.assertEqual(set(data["pipelines"]), {"qa", "sql"})
        for name in ("config_quality", "paired_tests", "error_propagation", "pareto", "step_findings"):
            self.assertTrue((out / "tables" / f"{name}.csv").is_file(), name)
        md = (out / "metrics_tables.md").read_text()
        self.assertIn("## Step findings", md)
        self.assertIn("non-inferior", md)

    def test_main_without_inputs(self):
        self.assertEqual(cm.main(["--inputs", str(self.root / "none" / "*.csv"),
                                  "--out-dir", str(self.root / "o")]), 1)


if __name__ == "__main__":
    unittest.main()
