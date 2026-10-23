"""Offline tests for error propagation tracking.

The judge is tested on hand-built CSV rows, and the runner is tested with a
fake pipeline trace and a fake scorer, so these tests need no API key, no
network, and no BIRD databases.

From the repository root:

    python -m unittest discover -s harness -t . -p "test_*.py"
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import experiment_runner
from experiment_runner import CSV_COLUMNS, append_rows, run_experiment
from harness.error_propagation import error_propagated

STRONG, WEAK = "flash", "flash-lite"

# Two Selector replies that choose the same tables and columns, written differently.
SELECTION_A = '{"customers": ["id", "name"], "orders": "drop_all"}'
SELECTION_A_REWRITTEN = '```json\n{"orders": "drop_all", "customers": ["name", "id"]}\n```'
# A Selector reply that chooses differently.
SELECTION_B = '{"customers": "keep_all", "orders": "keep_all"}'


def row(step, model, raw_output, is_correct, qid="1"):
    return {"question_id": qid, "pipeline": "sql", "step": step, "model": model,
            "raw_output": raw_output, "is_correct": is_correct}


def baseline(selection=SELECTION_A, correct="True"):
    """All-strong rows as read back from a CSV (is_correct is a string)."""
    return [row("sql.selector", STRONG, selection, correct),
            row("sql.decomposer", STRONG, "SELECT 1", correct)]


def cheap_selector(selection=SELECTION_B, correct=False):
    """Cheap-Selector rows as built in memory (is_correct is a bool)."""
    return [row("sql.selector", WEAK, selection, correct),
            row("sql.decomposer", STRONG, "SELECT 2", correct)]


class ErrorPropagatedTests(unittest.TestCase):
    def test_true_when_baseline_right_mixed_wrong_and_selector_changed(self):
        self.assertIs(error_propagated(cheap_selector(), baseline()), True)

    def test_false_when_selector_chose_the_same_tables(self):
        mixed = cheap_selector(selection=SELECTION_A_REWRITTEN)
        self.assertIs(error_propagated(mixed, baseline()), False)

    def test_false_when_baseline_was_also_wrong(self):
        self.assertIs(error_propagated(cheap_selector(), baseline(correct="False")), False)

    def test_false_when_mixed_config_was_right(self):
        self.assertIs(error_propagated(cheap_selector(correct=True), baseline()), False)

    def test_false_when_weak_model_never_ran(self):
        # e.g. a cheap refiner that was never triggered: every call used the strong model.
        mixed = [row("sql.selector", STRONG, SELECTION_B, False),
                 row("sql.decomposer", STRONG, "SELECT 2", False)]
        self.assertIs(error_propagated(mixed, baseline()), False)

    def test_weak_step_missing_from_baseline_counts_as_changed(self):
        mixed = [row("sql.selector", STRONG, SELECTION_A, False),
                 row("sql.decomposer", STRONG, "SELECT 1", False),
                 row("sql.refiner", WEAK, "SELECT 3", False)]
        self.assertIs(error_propagated(mixed, baseline()), True)

    def test_blank_without_baseline_rows(self):
        self.assertEqual(error_propagated(cheap_selector(), []), "")

    def test_blank_when_baseline_unscored(self):
        self.assertEqual(error_propagated(cheap_selector(), baseline(correct="")), "")

    def test_blank_when_mixed_unscored(self):
        self.assertEqual(error_propagated(cheap_selector(correct=""), baseline()), "")

    def test_blank_when_more_than_one_weak_step(self):
        mixed = [row("sql.selector", WEAK, SELECTION_B, False),
                 row("sql.decomposer", WEAK, "SELECT 2", False)]
        self.assertEqual(error_propagated(mixed, baseline()), "")

    def test_blank_when_baseline_is_not_all_one_model(self):
        mixed_baseline = [row("sql.selector", WEAK, SELECTION_A, "True"),
                          row("sql.decomposer", STRONG, "SELECT 1", "True")]
        self.assertEqual(error_propagated(cheap_selector(), mixed_baseline), "")


def fake_trace(selection):
    return {
        "final_sql": "SELECT 2",
        "stop_reason": "ok",
        "calls": [
            {"step": "sql.selector", "model": WEAK, "tokens_in": 10, "tokens_out": 2,
             "cost": 0.00001, "time_s": 0.1, "raw_output": selection, "cached": False},
            {"step": "sql.decomposer", "model": STRONG, "tokens_in": 20, "tokens_out": 5,
             "cost": 0.00007, "time_s": 0.3, "raw_output": "SELECT 2", "cached": False},
        ],
    }


class RunnerErrorPropagationTests(unittest.TestCase):
    SAMPLE = [
        {"question_id": 301, "difficulty": "simple", "db_id": "x", "question": "q1"},
        {"question_id": 302, "difficulty": "simple", "db_id": "x", "question": "q2"},
    ]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.config = self.folder / "cheap-selector.json"
        self.config.write_text(json.dumps({"sql": {
            "sql.selector": WEAK, "sql.decomposer": STRONG, "sql.refiner": STRONG,
        }}), encoding="utf-8")
        self.baseline = self.folder / "results-all-flash.csv"
        # Q301: the baseline chose differently and was right. Q302: same choice as the cheap run.
        append_rows(self.baseline,
                    [dict(r, question_id="301") for r in baseline(SELECTION_A)]
                    + [dict(r, question_id="302") for r in baseline(SELECTION_B)])
        self.results = self.folder / "results-cheap-selector.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def run_cheap_selector(self, baseline_path):
        with mock.patch.object(experiment_runner, "load_tasks", return_value=self.SAMPLE), \
             mock.patch.object(experiment_runner, "run_one",
                               return_value=fake_trace(SELECTION_B)), \
             mock.patch.object(experiment_runner, "score_answer", return_value=False):
            return run_experiment("sql", str(self.config), limit=2, results_path=self.results,
                                  db_dir=None, baseline_path=baseline_path)

    def read_results(self):
        with self.results.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_fills_error_propagated_per_question(self):
        self.assertEqual(self.run_cheap_selector(self.baseline), 2)
        rows = self.read_results()
        self.assertEqual(list(rows[0].keys()), CSV_COLUMNS)
        by_question = {r["question_id"]: {x["error_propagated"] for x in rows
                                          if x["question_id"] == r["question_id"]}
                       for r in rows}
        self.assertEqual(by_question, {"301": {"True"}, "302": {"False"}})

    def test_blank_without_baseline(self):
        self.run_cheap_selector(None)
        self.assertEqual({r["error_propagated"] for r in self.read_results()}, {""})

    def test_baseline_must_differ_from_results(self):
        with self.assertRaises(ValueError):
            self.run_cheap_selector(self.results)

    def test_missing_baseline_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            self.run_cheap_selector(self.folder / "missing.csv")


class AppendRowsHeaderTests(unittest.TestCase):
    def test_old_results_file_keeps_its_columns(self):
        old_columns = [c for c in CSV_COLUMNS if c != "error_propagated"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(old_columns)
            append_rows(path, [{c: c.upper() for c in CSV_COLUMNS}])
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0].keys()), old_columns)
        self.assertEqual(rows[0]["is_correct"], "IS_CORRECT")


if __name__ == "__main__":
    unittest.main()
