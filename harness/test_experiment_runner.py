"""Offline tests for experiment_runner.

These tests replace the pipeline run with a fake trace, so they need no
API key, no GPU, no network, and no BIRD databases.

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
from experiment_runner import (
    CSV_COLUMNS,
    append_rows,
    completed_ids,
    load_config,
    models_for_pipeline,
    rows_from_trace,
    run_experiment,
    task_difficulty,
    task_id,
)


FAKE_TRACE = {
    "final_sql": "SELECT 1",
    "stop_reason": "not executed (no database)",
    "calls": [
        {
            "step": "sql.selector",
            "model": "flash",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cost": 0.000033,
            "time_sec": 0.12,
            "raw_output": '{"customers": "keep_all"}',
            "cached": False,
        },
        {
            "step": "sql.decomposer",
            "model": "flash",
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "cost": 0.000075,
            "time_sec": 0.34,
            "raw_output": "```sql\nSELECT 1\n```",
            "cached": False,
        },
    ],
}


class ExperimentRunnerTests(unittest.TestCase):
    def test_all_flash_config_loads(self):
        config = load_config("all-flash")
        self.assertEqual(models_for_pipeline(config, "sql"), "flash")
        self.assertEqual(models_for_pipeline(config, "qa"), "flash")

    def test_per_step_config(self):
        plan = {
            "sql.selector": "flash-lite",
            "sql.decomposer": "flash",
            "sql.refiner": "flash",
        }
        self.assertEqual(models_for_pipeline(plan, "sql"), plan)

    def test_task_id_and_difficulty(self):
        self.assertEqual(task_id("sql", {"question_id": 231}), "231")
        self.assertEqual(task_difficulty("sql", {"difficulty": "challenging"}), "challenging")
        self.assertEqual(task_id("qa", {"id": "abc"}), "abc")
        self.assertEqual(task_difficulty("qa", {"type": "bridge"}), "bridge")

    def test_rows_from_trace(self):
        rows = rows_from_trace("sql", {"question_id": 231, "difficulty": "simple"}, FAKE_TRACE)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["step"] for r in rows], ["sql.selector", "sql.decomposer"])
        self.assertEqual(rows[0]["question_id"], "231")
        self.assertEqual(rows[0]["final_answer"], "SELECT 1")
        self.assertEqual(rows[0]["is_correct"], "")
        self.assertEqual(set(rows[0]), set(CSV_COLUMNS))

    def test_resume_skips_completed_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            append_rows(path, [{
                "question_id": "231",
                "pipeline": "sql",
                "difficulty": "challenging",
                "step": "sql.selector",
                "model": "flash",
                "tokens_in": 1,
                "tokens_out": 1,
                "cost": "0.00000001",
                "time_sec": "0.0100",
                "raw_output": "x",
                "final_answer": "SELECT 1",
                "is_correct": "",
            }])
            self.assertEqual(completed_ids(path, "sql"), {"231"})
            self.assertEqual(completed_ids(path, "qa"), set())

    def test_run_experiment_writes_csv_and_resumes(self):
        sample = [
            {"question_id": 101, "difficulty": "simple", "db_id": "x", "question": "q1"},
            {"question_id": 102, "difficulty": "moderate", "db_id": "x", "question": "q2"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            results = folder_path / "results.csv"
            config_path = folder_path / "all-flash.json"
            config_path.write_text(json.dumps({"sql": "flash"}), encoding="utf-8")

            with mock.patch.object(experiment_runner, "load_tasks", return_value=sample), \
                 mock.patch.object(experiment_runner, "run_one", return_value=FAKE_TRACE) as run_one:
                first = run_experiment(
                    "sql", str(config_path), limit=2, results_path=results, db_dir=None
                )
                second = run_experiment(
                    "sql", str(config_path), limit=2, results_path=results, db_dir=None
                )

            self.assertEqual(first, 2)
            self.assertEqual(second, 0)
            self.assertEqual(run_one.call_count, 2)

            with results.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0].keys()), CSV_COLUMNS)
            self.assertEqual(len(rows), 4)  # 2 questions × 2 steps
            self.assertEqual({r["question_id"] for r in rows}, {"101", "102"})

    def test_qa_not_wired_yet(self):
        with self.assertRaises(ValueError):
            experiment_runner.run_one("qa", {"id": "x"}, "flash")

    def test_run_experiment_skips_failed_question_without_writing(self):
        sample = [
            {"question_id": 201, "difficulty": "simple", "db_id": "x", "question": "q1"},
            {"question_id": 202, "difficulty": "moderate", "db_id": "x", "question": "q2"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            results = folder_path / "results.csv"
            config_path = folder_path / "all-flash.json"
            config_path.write_text(json.dumps({"sql": "flash"}), encoding="utf-8")

            def boom_then_ok(pipeline, task, models, db_dir=None):
                if task["question_id"] == 201:
                    raise RuntimeError("provider busy")
                return FAKE_TRACE

            with mock.patch.object(experiment_runner, "load_tasks", return_value=sample), \
                 mock.patch.object(experiment_runner, "run_one", side_effect=boom_then_ok):
                ran = run_experiment(
                    "sql", str(config_path), limit=2, results_path=results, db_dir=None
                )

            self.assertEqual(ran, 1)
            with results.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["question_id"] for r in rows}, {"202"})


if __name__ == "__main__":
    unittest.main()
