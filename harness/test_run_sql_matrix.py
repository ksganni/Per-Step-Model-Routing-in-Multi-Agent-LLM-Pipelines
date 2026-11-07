"""Offline tests for run_sql_matrix.

run_experiment is replaced with a recorder, so no model calls are made.

From the repository root:

    python -m unittest discover -s harness -t . -p "test_*.py"
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_sql_matrix as matrix
from experiment_runner import CONFIG_DIR, load_config, models_for_pipeline

SQL_STEPS = ["sql.selector", "sql.decomposer", "sql.refiner"]


class SqlMatrixConfigTests(unittest.TestCase):

    def test_all_seven_configs_exist_and_swap_one_step(self):
        self.assertEqual(len(matrix.SQL_MATRIX_CONFIGS), 7)
        for name in matrix.SQL_MATRIX_CONFIGS:
            self.assertTrue((CONFIG_DIR / f"{name}.json").is_file(), name)
            plan = models_for_pipeline(load_config(name), "sql")
            if name == matrix.BASELINE_CONFIG:
                self.assertEqual(plan, "flash")
                continue
            self.assertEqual(set(plan), set(SQL_STEPS), name)
            weak = {s: m for s, m in plan.items() if m != "flash"}
            self.assertEqual(len(weak), 1, name)
            tier, step = name.split("-")
            self.assertEqual(list(weak), [f"sql.{step}"], name)
            self.assertEqual(list(weak.values()), ["flash-lite" if tier == "cheap" else "qwen"], name)


class SqlMatrixRunTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmp.name) / "sql"
        self.calls = []

        def record(pipeline, name, **kwargs):
            self.calls.append((pipeline, name, kwargs))
            return 2

        self.patcher = mock.patch.object(matrix, "run_experiment", side_effect=record)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def test_baseline_runs_first_without_baseline_and_others_with_it(self):
        total = matrix.run_matrix(["qwen-refiner", "all-flash", "cheap-selector"],
                                  results_dir=self.results_dir, limit=5, db_dir="dbs")
        self.assertEqual(total, 6)
        names = [name for _, name, _ in self.calls]
        self.assertEqual(names, ["all-flash", "qwen-refiner", "cheap-selector"])
        first = self.calls[0][2]
        self.assertIsNone(first["baseline_path"])
        self.assertEqual(first["results_path"], self.results_dir / "all-flash.csv")
        self.assertEqual(first["limit"], 5)
        self.assertEqual(first["db_dir"], "dbs")
        for _, name, kwargs in self.calls[1:]:
            self.assertEqual(kwargs["baseline_path"], self.results_dir / "all-flash.csv", name)
            self.assertEqual(kwargs["results_path"], self.results_dir / f"{name}.csv", name)
        self.assertTrue(self.results_dir.is_dir())

    def test_default_order_is_full_matrix_with_baseline_first(self):
        matrix.run_matrix(matrix.SQL_MATRIX_CONFIGS, results_dir=self.results_dir)
        self.assertEqual([n for _, n, _ in self.calls], matrix.SQL_MATRIX_CONFIGS)
        self.assertEqual(self.calls[0][1], matrix.BASELINE_CONFIG)

    def test_subset_without_baseline_still_points_at_baseline_file(self):
        matrix.run_matrix(["cheap-refiner"], results_dir=self.results_dir)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][2]["baseline_path"], self.results_dir / "all-flash.csv")


if __name__ == "__main__":
    unittest.main()
