"""Offline tests for answer scoring in experiment_runner.

Each test builds a tiny SQLite database in a temp folder, laid out like the
BIRD databases (<db_id>/<db_id>.sqlite), so the tests need no BIRD download,
no API key, and no network.

From the repository root:

    python -m unittest discover -s harness -t . -p "test_*.py"
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiment_runner import rows_from_trace, score_answer

DB_ID = "toy"
GOLD_SQL = "SELECT name FROM people WHERE id = 1"
TASK = {"question_id": 1, "db_id": DB_ID, "SQL": GOLD_SQL, "difficulty": "simple"}


def make_db(root: Path) -> Path:
    """Create root/toy/toy.sqlite with a two-row table and return root."""
    folder = root / DB_ID
    folder.mkdir()
    con = sqlite3.connect(folder / f"{DB_ID}.sqlite")
    con.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT)")
    con.executemany("INSERT INTO people VALUES (?, ?)", [(1, "alice"), (2, "bob")])
    con.commit()
    con.close()
    return root


def fake_trace(final_sql: str) -> dict:
    """A trace with two model calls, shaped like run_sql_trace output."""
    return {
        "final_sql": final_sql,
        "stop_reason": "ok",
        "calls": [
            {"step": "sql.selector", "model": "flash", "prompt_tokens": 10, "completion_tokens": 2},
            {"step": "sql.decomposer", "model": "flash", "prompt_tokens": 20, "completion_tokens": 5},
        ],
    }


class ScoreAnswerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_dir = make_db(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_gold_sql_scores_true(self):
        self.assertEqual(score_answer("sql", TASK, GOLD_SQL, self.db_dir), True)

    def test_different_sql_with_same_result_scores_true(self):
        sql = "SELECT name FROM people WHERE name = 'alice'"
        self.assertEqual(score_answer("sql", TASK, sql, self.db_dir), True)

    def test_wrong_sql_scores_false(self):
        sql = "SELECT name FROM people WHERE id = 2"
        self.assertEqual(score_answer("sql", TASK, sql, self.db_dir), False)

    def test_empty_sql_scores_false(self):
        self.assertEqual(score_answer("sql", TASK, "", self.db_dir), False)

    def test_invalid_sql_scores_false(self):
        self.assertEqual(score_answer("sql", TASK, "SELEC name FROM", self.db_dir), False)

    def test_missing_database_is_blank(self):
        task = dict(TASK, db_id="not_here")
        self.assertEqual(score_answer("sql", task, GOLD_SQL, self.db_dir), "")

    def test_no_db_dir_is_blank(self):
        self.assertEqual(score_answer("sql", TASK, GOLD_SQL, None), "")

    def test_task_without_gold_is_blank(self):
        task = {"question_id": 1, "difficulty": "simple"}
        self.assertEqual(score_answer("sql", task, GOLD_SQL, self.db_dir), "")

    def test_rows_from_trace_scores_every_row(self):
        rows = rows_from_trace("sql", TASK, fake_trace(GOLD_SQL), db_dir=self.db_dir)
        self.assertEqual([r["is_correct"] for r in rows], [True, True])

    def test_rows_from_trace_marks_wrong_answer(self):
        trace = fake_trace("SELECT name FROM people WHERE id = 2")
        rows = rows_from_trace("sql", TASK, trace, db_dir=self.db_dir)
        self.assertEqual([r["is_correct"] for r in rows], [False, False])


if __name__ == "__main__":
    unittest.main()
