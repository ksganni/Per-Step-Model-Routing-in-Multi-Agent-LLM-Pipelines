"""Offline tests for run_pipeline.

These tests replace ``litellm.completion`` with fake replies, so they need
no API key, no GPU, no network, and no BIRD databases.

From the repository root:

    python -m unittest discover -s harness -t . -p "test_*.py"
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from harness.test_harness import HarnessTestCase, fake_response
from run_pipeline import (
    load_qa_task,
    load_sql_task,
    run_pipeline,
    run_qa_trace,
    run_sql_trace,
    run_trace,
)

QID = 1471  # debit_card_specializing
SELECTOR_REPLY = '```json\n{"customers": "keep_all"}\n```'
BAD_SQL_REPLY = "Sub question 1: ratio?\n```sql\nSELECT missing_column FROM customers\n```"
GOOD_SQL_REPLY = "Sub question 1: ratio?\n```sql\nSELECT COUNT(*) FROM customers\n```"

# QA fake replies
QA_DECOMPOSER_REPLY = '```json\n{"question_type": "bridge", "sub_questions": ["Who directed Jaws?", "When was #1 born?"]}\n```'
QA_HOP_REPLY_1 = '```json\n{"answer": "Steven Spielberg", "found": true, "evidence": []}\n```'
QA_HOP_REPLY_2 = '```json\n{"answer": "December 18, 1946", "found": true, "evidence": []}\n```'
QA_COMPOSER_REPLY = '```json\n{"answer": "Steven Spielberg was born on December 18, 1946", "reasoning": "Combined hop answers"}\n```'
QA_FORMATTER_REPLY = '```json\n{"final_answer": "December 18, 1946"}\n```'

# A minimal HotpotQA task (already converted to pipeline format)
FAKE_QA_TASK = {
    "task_id": "test_qa_001",
    "question": "When was the director of Jaws born?",
    "type": "bridge",
    "level": "medium",
    "paragraphs": [
        {"title": "Jaws (film)", "sentences": ["Jaws is a 1975 film directed by Steven Spielberg."]},
        {"title": "Steven Spielberg", "sentences": ["Steven Spielberg was born on December 18, 1946."]},
        {"title": "Decoy paragraph", "sentences": ["This paragraph is not relevant to the question."]},
    ],
    "gold_answer": "December 18, 1946",
    "supporting_facts": [["Jaws (film)", 0], ["Steven Spielberg", 0]],
}


class SqlPipelineTests(HarnessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["GEMINI_API_KEY"] = "test-key"

    def replies(self, *texts):
        self.completion.side_effect = [fake_response(t) for t in texts]

    def test_all_flash_returns_sql(self):
        self.replies(SELECTOR_REPLY, GOOD_SQL_REPLY)
        sql = run_pipeline("sql", QID, "flash", db_dir=None)
        self.assertEqual(sql, "SELECT COUNT(*) FROM customers")
        models = [c.kwargs["model"] for c in self.completion.call_args_list]
        self.assertEqual(models, ["gemini/gemini-3.5-flash"] * 2)

    def test_per_step_models_are_used(self):
        self.replies(SELECTOR_REPLY, GOOD_SQL_REPLY)
        trace = run_sql_trace(QID, {"sql.selector": "flash-lite", "sql.decomposer": "flash",
                                    "sql.refiner": "flash"}, db_dir=None)
        self.assertEqual([c["model"] for c in trace["calls"]], ["flash-lite", "flash"])
        self.assertEqual([c["step"] for c in trace["calls"]], ["sql.selector", "sql.decomposer"])

    def test_refiner_runs_against_the_database(self):
        task = load_sql_task(QID)
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / task["db_id"] / f"{task['db_id']}.sqlite"
            db.parent.mkdir()
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE customers (CustomerID INTEGER, Currency TEXT)")
                conn.execute("INSERT INTO customers VALUES (1, 'EUR')")
            conn.close()
            self.replies(SELECTOR_REPLY, BAD_SQL_REPLY, GOOD_SQL_REPLY)
            trace = run_sql_trace(QID, "flash", db_dir=folder)
        self.assertEqual(trace["final_sql"], "SELECT COUNT(*) FROM customers")
        self.assertEqual([c["step"] for c in trace["calls"]],
                         ["sql.selector", "sql.decomposer", "sql.refiner"])
        self.assertEqual(trace["refines"], 1)

    def test_no_sql_returns_empty_string(self):
        self.replies(SELECTOR_REPLY, "I cannot answer this.")
        self.assertEqual(run_pipeline("sql", QID, "flash", db_dir=None), "")

    def test_sample_row_without_schema_is_completed(self):
        task = load_sql_task({"question_id": QID, "question": "ignored"})
        self.assertIn("CREATE TABLE", task["schema"])

    def test_bad_config_is_rejected(self):
        with self.assertRaises(ValueError):
            run_pipeline("sql", QID, "gpt")
        with self.assertRaises(ValueError):
            run_pipeline("sql", QID, {"sql.selector": "flash"})
        self.completion.assert_not_called()

    def test_sql_trace_has_metadata(self):
        """Each call in the trace has tokens_in, tokens_out, cost, time_s, raw_output."""
        self.replies(SELECTOR_REPLY, GOOD_SQL_REPLY)
        trace = run_sql_trace(QID, "flash", db_dir=None)
        for c in trace["calls"]:
            self.assertIn("tokens_in", c)
            self.assertIn("tokens_out", c)
            self.assertIn("cost", c)
            self.assertIn("time_s", c)
            self.assertIn("raw_output", c)
            self.assertIsInstance(c["cost"], float)
            self.assertIsInstance(c["time_s"], float)
        self.assertIn("total_cost", trace)
        self.assertIn("total_tokens_in", trace)
        self.assertIn("total_tokens_out", trace)
        self.assertIn("total_time_s", trace)


class QaPipelineTests(HarnessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["GEMINI_API_KEY"] = "test-key"

    def replies(self, *texts):
        self.completion.side_effect = [fake_response(t) for t in texts]

    def test_qa_returns_final_answer(self):
        """QA pipeline runs all 4 steps (decomposer, 2 hops, composer, formatter)."""
        self.replies(QA_DECOMPOSER_REPLY, QA_HOP_REPLY_1, QA_HOP_REPLY_2,
                     QA_COMPOSER_REPLY, QA_FORMATTER_REPLY)
        answer = run_pipeline("qa", FAKE_QA_TASK, "flash")
        self.assertEqual(answer, "December 18, 1946")

    def test_qa_trace_has_all_steps(self):
        self.replies(QA_DECOMPOSER_REPLY, QA_HOP_REPLY_1, QA_HOP_REPLY_2,
                     QA_COMPOSER_REPLY, QA_FORMATTER_REPLY)
        trace = run_qa_trace(FAKE_QA_TASK, "flash")
        steps = [c["step"] for c in trace["calls"]]
        self.assertEqual(steps, ["qa.decomposer", "qa.hop_reader", "qa.hop_reader",
                                 "qa.composer", "qa.formatter"])

    def test_qa_trace_has_metadata(self):
        """Each call in the QA trace has tokens_in, tokens_out, cost, time_s, raw_output."""
        self.replies(QA_DECOMPOSER_REPLY, QA_HOP_REPLY_1, QA_HOP_REPLY_2,
                     QA_COMPOSER_REPLY, QA_FORMATTER_REPLY)
        trace = run_qa_trace(FAKE_QA_TASK, "flash")
        for c in trace["calls"]:
            self.assertIn("tokens_in", c)
            self.assertIn("tokens_out", c)
            self.assertIn("cost", c)
            self.assertIn("time_s", c)
            self.assertIn("raw_output", c)
        self.assertIn("total_cost", trace)
        self.assertIn("total_tokens_in", trace)
        self.assertIn("total_tokens_out", trace)
        self.assertIn("total_time_s", trace)

    def test_qa_per_step_models(self):
        self.replies(QA_DECOMPOSER_REPLY, QA_HOP_REPLY_1, QA_HOP_REPLY_2,
                     QA_COMPOSER_REPLY, QA_FORMATTER_REPLY)
        trace = run_qa_trace(FAKE_QA_TASK, {
            "qa.decomposer": "flash-lite",
            "qa.hop_reader": "flash",
            "qa.composer": "flash",
            "qa.formatter": "flash-lite",
        })
        models = [c["model"] for c in trace["calls"]]
        self.assertEqual(models, ["flash-lite", "flash", "flash", "flash", "flash-lite"])

    def test_qa_bad_config_is_rejected(self):
        with self.assertRaises(ValueError):
            run_pipeline("qa", FAKE_QA_TASK, "gpt")
        with self.assertRaises(ValueError):
            run_pipeline("qa", FAKE_QA_TASK, {"qa.decomposer": "flash"})
        self.completion.assert_not_called()

    def test_run_trace_dispatches(self):
        """run_trace() dispatches to the right pipeline."""
        self.replies(SELECTOR_REPLY, GOOD_SQL_REPLY)
        trace = run_trace("sql", QID, "flash", db_dir=None)
        self.assertEqual(trace["pipeline"], "sql")

        self.replies(QA_DECOMPOSER_REPLY, QA_HOP_REPLY_1, QA_HOP_REPLY_2,
                     QA_COMPOSER_REPLY, QA_FORMATTER_REPLY)
        trace = run_trace("qa", FAKE_QA_TASK, "flash")
        self.assertEqual(trace["pipeline"], "qa")

    def test_unknown_pipeline_is_rejected(self):
        with self.assertRaises(ValueError):
            run_pipeline("summarize", "anything", "flash")

    def test_load_qa_task_from_file(self):
        """load_qa_task can find a task by _id from hotpot_sample_150.json."""
        import json
        hotpot = Path(__file__).resolve().parent.parent / "data" / "hotpot_sample_150.json"
        if hotpot.is_file():
            with hotpot.open() as f:
                first = json.load(f)[0]
            rid = first.get("_id") or first.get("id")
            task = load_qa_task(rid)
            self.assertEqual(task["task_id"], rid)
            self.assertIn("question", task)
            self.assertIn("paragraphs", task)


if __name__ == "__main__":
    unittest.main()