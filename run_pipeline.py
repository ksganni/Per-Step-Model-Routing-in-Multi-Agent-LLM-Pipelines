"""Run one task through a pipeline with real model calls.

This file wires the model client (``harness.call``) into both the SQL
pipeline (Selector -> Decomposer -> Refiner) and the QA pipeline
(Decomposer -> Hop Reader -> Composer -> Formatter). The model config
says which model serves each step: one name for every step, or a dict
keyed by step name.

From Python:

    from run_pipeline import run_pipeline, run_trace

    # SQL — same model for all steps
    sql = run_pipeline("sql", 1471, "flash")

    # SQL — different model per step
    sql = run_pipeline("sql", 1471, {"sql.selector": "flash-lite",
                                     "sql.decomposer": "flash",
                                     "sql.refiner": "flash"})

    # QA — same model for all steps
    answer = run_pipeline("qa", "5a83880e554299123d8c214e", "flash")

    # Full trace with per-step metadata
    trace = run_trace("sql", 1471, "flash")
    trace = run_trace("qa", "5a83880e554299123d8c214e", "flash")

From the repository root:

    python run_pipeline.py --pipeline sql --qid 1471 --model flash
    python run_pipeline.py --pipeline qa --qid 5a83880e554299123d8c214e --model flash

When the BIRD databases are in ``data/bird/`` (``python -m
pipelines.download_data``), the SQL is executed and the refiner runs.
Without them the pipeline stops after the decomposer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from harness.client import MODELS, call
from pipelines.qa import steps as qa_steps
from pipelines.sql import steps as sql_steps

ROOT = Path(__file__).resolve().parent
MINIDEV_FILE = ROOT / "testkit" / "mini_dev_prompt.jsonl"
DB_DIR = ROOT / "data" / "bird" / "minidev" / "MINIDEV" / "dev_databases"
HOTPOT_FILE = ROOT / "data" / "hotpot_sample_150.json"

SQL_STEPS = (sql_steps.STEP_SELECTOR, sql_steps.STEP_DECOMPOSER, sql_steps.STEP_REFINER)
QA_STEPS = (qa_steps.STEP_DECOMPOSER, qa_steps.STEP_HOP, qa_steps.STEP_COMPOSER, qa_steps.STEP_FORMATTER)


def models_per_step(models: str | dict[str, str], steps: tuple[str, ...]) -> dict[str, str]:
    """Expand the model config into one model name per step.

    Args:
        models: A model name used for every step, or a dict with an entry
            for every step.
        steps: The step names of the pipeline.
    """
    plan = dict.fromkeys(steps, models) if isinstance(models, str) else dict(models)
    missing = [s for s in steps if s not in plan]
    extra = [s for s in plan if s not in steps]
    if missing or extra:
        raise ValueError(f"Model config must name exactly the steps {list(steps)}; "
                         f"missing {missing}, unknown {extra}")
    unknown = sorted({m for m in plan.values() if m not in MODELS})
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Use one of: {', '.join(MODELS)}")
    return plan


# --------------------------------------------------------------------------- loaders


def load_sql_task(question: int | str | dict) -> dict:
    """Return a BIRD Mini-Dev task with its schema.

    ``question`` is a question_id, or a task dict. A dict without a
    ``schema`` field (for example a row of data/bird_sample_150.json) is
    completed from testkit/mini_dev_prompt.jsonl.
    """
    if isinstance(question, dict) and "schema" in question:
        return question
    qid = int(question["question_id"] if isinstance(question, dict) else question)
    with MINIDEV_FILE.open(encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            if task["question_id"] == qid:
                return task
    raise KeyError(f"question_id {qid} is not in {MINIDEV_FILE.name}")


def _normalise_hotpot(rec: dict) -> dict:
    """Normalise a HotpotQA record to the row-based format that from_hotpot expects.

    Our sample file (hotpot_sample_150.json) stores context and supporting_facts
    as columnar dicts (``{"title": [...], "sentences": [...]}``) while the
    original HotpotQA dataset stores them as lists of pairs
    (``[[title, sentences], ...]``). This converts the columnar form and also
    normalises ``id`` → ``_id``.
    """
    rec = dict(rec)
    if "_id" not in rec and "id" in rec:
        rec["_id"] = rec["id"]
    ctx = rec.get("context")
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        rec["context"] = list(zip(ctx["title"], ctx["sentences"]))
    sf = rec.get("supporting_facts")
    if isinstance(sf, dict) and "title" in sf and "sent_id" in sf:
        rec["supporting_facts"] = list(zip(sf["title"], sf["sent_id"]))
    return rec


def load_qa_task(question: str | dict) -> dict:
    """Return a HotpotQA task in the pipeline's format.

    ``question`` is a HotpotQA ``_id`` string, or a raw HotpotQA record
    dict. A dict with a ``task_id`` field (already converted) is returned
    as-is.
    """
    if isinstance(question, dict):
        if "task_id" in question:
            return question
        if "_id" in question or "id" in question:
            return qa_steps.from_hotpot(_normalise_hotpot(question))
    qid = str(question)
    with HOTPOT_FILE.open(encoding="utf-8") as f:
        for rec in json.load(f):
            # The full HotpotQA dataset uses "_id"; our sample uses "id".
            rid = rec.get("_id") or rec.get("id")
            if rid == qid:
                return qa_steps.from_hotpot(_normalise_hotpot(rec))
    raise KeyError(f"_id {qid!r} is not in {HOTPOT_FILE.name}")


# --------------------------------------------------------------------------- traces


def run_sql_trace(question: int | str | dict, models: str | dict[str, str],
                  db_dir: str | Path | None = DB_DIR) -> dict:
    """Run the SQL pipeline and return the full trace.

    Every step's model call is added to the trace under ``calls`` with
    per-step metadata: tokens_in, tokens_out, cost, time_s, raw_output.
    """
    task = load_sql_task(question)
    plan = models_per_step(models, SQL_STEPS)
    calls: list[dict] = []

    def llm(step: str, prompt: str) -> str:
        t0 = time.monotonic()
        response = call(plan[step], [{"role": "user", "content": prompt}])
        elapsed = time.monotonic() - t0
        calls.append({
            "step": step,
            "model": response.model,
            "tokens_in": response.prompt_tokens,
            "tokens_out": response.completion_tokens,
            "cost": response.cost,
            "time_s": round(elapsed, 3),
            "cached": response.cached,
            "raw_output": response.text,
        })
        return response.text

    db_path = None
    if db_dir is not None:
        candidate = Path(db_dir) / task["db_id"] / f"{task['db_id']}.sqlite"
        db_path = candidate if candidate.is_file() else None

    trace = sql_steps.run(task, llm, db_path)
    trace["models"] = plan
    trace["calls"] = calls
    trace["total_cost"] = sum(c["cost"] for c in calls)
    trace["total_tokens_in"] = sum(c["tokens_in"] for c in calls)
    trace["total_tokens_out"] = sum(c["tokens_out"] for c in calls)
    trace["total_time_s"] = round(sum(c["time_s"] for c in calls), 3)
    return trace


def run_qa_trace(question: str | dict, models: str | dict[str, str]) -> dict:
    """Run the QA pipeline and return the full trace.

    Every step's model call is added to the trace under ``calls`` with
    per-step metadata: tokens_in, tokens_out, cost, time_s, raw_output.
    """
    task = load_qa_task(question)
    plan = models_per_step(models, QA_STEPS)
    calls: list[dict] = []

    def llm(step: str, prompt: str) -> str:
        t0 = time.monotonic()
        response = call(plan[step], [{"role": "user", "content": prompt}])
        elapsed = time.monotonic() - t0
        calls.append({
            "step": step,
            "model": response.model,
            "tokens_in": response.prompt_tokens,
            "tokens_out": response.completion_tokens,
            "cost": response.cost,
            "time_s": round(elapsed, 3),
            "cached": response.cached,
            "raw_output": response.text,
        })
        return response.text

    trace = qa_steps.run(task, llm)
    trace["models"] = plan
    trace["calls"] = calls
    trace["total_cost"] = sum(c["cost"] for c in calls)
    trace["total_tokens_in"] = sum(c["tokens_in"] for c in calls)
    trace["total_tokens_out"] = sum(c["tokens_out"] for c in calls)
    trace["total_time_s"] = round(sum(c["time_s"] for c in calls), 3)
    return trace


def run_trace(pipeline: str, question, models: str | dict[str, str], **kwargs) -> dict:
    """Run a pipeline and return its full trace with per-step metadata."""
    if pipeline == "sql":
        return run_sql_trace(question, models, **kwargs)
    if pipeline == "qa":
        return run_qa_trace(question, models)
    raise ValueError(f"Unknown pipeline {pipeline!r}. Use: sql, qa")


# --------------------------------------------------------------------------- convenience


def run_pipeline(pipeline: str, question, models: str | dict[str, str],
                 db_dir: str | Path | None = DB_DIR) -> str:
    """Run every step of a pipeline and return the final answer.

    Args:
        pipeline: ``sql`` or ``qa``.
        question: A BIRD Mini-Dev question_id (int) for SQL, or a
            HotpotQA ``_id`` string for QA, or a task dict.
        models: ``flash``, ``flash-lite`` or ``qwen`` for every step, or a
            dict mapping each step name to one of them.
        db_dir: (SQL only) Folder with ``<db_id>/<db_id>.sqlite``. The
            refiner runs only when the task's database is found there.

    Returns:
        The final SQL query (SQL pipeline) or the final answer string
        (QA pipeline).
    """
    if pipeline == "sql":
        return run_sql_trace(question, models, db_dir)["final_sql"]
    if pipeline == "qa":
        return run_qa_trace(question, models)["final_answer"]
    raise ValueError(f"Unknown pipeline {pipeline!r}. Use: sql, qa")


def _print_trace(trace: dict) -> None:
    """Pretty-print a trace's per-step metadata."""
    for c in trace["calls"]:
        print(f"{c['step']:<16} {c['model']:<12} {c['tokens_in']:>6} in "
              f"{c['tokens_out']:>5} out  ${c['cost']:.6f}  "
              f"{c['time_s']:>6.2f}s  cached={c['cached']}")
    print(f"{'':16} {'TOTAL':<12} {trace['total_tokens_in']:>6} in "
          f"{trace['total_tokens_out']:>5} out  ${trace['total_cost']:.6f}  "
          f"{trace['total_time_s']:>6.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one question through the SQL or QA pipeline.")
    parser.add_argument("--pipeline", default="sql", choices=["sql", "qa"],
                        help="which pipeline to run (default: sql)")
    parser.add_argument("--qid", required=True,
                        help="BIRD question_id (int) for SQL, or HotpotQA _id for QA")
    parser.add_argument("--model", default="flash", choices=list(MODELS),
                        help="model used for every step (default: flash)")
    parser.add_argument("--db-dir", default=str(DB_DIR),
                        help="(SQL only) folder with <db_id>/<db_id>.sqlite")
    args = parser.parse_args()

    if args.pipeline == "sql":
        trace = run_sql_trace(int(args.qid), args.model, args.db_dir)
        _print_trace(trace)
        print(f"stop: {trace['stop_reason']}")
        print(f"SQL:  {trace['final_sql']}")
    else:
        trace = run_qa_trace(args.qid, args.model)
        _print_trace(trace)
        print(f"answer: {trace['final_answer']}")


if __name__ == "__main__":
    main()