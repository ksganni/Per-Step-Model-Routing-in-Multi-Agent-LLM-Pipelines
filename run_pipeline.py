"""Run one task through a pipeline with real model calls.

This file wires the model client (``harness.call``) into the SQL pipeline
(Selector -> Decomposer -> Refiner). The model config says which model
serves each step: one name for every step, or a dict keyed by step name.

From Python:

    from run_pipeline import run_pipeline
    sql = run_pipeline("sql", 1471, "flash")
    sql = run_pipeline("sql", 1471, {"sql.selector": "flash-lite",
                                     "sql.decomposer": "flash",
                                     "sql.refiner": "flash"})

From the repository root:

    python run_pipeline.py --qid 1471 --model flash

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
from pipelines.sql import steps as sql_steps

ROOT = Path(__file__).resolve().parent
MINIDEV_FILE = ROOT / "testkit" / "mini_dev_prompt.jsonl"
DB_DIR = ROOT / "data" / "bird" / "minidev" / "MINIDEV" / "dev_databases"

SQL_STEPS = (sql_steps.STEP_SELECTOR, sql_steps.STEP_DECOMPOSER, sql_steps.STEP_REFINER)


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


def run_sql_trace(question: int | str | dict, models: str | dict[str, str],
                  db_dir: str | Path | None = DB_DIR) -> dict:
    """Run the SQL pipeline and return the full trace.

    Every step's model call is added to the trace under ``calls``.
    """
    task = load_sql_task(question)
    plan = models_per_step(models, SQL_STEPS)
    calls: list[dict] = []

    def llm(step: str, prompt: str) -> str:
        started = time.perf_counter()
        response = call(plan[step], [{"role": "user", "content": prompt}])
        elapsed = time.perf_counter() - started
        calls.append({
            "step": step,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost": response.cost,
            "time_sec": elapsed,
            "raw_output": response.text,
            "cached": response.cached,
        })
        return response.text

    db_path = None
    if db_dir is not None:
        candidate = Path(db_dir) / task["db_id"] / f"{task['db_id']}.sqlite"
        db_path = candidate if candidate.is_file() else None

    trace = sql_steps.run(task, llm, db_path)
    trace["models"] = plan
    trace["calls"] = calls
    return trace


def run_pipeline(pipeline: str, question: int | str | dict, models: str | dict[str, str],
                 db_dir: str | Path | None = DB_DIR) -> str:
    """Run every step of a pipeline and return the final answer.

    Args:
        pipeline: Only ``sql`` is supported.
        question: A BIRD Mini-Dev question_id or task dict.
        models: ``flash``, ``flash-lite`` or ``qwen`` for every step, or a
            dict mapping each step name to one of them.
        db_dir: Folder with ``<db_id>/<db_id>.sqlite``. The refiner runs only
            when the task's database is found there.

    Returns:
        The final SQL query, or "" if the decomposer produced none.
    """
    if pipeline != "sql":
        raise ValueError(f"Unknown pipeline {pipeline!r}. Use: sql")
    return run_sql_trace(question, models, db_dir)["final_sql"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one BIRD question through the SQL pipeline.")
    parser.add_argument("--qid", type=int, required=True, help="BIRD Mini-Dev question_id")
    parser.add_argument("--model", default="flash", choices=list(MODELS),
                        help="model used for every step")
    parser.add_argument("--db-dir", default=str(DB_DIR),
                        help="folder with <db_id>/<db_id>.sqlite (enables the refiner)")
    args = parser.parse_args()

    trace = run_sql_trace(args.qid, args.model, args.db_dir)
    for c in trace["calls"]:
        print(f"{c['step']:<16} {c['model']:<10} {c['prompt_tokens']:>6} in "
              f"{c['completion_tokens']:>5} out  "
              f"${c['cost']:.6f}  {c['time_sec']:.2f}s  cached={c['cached']}")
    print(f"stop: {trace['stop_reason']}")
    print(f"SQL:  {trace['final_sql']}")


if __name__ == "__main__":
    main()
