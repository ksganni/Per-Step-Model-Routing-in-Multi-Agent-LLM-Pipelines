"""Run the sample tasks through a pipeline and log one CSV row per step.

Reads data/bird_sample_150.json or data/hotpot_sample_150.json, applies a
routing config (which model serves each step), calls run_pipeline on every
question, and appends rows to experiments/results.csv. If the script stops
partway, a later run skips question_ids that are already in the CSV.

CSV columns: question_id, pipeline, difficulty, step, model, tokens_in,
tokens_out, cost, time_sec, raw_output, final_answer, is_correct,
error_propagated.

``is_correct`` is True or False from harness.verifiers, or blank when the
answer cannot be scored (no database folder, or the gold query fails).

``error_propagated`` is filled only when --baseline points at the results
CSV of an all-strong run of the same questions. For a config with one weak
step it is True when the baseline answered correctly, this run answered
incorrectly, and the weak step's output differs from the baseline's (see
harness/error_propagation.py). Without --baseline it is left blank.

Resume skips question_ids already in the results file, so each config needs
its own --results file.

From the repository root:

    python experiment_runner.py --pipeline sql --config all-flash --limit 10

Error propagation for a cheap Selector, on the first 5 BIRD questions:

    python experiment_runner.py --pipeline sql --config all-flash --limit 5 \
        --results experiments/results-all-flash.csv
    python experiment_runner.py --pipeline sql --config cheap-selector --limit 5 \
        --results experiments/results-cheap-selector.csv \
        --baseline experiments/results-all-flash.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from harness.error_propagation import error_propagated as judge_propagation
from harness.verifiers import verify_qa, verify_sql
from run_pipeline import DB_DIR, run_sql_trace

ROOT = Path(__file__).resolve().parent
BIRD_SAMPLE = ROOT / "data" / "bird_sample_150.json"
HOTPOT_SAMPLE = ROOT / "data" / "hotpot_sample_150.json"
CONFIG_DIR = ROOT / "experiments" / "configs"
DEFAULT_RESULTS = ROOT / "experiments" / "results.csv"

CSV_COLUMNS = [
    "question_id",
    "pipeline",
    "difficulty",
    "step",
    "model",
    "tokens_in",
    "tokens_out",
    "cost",
    "time_sec",
    "raw_output",
    "final_answer",
    "is_correct",
    "error_propagated",
]


def load_config(name: str) -> dict:
    """Load a routing config by name or path.

    A named config is read from experiments/configs/<name>.json. The file
    may be a single model name, a per-step dict, or a dict keyed by
    pipeline (``sql`` / ``qa``).
    """
    path = Path(name)
    if not path.is_file():
        path = CONFIG_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Config {name!r} not found. Put a JSON file at {CONFIG_DIR / (name + '.json')} "
            f"or pass a path to one."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def models_for_pipeline(config: object, pipeline: str) -> str | dict[str, str]:
    """Pick the model plan for one pipeline from a loaded config."""
    if isinstance(config, str):
        return config
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a model name or a JSON object, got {type(config).__name__}")
    if pipeline in config:
        plan = config[pipeline]
        if not isinstance(plan, (str, dict)):
            raise ValueError(f"Config[{pipeline!r}] must be a model name or a per-step object")
        return plan
    # Per-step dict with no pipeline key, e.g. {"sql.selector": "flash", ...}.
    if any(key.startswith(f"{pipeline}.") for key in config):
        return config
    raise ValueError(
        f"Config has no entry for pipeline {pipeline!r}. "
        f"Use a string, a per-step object, or keys sql/qa."
    )


def load_tasks(pipeline: str) -> list[dict]:
    """Return the sample tasks for a pipeline, in file order."""
    if pipeline == "sql":
        return json.loads(BIRD_SAMPLE.read_text(encoding="utf-8"))
    if pipeline == "qa":
        return json.loads(HOTPOT_SAMPLE.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown pipeline {pipeline!r}. Use: sql, qa")


def task_id(pipeline: str, task: dict) -> str:
    """Stable id used in the CSV and for resume."""
    if pipeline == "sql":
        return str(task["question_id"])
    return str(task.get("id") or task.get("_id"))


def task_difficulty(pipeline: str, task: dict) -> str:
    """Difficulty for SQL; HotpotQA type (bridge/comparison) for QA."""
    if pipeline == "sql":
        return str(task.get("difficulty", ""))
    return str(task.get("type", ""))


def completed_ids(results_path: Path, pipeline: str) -> set[str]:
    """Return question_ids that already have at least one row for this pipeline."""
    if not results_path.is_file():
        return set()
    done: set[str] = set()
    with results_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("pipeline") == pipeline and row.get("question_id"):
                done.add(row["question_id"])
    return done


def append_rows(results_path: Path, rows: list[dict]) -> None:
    """Append rows to the CSV, writing the header when the file is new.

    An existing file keeps its own header, so appending to a results file
    written before a column was added does not shift the columns.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results_path.is_file() or results_path.stat().st_size == 0
    fieldnames = CSV_COLUMNS
    if not new_file:
        with results_path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), None)
        if header:
            fieldnames = header
    with results_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def load_baseline(baseline_path: Path, pipeline: str) -> dict[str, list[dict]]:
    """Read an all-strong results CSV into {question_id: [step rows]}."""
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline {baseline_path} not found. Run the all-strong config first "
            f"with --results {baseline_path}."
        )
    baseline: dict[str, list[dict]] = {}
    with baseline_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("pipeline") == pipeline and row.get("question_id"):
                baseline.setdefault(row["question_id"], []).append(row)
    return baseline


def score_answer(pipeline: str, task: dict, final_answer: str,
                 db_dir: str | Path | None = DB_DIR) -> bool | str:
    """Score a final answer against the gold answer.

    Returns True or False, or "" when the answer cannot be scored.
    """
    if pipeline == "sql":
        if not final_answer:
            return False
        db_id = task.get("db_id")
        if db_dir is None or not db_id or not task.get("SQL"):
            return ""
        db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            return ""
        try:
            return verify_sql(str(db_path), final_answer, task["SQL"])
        except Exception:
            return ""
    if not task.get("answer"):
        return ""
    return verify_qa(final_answer or "", task["answer"])


def rows_from_trace(pipeline: str, task: dict, trace: dict,
                    db_dir: str | Path | None = DB_DIR) -> list[dict]:
    """Turn one pipeline trace into CSV rows (one per model call)."""
    qid = task_id(pipeline, task)
    difficulty = task_difficulty(pipeline, task)
    final_answer = trace.get("final_sql") if pipeline == "sql" else trace.get("final_answer", "")
    is_correct = score_answer(pipeline, task, final_answer or "", db_dir)
    rows = []
    for call in trace.get("calls", []):
        rows.append({
            "question_id": qid,
            "pipeline": pipeline,
            "difficulty": difficulty,
            "step": call["step"],
            "model": call["model"],
            "tokens_in": call.get("tokens_in", call.get("prompt_tokens", 0)),
            "tokens_out": call.get("tokens_out", call.get("completion_tokens", 0)),
            "cost": f"{float(call.get('cost', 0.0)):.8f}",
            "time_sec": f"{float(call.get('time_s', call.get('time_sec', 0.0))):.4f}",
            "raw_output": call.get("raw_output", ""),
            "final_answer": final_answer or "",
            "is_correct": is_correct,
            "error_propagated": "",
        })
    return rows


def run_one(pipeline: str, task: dict, models: str | dict[str, str],
            db_dir: str | Path | None = DB_DIR) -> dict:
    """Run one question and return the full trace."""
    if pipeline == "sql":
        return run_sql_trace(task, models, db_dir=db_dir)
    raise ValueError(
        "QA is not wired into run_pipeline yet. Use --pipeline sql for now, "
        "or wait for the QA extension of run_pipeline."
    )


def run_experiment(
    pipeline: str,
    config_name: str,
    *,
    limit: int | None = None,
    results_path: Path = DEFAULT_RESULTS,
    db_dir: str | Path | None = DB_DIR,
    baseline_path: Path | None = None,
) -> int:
    """Run the sample tasks and append step rows to the results CSV.

    With ``baseline_path``, every question's rows also get error_propagated,
    judged against that question's rows in the baseline CSV.

    Returns the number of questions that were run in this invocation.
    """
    models = models_for_pipeline(load_config(config_name), pipeline)
    baseline = None
    if baseline_path is not None:
        if Path(baseline_path).resolve() == Path(results_path).resolve():
            raise ValueError("--baseline must be a different file from --results.")
        baseline = load_baseline(Path(baseline_path), pipeline)
    propagation_counts = {"True": 0, "False": 0, "blank": 0}
    tasks = load_tasks(pipeline)
    if limit is not None:
        tasks = tasks[:limit]

    done = completed_ids(results_path, pipeline)
    ran = 0
    for task in tasks:
        qid = task_id(pipeline, task)
        if qid in done:
            print(f"skip {pipeline} {qid} (already in {results_path.name})")
            continue
        print(f"run  {pipeline} {qid} ...")
        try:
            trace = run_one(pipeline, task, models, db_dir=db_dir)
        except Exception as exc:
            # Leave this question out of the CSV so a later run retries it.
            print(f"      failed {qid}: {type(exc).__name__}: {exc}")
            continue
        rows = rows_from_trace(pipeline, task, trace, db_dir=db_dir)
        if not rows:
            # Still record a placeholder so resume does not retry forever
            # when the pipeline made no model calls.
            rows = [{
                "question_id": qid,
                "pipeline": pipeline,
                "difficulty": task_difficulty(pipeline, task),
                "step": "",
                "model": "",
                "tokens_in": 0,
                "tokens_out": 0,
                "cost": "0.00000000",
                "time_sec": "0.0000",
                "raw_output": "",
                "final_answer": (
                    trace.get("final_sql") if pipeline == "sql"
                    else trace.get("final_answer", "")
                ) or "",
                "is_correct": "",
                "error_propagated": "",
            }]
        if baseline is not None:
            propagated = judge_propagation(rows, baseline.get(qid, []))
            for row in rows:
                row["error_propagated"] = propagated
            propagation_counts[str(propagated) if propagated != "" else "blank"] += 1
        append_rows(results_path, rows)
        ran += 1
        done.add(qid)
        print(f"      {len(rows)} step row(s), stop={trace.get('stop_reason', '')}")
    if baseline is not None:
        print("error_propagated: "
              + ", ".join(f"{k}={v}" for k, v in propagation_counts.items()))
    return ran


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sample tasks through a pipeline and log step results to CSV."
    )
    parser.add_argument("--pipeline", required=True, choices=["sql", "qa"],
                        help="which pipeline to run")
    parser.add_argument("--config", required=True,
                        help="config name under experiments/configs/ or a JSON path")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N sample questions")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS),
                        help="CSV path (default: experiments/results.csv)")
    parser.add_argument("--db-dir", default=str(DB_DIR),
                        help="folder with <db_id>/<db_id>.sqlite (enables the SQL refiner)")
    parser.add_argument("--baseline", default=None,
                        help="results CSV of an all-strong run of the same questions; "
                             "fills error_propagated")
    args = parser.parse_args()

    db_dir: str | Path | None = args.db_dir
    if db_dir == "":
        db_dir = None

    ran = run_experiment(
        args.pipeline,
        args.config,
        limit=args.limit,
        results_path=Path(args.results),
        db_dir=db_dir,
        baseline_path=Path(args.baseline) if args.baseline else None,
    )
    print(f"done: ran {ran} question(s) -> {args.results}")


if __name__ == "__main__":
    main()
