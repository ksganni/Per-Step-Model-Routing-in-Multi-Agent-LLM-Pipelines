"""Merge experiment results and recommend a model for each pipeline step.

Reads the results CSVs written by experiment_runner (one row per model call,
or one summary row per question as run_qa_matrix writes them), merges them,
and produces three files under experiments/:

    all_results.csv     every input row, with one column layout
    config_summary.csv  one row per (pipeline, config): accuracy, cost,
                        error propagation, change vs the all-strong baseline
    summary.csv         one row per (pipeline, step, model option) with a
                        recommended flag on the option to use for that step

Routing configs are read from experiments/configs/<config>.json so a
config's weak step is known even when its rows carry no step column (QA
question rows). When no JSON exists for a config, the weak step is taken
from the model column of its step rows, if it has any.

Recommendation rule, per step: among the options that were run and scored,
keep those whose accuracy is within --tolerance percentage points of the
all-strong baseline, then pick the one with the lowest cost per question.
Ties go to the baseline. The tolerance defaults to 5 points: with 150
questions one question is 0.67 points and the sampling noise is about 4
points, so smaller differences are not evidence of a real change.

Two measures of error propagation appear in the output:

    propagation_rate    share of judged questions where error_propagated is
                        True (harness/error_propagation.py). Only step rows
                        run with --baseline carry this column, so it is
                        blank for the QA matrix.
    right_to_wrong_rate share of questions the baseline got right that this
                        config got wrong. Computable from any rows, so it
                        covers both pipelines. It does not check whether the
                        weak step's output changed, so it is an upper bound
                        on propagation, not the same number.

From the repository root:

    python aggregate_results.py
    python aggregate_results.py --inputs "experiments/sql/*.csv" experiments/qa_results.csv
    python aggregate_results.py --tolerance 3 --strong flash
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import OrderedDict
from pathlib import Path

from experiment_runner import CONFIG_DIR, load_config, models_for_pipeline

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"

DEFAULT_INPUT_GLOBS = [
    "experiments/sql/*.csv",
    "experiments/qa/*.csv",
    "experiments/results*.csv",
    "experiments/sql_results*.csv",
    "experiments/qa_results*.csv",
]
OUTPUT_NAMES = {"all_results.csv", "config_summary.csv", "summary.csv"}

ALL_COLUMNS = [
    "config",
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
    "f1",
    "em",
    "error_propagated",
]

CONFIG_SUMMARY_COLUMNS = [
    "pipeline",
    "config",
    "weak_step",
    "weak_model",
    "is_baseline",
    "n_questions",
    "n_scored",
    "n_correct",
    "accuracy",
    "mean_f1",
    "em_rate",
    "total_cost",
    "cost_per_question",
    "total_tokens_in",
    "total_tokens_out",
    "total_time_sec",
    "n_propagation_judged",
    "n_propagated",
    "propagation_rate",
    "n_paired",
    "n_baseline_right",
    "n_right_to_wrong",
    "right_to_wrong_rate",
    "n_wrong_to_right",
    "delta_accuracy_pts",
    "delta_cost_pct",
]

SUMMARY_COLUMNS = [
    "pipeline",
    "step",
    "model",
    "config",
    "n_scored",
    "accuracy",
    "delta_accuracy_pts",
    "cost_per_question",
    "delta_cost_pct",
    "propagation_rate",
    "right_to_wrong_rate",
    "recommended",
    "reason",
]


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def config_from_filename(path: Path) -> str:
    """Config name implied by a results file name, e.g. results-all-flash.csv."""
    stem = path.stem
    for prefix in ("results-", "results_", "sql_results-", "sql_results_",
                   "qa_results-", "qa_results_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def read_results(paths: list[Path]) -> list[dict]:
    """Read results CSVs into rows with every column in ALL_COLUMNS.

    Missing columns become "". A blank config is filled from the file name.
    """
    rows: list[dict] = []
    for path in paths:
        fallback = config_from_filename(path)
        with path.open(encoding="utf-8", newline="") as f:
            for raw in csv.DictReader(f):
                if not raw.get("question_id"):
                    continue
                row = {col: (raw.get(col) or "") for col in ALL_COLUMNS}
                if not row["config"]:
                    row["config"] = fallback
                rows.append(row)
    return rows


def expand_inputs(patterns: list[str]) -> list[Path]:
    """Resolve globs and paths, skipping this script's own output files."""
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True)) or [pattern]
        for match in matches:
            path = Path(match)
            if not path.is_file() or path.name in OUTPUT_NAMES:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(path)
    return found


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _is_true(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _is_false(value: object) -> bool:
    return value is False or str(value).strip().lower() == "false"


def _scored(value: object) -> bool:
    return _is_true(value) or _is_false(value)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fmt(value: float | None, places: int) -> str:
    return "" if value is None else f"{value:.{places}f}"


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


# --------------------------------------------------------------------------
# Collapsing to one row per question
# --------------------------------------------------------------------------

def question_rows(rows: list[dict]) -> list[dict]:
    """Collapse step rows to one record per (pipeline, config, question_id).

    Cost, tokens and time are summed over the question's calls. is_correct,
    f1, em and error_propagated are the same on every call row, so the first
    non-blank value is kept. ``steps`` maps each step that ran to the model
    that served it (empty for question rows, which carry no step).
    """
    grouped: "OrderedDict[tuple, dict]" = OrderedDict()
    for row in rows:
        key = (row["pipeline"], row["config"], row["question_id"])
        rec = grouped.get(key)
        if rec is None:
            rec = {
                "pipeline": row["pipeline"],
                "config": row["config"],
                "question_id": row["question_id"],
                "difficulty": row["difficulty"],
                "is_correct": "",
                "f1": "",
                "em": "",
                "error_propagated": "",
                "cost": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "time_sec": 0.0,
                "steps": {},
            }
            grouped[key] = rec
        rec["cost"] += _float(row["cost"])
        rec["tokens_in"] += _int(row["tokens_in"])
        rec["tokens_out"] += _int(row["tokens_out"])
        rec["time_sec"] += _float(row["time_sec"])
        for col in ("is_correct", "f1", "em", "error_propagated", "difficulty"):
            if rec[col] == "" and row[col] != "":
                rec[col] = row[col]
        if row["step"] and row["step"] not in rec["steps"]:
            rec["steps"][row["step"]] = row["model"]
    return list(grouped.values())


# --------------------------------------------------------------------------
# Routing plans
# --------------------------------------------------------------------------

def plan_for(config_name: str, pipeline: str, config_dir: Path) -> dict[str, str] | str | None:
    """The model plan for one config and pipeline, or None if no JSON exists."""
    path = config_dir / f"{config_name}.json"
    if not path.is_file():
        return None
    try:
        return models_for_pipeline(load_config(str(path)), pipeline)
    except ValueError:
        return None


def weak_step_for(plan: dict[str, str] | str | None, steps_seen: dict[str, str],
                  strong: str) -> tuple[str, str]:
    """Return (weak_step, weak_model) for a config.

    "" for both when the config is all-strong; "multiple" as the step when
    more than one step is weak; "unknown" when nothing says which step the
    config changes.
    """
    if isinstance(plan, str):
        return ("", "") if plan == strong else ("all", plan)
    source = plan if isinstance(plan, dict) else steps_seen
    if not source:
        return ("unknown", "")
    weak = {step: model for step, model in source.items() if model != strong}
    if not weak:
        return ("", "")
    if len(weak) > 1:
        return ("multiple", ",".join(sorted(set(weak.values()))))
    (step, model), = weak.items()
    return (step, model)


# --------------------------------------------------------------------------
# Per-config summary
# --------------------------------------------------------------------------

def summarise_configs(qrows: list[dict], config_dir: Path, strong: str) -> list[dict]:
    """One summary record per (pipeline, config), baselines first."""
    by_config: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for rec in qrows:
        by_config.setdefault((rec["pipeline"], rec["config"]), []).append(rec)

    summaries: list[dict] = []
    for (pipeline, config), recs in by_config.items():
        plan = plan_for(config, pipeline, config_dir)
        steps_seen: dict[str, str] = {}
        for rec in recs:
            for step, model in rec["steps"].items():
                steps_seen.setdefault(step, model)
        weak_step, weak_model = weak_step_for(plan, steps_seen, strong)

        scored = [r for r in recs if _scored(r["is_correct"])]
        correct = [r for r in scored if _is_true(r["is_correct"])]
        f1_vals = [_float(r["f1"]) for r in scored if r["f1"] != ""]
        em_vals = [r for r in scored if r["em"] != ""]
        em_true = [r for r in em_vals if _is_true(r["em"])]
        judged = [r for r in recs if _scored(r["error_propagated"])]
        propagated = [r for r in judged if _is_true(r["error_propagated"])]
        total_cost = sum(r["cost"] for r in recs)

        summaries.append({
            "pipeline": pipeline,
            "config": config,
            "weak_step": weak_step,
            "weak_model": weak_model,
            "is_baseline": weak_step == "" and weak_model == "",
            "n_questions": len(recs),
            "n_scored": len(scored),
            "n_correct": len(correct),
            "accuracy": _rate(len(correct), len(scored)),
            "mean_f1": (sum(f1_vals) / len(f1_vals)) if f1_vals else None,
            "em_rate": _rate(len(em_true), len(em_vals)),
            "total_cost": total_cost,
            "cost_per_question": _rate_float(total_cost, len(recs)),
            "total_tokens_in": sum(r["tokens_in"] for r in recs),
            "total_tokens_out": sum(r["tokens_out"] for r in recs),
            "total_time_sec": sum(r["time_sec"] for r in recs),
            "n_propagation_judged": len(judged),
            "n_propagated": len(propagated),
            "propagation_rate": _rate(len(propagated), len(judged)),
            "n_paired": 0,
            "n_baseline_right": 0,
            "n_right_to_wrong": 0,
            "right_to_wrong_rate": None,
            "n_wrong_to_right": 0,
            "delta_accuracy_pts": None,
            "delta_cost_pct": None,
            "_recs": recs,
        })

    _compare_to_baseline(summaries)
    summaries.sort(key=lambda s: (s["pipeline"], not s["is_baseline"], s["config"]))
    return summaries


def _rate_float(numerator: float, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def baseline_for(summaries: list[dict], pipeline: str) -> dict | None:
    """The all-strong summary for a pipeline, or None.

    When several configs are all-strong (say the same run under two names),
    the one with the most questions wins.
    """
    candidates = [s for s in summaries if s["pipeline"] == pipeline and s["is_baseline"]]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (s["n_questions"], s["config"]))


def _compare_to_baseline(summaries: list[dict]) -> None:
    """Fill the paired right-to-wrong counts and deltas against the baseline."""
    for summary in summaries:
        base = baseline_for(summaries, summary["pipeline"])
        if base is None or base is summary:
            continue
        base_by_qid = {r["question_id"]: r for r in base["_recs"]}
        paired = baseline_right = right_to_wrong = wrong_to_right = 0
        for rec in summary["_recs"]:
            other = base_by_qid.get(rec["question_id"])
            if other is None or not (_scored(rec["is_correct"]) and _scored(other["is_correct"])):
                continue
            paired += 1
            if _is_true(other["is_correct"]):
                baseline_right += 1
                if _is_false(rec["is_correct"]):
                    right_to_wrong += 1
            elif _is_true(rec["is_correct"]):
                wrong_to_right += 1
        summary["n_paired"] = paired
        summary["n_baseline_right"] = baseline_right
        summary["n_right_to_wrong"] = right_to_wrong
        summary["right_to_wrong_rate"] = _rate(right_to_wrong, baseline_right)
        summary["n_wrong_to_right"] = wrong_to_right
        if summary["accuracy"] is not None and base["accuracy"] is not None:
            summary["delta_accuracy_pts"] = 100.0 * (summary["accuracy"] - base["accuracy"])
        if (summary["cost_per_question"] is not None and base["cost_per_question"]):
            summary["delta_cost_pct"] = 100.0 * (
                summary["cost_per_question"] / base["cost_per_question"] - 1.0)


# --------------------------------------------------------------------------
# Per-step summary and recommendation
# --------------------------------------------------------------------------

def summarise_steps(config_summaries: list[dict], strong: str, tolerance: float) -> list[dict]:
    """One record per (pipeline, step, option), with one recommended per step."""
    out: list[dict] = []
    pipelines = sorted({s["pipeline"] for s in config_summaries})
    for pipeline in pipelines:
        base = baseline_for(config_summaries, pipeline)
        swaps = [s for s in config_summaries
                 if s["pipeline"] == pipeline and s["weak_step"] not in ("", "all", "multiple", "unknown")]
        steps = sorted({s["weak_step"] for s in swaps})
        for step in steps:
            options: list[dict] = []
            if base is not None:
                options.append(_option(pipeline, step, strong, base))
            for s in sorted((s for s in swaps if s["weak_step"] == step),
                            key=lambda s: s["config"]):
                options.append(_option(pipeline, step, s["weak_model"], s))
            _recommend(options, base is not None, tolerance)
            out.extend(options)
    return out


def _option(pipeline: str, step: str, model: str, summary: dict) -> dict:
    return {
        "pipeline": pipeline,
        "step": step,
        "model": model,
        "config": summary["config"],
        "n_scored": summary["n_scored"],
        "accuracy": summary["accuracy"],
        "delta_accuracy_pts": 0.0 if summary["is_baseline"] else summary["delta_accuracy_pts"],
        "cost_per_question": summary["cost_per_question"],
        "delta_cost_pct": 0.0 if summary["is_baseline"] else summary["delta_cost_pct"],
        "propagation_rate": summary["propagation_rate"],
        "right_to_wrong_rate": summary["right_to_wrong_rate"],
        "recommended": False,
        "reason": "",
        "_is_baseline": summary["is_baseline"],
    }


def _recommend(options: list[dict], have_baseline: bool, tolerance: float) -> None:
    """Mark exactly one option per step as recommended and explain each one."""
    if not have_baseline:
        for opt in options:
            opt["reason"] = "no all-strong baseline run; cannot compare"
        return
    runnable = [o for o in options if o["n_scored"] > 0 and o["accuracy"] is not None]
    for opt in options:
        if opt not in runnable:
            opt["reason"] = "not run or nothing scored"
    within = [o for o in runnable if o["delta_accuracy_pts"] is not None
              and o["delta_accuracy_pts"] >= -tolerance]
    for opt in runnable:
        if opt not in within:
            opt["reason"] = (f"accuracy {opt['delta_accuracy_pts']:+.1f} pts vs all-strong, "
                             f"beyond the {tolerance:g} pt tolerance")
    if not within:
        return
    best = min(within, key=lambda o: (
        o["cost_per_question"] if o["cost_per_question"] is not None else float("inf"),
        not o["_is_baseline"],
    ))
    best["recommended"] = True
    for opt in within:
        if opt is best:
            if opt["_is_baseline"]:
                opt["reason"] = "no cheaper option within tolerance"
            else:
                opt["reason"] = (f"within {tolerance:g} pts of all-strong "
                                 f"({opt['delta_accuracy_pts']:+.1f}) at lowest cost per question "
                                 f"({opt['delta_cost_pct']:+.1f}%)")
        elif opt["_is_baseline"]:
            opt["reason"] = f"a cheaper option is within tolerance ({best['model']})"
        else:
            opt["reason"] = (f"within tolerance ({opt['delta_accuracy_pts']:+.1f} pts) "
                             f"but not the cheapest ({best['model']} is)")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def _config_summary_row(summary: dict) -> dict:
    row = {k: summary[k] for k in CONFIG_SUMMARY_COLUMNS}
    for key, places in (("accuracy", 4), ("mean_f1", 4), ("em_rate", 4),
                        ("propagation_rate", 4), ("right_to_wrong_rate", 4),
                        ("total_cost", 6), ("cost_per_question", 6),
                        ("total_time_sec", 2), ("delta_accuracy_pts", 2),
                        ("delta_cost_pct", 2)):
        row[key] = _fmt(row[key], places)
    return row


def _summary_row(option: dict) -> dict:
    row = {k: option[k] for k in SUMMARY_COLUMNS}
    for key, places in (("accuracy", 4), ("delta_accuracy_pts", 2),
                        ("cost_per_question", 6), ("delta_cost_pct", 2),
                        ("propagation_rate", 4), ("right_to_wrong_rate", 4)):
        row[key] = _fmt(row[key], places)
    return row


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(inputs: list[Path], out_dir: Path, *, config_dir: Path = CONFIG_DIR,
              strong: str = "flash", tolerance: float = 5.0) -> dict:
    """Run the whole pipeline and write the three output files.

    Returns the in-memory results so callers (and tests) can inspect them.
    """
    rows = read_results(inputs)
    qrows = question_rows(rows)
    config_summaries = summarise_configs(qrows, config_dir, strong)
    step_summaries = summarise_steps(config_summaries, strong, tolerance)

    write_csv(out_dir / "all_results.csv", ALL_COLUMNS, rows)
    write_csv(out_dir / "config_summary.csv", CONFIG_SUMMARY_COLUMNS,
              [_config_summary_row(s) for s in config_summaries])
    write_csv(out_dir / "summary.csv", SUMMARY_COLUMNS,
              [_summary_row(o) for o in step_summaries])
    return {
        "rows": rows,
        "questions": qrows,
        "configs": config_summaries,
        "steps": step_summaries,
    }


def print_report(result: dict) -> None:
    """A short console view of what was written."""
    print(f"rows: {len(result['rows'])}, questions: {len(result['questions'])}, "
          f"configs: {len(result['configs'])}")
    for s in result["configs"]:
        acc = _fmt(s["accuracy"], 3) or "n/a"
        print(f"  {s['pipeline']:3} {s['config']:24} n={s['n_questions']:3} scored={s['n_scored']:3} "
              f"acc={acc:5} cost/q={_fmt(s['cost_per_question'], 5) or 'n/a':8} "
              f"prop={_fmt(s['propagation_rate'], 3) or 'n/a':5} "
              f"r2w={_fmt(s['right_to_wrong_rate'], 3) or 'n/a':5}"
              + ("  (baseline)" if s["is_baseline"] else ""))
    recommended = [o for o in result["steps"] if o["recommended"]]
    if recommended:
        print("recommendations:")
        for o in recommended:
            print(f"  {o['pipeline']:3} {o['step']:16} -> {o['model']:10} {o['reason']}")
    for o in result["steps"]:
        if o["reason"].startswith("no all-strong"):
            print(f"  {o['pipeline']}: no all-strong baseline found; summary has no recommendations")
            break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge results CSVs and recommend a model per pipeline step."
    )
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUT_GLOBS,
                        help="results CSV paths or globs (default: the experiments/ layouts "
                             "used by run_sql_matrix.py and run_qa_matrix.py)")
    parser.add_argument("--out-dir", default=str(EXPERIMENTS),
                        help="folder for all_results.csv, config_summary.csv, summary.csv")
    parser.add_argument("--config-dir", default=str(CONFIG_DIR),
                        help="folder with <config>.json routing files")
    parser.add_argument("--strong", default="flash",
                        help="model name of the strong model (default: flash)")
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="max accuracy drop in percentage points a cheaper option "
                             "may have and still be recommended (default: 5)")
    args = parser.parse_args(argv)

    inputs = expand_inputs(args.inputs)
    if not inputs:
        print("no results files found for: " + ", ".join(args.inputs), file=sys.stderr)
        return 1
    print("reading: " + ", ".join(str(p) for p in inputs))
    result = aggregate(inputs, Path(args.out_dir), config_dir=Path(args.config_dir),
                       strong=args.strong, tolerance=args.tolerance)
    print_report(result)
    print(f"wrote {Path(args.out_dir) / 'all_results.csv'}, "
          f"{Path(args.out_dir) / 'config_summary.csv'}, "
          f"{Path(args.out_dir) / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
