"""Run the Sprint 4 SQL routing matrix into experiments/sql/<config>.csv.

Seven configs x 150 BIRD questions, one results file per config:

  all-flash                       the all-strong baseline, run first
  cheap-{selector,decomposer,refiner}
  qwen-{selector,decomposer,refiner}

Every config after the baseline is run with --baseline pointing at the
baseline file, so its rows carry error_propagated. Rows are one per model
call (step rows), which error propagation needs.

Each config has its own file because experiment_runner's resume skips a
question_id that is already in the results file, whatever config wrote it.
Re-run the same command after a failure; completed questions are skipped.

From the repository root:

    python run_sql_matrix.py
    python run_sql_matrix.py --limit 5
    python run_sql_matrix.py --configs all-flash cheap-selector
    python run_sql_matrix.py --configs qwen-refiner      # baseline file must exist

Then merge and summarise with:

    python aggregate_results.py --inputs "experiments/sql/*.csv"

Keep Qwen warm on Kaggle while any qwen-* config runs (see harness/
qwen_keep_warm.py once K8 lands, or a tab with the notebook open).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_runner import ROOT, run_experiment
from run_pipeline import DB_DIR

BASELINE_CONFIG = "all-flash"
SQL_MATRIX_CONFIGS = [
    BASELINE_CONFIG,
    "cheap-selector",
    "cheap-decomposer",
    "cheap-refiner",
    "qwen-selector",
    "qwen-decomposer",
    "qwen-refiner",
]
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "sql"


def results_path(results_dir: Path, config: str) -> Path:
    return results_dir / f"{config}.csv"


def run_matrix(configs: list[str], *, results_dir: Path = DEFAULT_RESULTS_DIR,
               limit: int | None = None, db_dir: str | Path | None = DB_DIR) -> int:
    """Run the configs in order, baseline first, and return questions run."""
    ordered = ([BASELINE_CONFIG] if BASELINE_CONFIG in configs else []) + [
        c for c in configs if c != BASELINE_CONFIG]
    baseline_file = results_path(results_dir, BASELINE_CONFIG)
    results_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in ordered:
        print(f"=== config {name} ===")
        ran = run_experiment(
            "sql",
            name,
            limit=limit,
            results_path=results_path(results_dir, name),
            db_dir=db_dir,
            baseline_path=None if name == BASELINE_CONFIG else baseline_file,
        )
        total += ran
        print(f"=== {name}: ran {ran} ===\n")
    print(f"matrix done: ran {total} question(s) across {len(ordered)} config(s) -> {results_dir}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all SQL routing configs, one results CSV per config."
    )
    parser.add_argument("--configs", nargs="+", default=SQL_MATRIX_CONFIGS,
                        help="config names under experiments/configs/ (default: all 7)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N BIRD sample questions per config")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                        help="folder for <config>.csv files (default: experiments/sql)")
    parser.add_argument("--db-dir", default=str(DB_DIR),
                        help="folder with <db_id>/<db_id>.sqlite (needed for scoring)")
    args = parser.parse_args()
    run_matrix(args.configs, results_dir=Path(args.results_dir), limit=args.limit,
               db_dir=args.db_dir or None)


if __name__ == "__main__":
    main()
