"""Run the Sprint 4 QA routing matrix into experiments/qa_results.csv.

Nine configs x 150 HotpotQA questions ≈ 1,350 summary rows with F1/EM:

  qa-all-strong
  qa-cheap-{decomposer,hop-reader,composer,formatter}
  qa-qwen-{decomposer,hop-reader,composer,formatter}

Resume is per (config, question_id). Re-run the same command after a
failure; completed pairs are skipped. Keep Qwen warm in another terminal
when any qa-qwen-* config is in the list:

    python -m harness.qwen_keep_warm

From the repository root:

    python run_qa_matrix.py
    python run_qa_matrix.py --limit 5
    python run_qa_matrix.py --configs qa-all-strong qa-cheap-decomposer
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_runner import DEFAULT_QA_RESULTS, run_experiment

QA_MATRIX_CONFIGS = [
    "qa-all-strong",
    "qa-cheap-decomposer",
    "qa-cheap-hop-reader",
    "qa-cheap-composer",
    "qa-cheap-formatter",
    "qa-qwen-decomposer",
    "qa-qwen-hop-reader",
    "qa-qwen-composer",
    "qa-qwen-formatter",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all QA routing configs into experiments/qa_results.csv."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=QA_MATRIX_CONFIGS,
        help="config names under experiments/configs/ (default: full matrix of 9)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N HotpotQA sample questions per config")
    parser.add_argument("--results", default=str(DEFAULT_QA_RESULTS),
                        help="CSV path (default: experiments/qa_results.csv)")
    args = parser.parse_args()

    results_path = Path(args.results)
    total = 0
    for name in args.configs:
        print(f"=== config {name} ===")
        ran = run_experiment(
            "qa",
            name,
            limit=args.limit,
            results_path=results_path,
            question_rows=True,
        )
        total += ran
        print(f"=== {name}: ran {ran} ===\n")
    print(f"matrix done: ran {total} question(s) across {len(args.configs)} config(s) -> {results_path}")


if __name__ == "__main__":
    main()
