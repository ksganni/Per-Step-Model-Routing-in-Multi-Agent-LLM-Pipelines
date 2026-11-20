"""Statistical summary of the routing experiments for the final report.

Reads the same results CSVs as aggregate_results.py (SQL step rows and QA
question rows) and writes:

    reports/metrics.json          every number below, machine readable
    reports/tables/*.csv          one CSV per summary table
    reports/metrics_tables.md     the same tables as Markdown, ready for the report

What is computed, per pipeline (sql, qa):

1. Per-config quality with 95% confidence intervals.
   accuracy (is_correct) and EM use Wilson score intervals; mean F1 and mean
   cost per question use percentile bootstrap intervals.

2. Paired comparison of every config against the all-strong baseline, on the
   question_ids both answered. Accuracy: McNemar's exact test, with a
   Bonett-Price adjusted Wald interval for the paired difference (it stays
   honest when few or no answers change, where a bootstrap interval collapses
   to a point). F1 and cost: paired bootstrap interval and p-value; bootstrap
   p-values are two-sided, from the null-centred bootstrap distribution of the
   mean paired difference. p-values are Holm-corrected within each pipeline
   and metric.

3. Error propagation per step and model tier.
   right_to_wrong_rate: of the questions the baseline got right, the share the
   single-step swap got wrong (the same proxy as the Sankey figure).
   propagation_rate: share of judged questions with error_propagated True, which
   needs step rows run with --baseline (SQL only); blank otherwise.

4. Cost-quality Pareto frontier. A config is on the frontier when no other
   config is at least as cheap and at least as good with one of them strictly
   better. Computed on each config's own questions and, because configs finished
   different numbers of questions, again on the question_ids every config
   answered.

5. Step findings: per step and tier, the paired change in the primary metric
   with its CI and a verdict.
       too few        fewer than 20 paired questions; no verdict
       non-inferior   CI lower bound above -margin (default 0.05)
       worse          CI upper bound below 0 and Holm p below alpha
       inconclusive   anything else
   "non-inferior" is the claim the report needs for "this step tolerates a
   cheaper model"; a non-significant difference alone does not support it.

The primary metric is mean F1 for QA (the headline metric of the figures) and
accuracy for SQL (no F1 there). Bootstrap uses a fixed seed so reruns match.

From the repository root:

    python compute_metrics.py
    python compute_metrics.py --inputs experiments/qa_results.csv "experiments/sql/*.csv"
    python compute_metrics.py --resamples 20000 --margin 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import aggregate_results as agg

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

STRONG = "flash"
TIER_NAMES = {"flash": "strong", "flash-lite": "cheap", "qwen": "local"}
MODEL_LABELS = {"flash": "Flash", "flash-lite": "Flash-Lite", "qwen": "Qwen"}
Z95 = 1.959963984540054
MIN_COMMON = 20
MIN_PAIRED = 20


# --------------------------------------------------------------------------
# Small statistics helpers (stdlib only, so CI needs no numpy)
# --------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted list, q in [0, 1]."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_means(values: list[float], resamples: int, rng: random.Random) -> list[float]:
    """Sorted means of `resamples` bootstrap resamples of `values`."""
    n = len(values)
    out = [sum(rng.choices(values, k=n)) / n for _ in range(resamples)]
    out.sort()
    return out


def bootstrap_mean_ci(values: list[float], resamples: int, rng: random.Random,
                      level: float = 0.95) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    if len(set(values)) == 1:
        return (values[0], values[0])
    boots = bootstrap_means(values, resamples, rng)
    tail = (1 - level) / 2
    return (percentile(boots, tail), percentile(boots, 1 - tail))


def paired_bootstrap(diffs: list[float], resamples: int, rng: random.Random,
                     level: float = 0.95) -> dict:
    """Mean paired difference with percentile CI and two-sided bootstrap p.

    The p-value is the share of resampled means at least as far from the
    observed mean as the observed mean is from zero (the bootstrap
    distribution shifted to the null), with the +1 correction.
    """
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "p": None}
    observed = sum(diffs) / n
    if len(set(diffs)) == 1:
        return {"n": n, "mean": observed, "ci_low": observed, "ci_high": observed,
                "p": 1.0 if observed == 0 else 0.0}
    boots = bootstrap_means(diffs, resamples, rng)
    tail = (1 - level) / 2
    extreme = sum(1 for b in boots if abs(b - observed) >= abs(observed))
    return {
        "n": n,
        "mean": observed,
        "ci_low": percentile(boots, tail),
        "ci_high": percentile(boots, 1 - tail),
        "p": (extreme + 1) / (resamples + 1),
    }


def bonett_price_ci(b: int, c: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Adjusted Wald CI for the paired difference in proportions (c - b) / n.

    b = pairs right under the first condition only, c = right under the second
    only. Bonett and Price (2012) add one to each discordant cell, which keeps
    the interval wide when b and c are small instead of collapsing to zero.
    """
    if n == 0:
        return (None, None)
    pc = (c + 1) / (n + 2)
    pb = (b + 1) / (n + 2)
    diff = pc - pb
    half = z * math.sqrt(max(0.0, pc + pb - diff * diff) / (n + 2))
    return (max(-1.0, diff - half), min(1.0, diff + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values; None entries stay None."""
    indexed = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    indexed.sort()
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def pareto_front(points: list[dict], quality_key: str, cost_key: str) -> dict[str, list[str]]:
    """Map config -> configs that dominate it (empty list = on the frontier)."""
    dominated_by: dict[str, list[str]] = {}
    usable = [p for p in points if p.get(quality_key) is not None and p.get(cost_key) is not None]
    for p in usable:
        dominated_by[p["config"]] = sorted(
            q["config"] for q in usable
            if q is not p
            and q[cost_key] <= p[cost_key] and q[quality_key] >= p[quality_key]
            and (q[cost_key] < p[cost_key] or q[quality_key] > p[quality_key])
        )
    return dominated_by


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_questions(paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Question records from all inputs, de-duplicated across files.

    If the same (pipeline, config, question_id) appears in more than one input
    file (say experiments/results-all-flash.csv and experiments/sql/all-flash.csv),
    only the first file's rows are kept, so cost is not double-counted.
    Returns the records and a list of data-quality warnings.
    """
    warnings: list[str] = []
    owner: dict[tuple, Path] = {}
    kept_rows: list[dict] = []
    for path in paths:
        rows = agg.read_results([path])
        skipped: set[tuple] = set()
        seen_here: dict[tuple, int] = {}
        for row in rows:
            key = (row["pipeline"], row["config"], row["question_id"])
            first = owner.setdefault(key, path)
            if first != path:
                skipped.add(key)
                continue
            if row["step"] == "":
                seen_here[key] = seen_here.get(key, 0) + 1
            kept_rows.append(row)
        if skipped:
            warnings.append(f"{path}: {len(skipped)} question(s) already read from an earlier file; skipped")
        dupes = [k for k, count in seen_here.items() if count > 1]
        if dupes:
            warnings.append(f"{path}: {len(dupes)} question(s) have more than one summary row; "
                            "their cost is summed")
    return agg.question_rows(kept_rows), warnings


def _correct(record: dict) -> bool | None:
    if agg._is_true(record["is_correct"]):
        return True
    if agg._is_false(record["is_correct"]):
        return False
    return None


def _f1(record: dict) -> float | None:
    return float(record["f1"]) if record["f1"] not in ("", None) else None


def _em(record: dict) -> bool | None:
    if record["em"] == "":
        return None
    return agg._is_true(record["em"])


# --------------------------------------------------------------------------
# Per-pipeline analysis
# --------------------------------------------------------------------------

def config_entries(qrows: list[dict], config_dir: Path) -> "OrderedDict[tuple, dict]":
    """Group question records by (pipeline, config) with weak step and tier."""
    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for rec in qrows:
        key = (rec["pipeline"], rec["config"])
        entry = groups.setdefault(key, {"pipeline": rec["pipeline"], "config": rec["config"],
                                        "by_qid": OrderedDict(), "steps_seen": {}})
        entry["by_qid"][rec["question_id"]] = rec
        for step, model in rec["steps"].items():
            entry["steps_seen"].setdefault(step, model)
    for entry in groups.values():
        plan = agg.plan_for(entry["config"], entry["pipeline"], config_dir)
        weak_step, weak_model = agg.weak_step_for(plan, entry["steps_seen"], STRONG)
        entry["weak_step"] = weak_step
        entry["weak_model"] = weak_model
        entry["is_baseline"] = weak_step == "" and weak_model == ""
        entry["tier"] = "strong" if entry["is_baseline"] else TIER_NAMES.get(weak_model, weak_model)
    return groups


def summarise_config(entry: dict, resamples: int, rng: random.Random, ids: set | None = None) -> dict:
    recs = [r for q, r in entry["by_qid"].items() if ids is None or q in ids]
    correct = [c for c in (_correct(r) for r in recs) if c is not None]
    f1s = [v for v in (_f1(r) for r in recs) if v is not None]
    ems = [v for v in (_em(r) for r in recs) if v is not None]
    costs = [r["cost"] for r in recs]
    acc_lo, acc_hi = wilson_ci(sum(correct), len(correct))
    em_lo, em_hi = wilson_ci(sum(ems), len(ems))
    f1_lo, f1_hi = bootstrap_mean_ci(f1s, resamples, rng)
    cost_lo, cost_hi = bootstrap_mean_ci(costs, resamples, rng)
    return {
        "config": entry["config"],
        "weak_step": entry["weak_step"],
        "weak_model": entry["weak_model"],
        "tier": entry["tier"],
        "is_baseline": entry["is_baseline"],
        "n_questions": len(recs),
        "n_scored": len(correct),
        "accuracy": mean([1.0 if c else 0.0 for c in correct]),
        "accuracy_ci": [acc_lo, acc_hi],
        "n_f1": len(f1s),
        "mean_f1": mean(f1s),
        "mean_f1_ci": [f1_lo, f1_hi],
        "n_em": len(ems),
        "em_rate": mean([1.0 if e else 0.0 for e in ems]),
        "em_rate_ci": [em_lo, em_hi],
        "cost_per_question": mean(costs),
        "cost_per_question_ci": [cost_lo, cost_hi],
        "total_cost": sum(costs),
    }


def compare_to_baseline(entry: dict, base: dict, resamples: int, rng: random.Random) -> dict:
    """Paired comparison of one config against the baseline on shared ids."""
    shared = [q for q in entry["by_qid"] if q in base["by_qid"]]
    acc_diffs, f1_diffs, cost_diffs = [], [], []
    b = c = both_right = both_wrong = 0
    base_cost = conf_cost = 0.0
    for q in shared:
        x, y = base["by_qid"][q], entry["by_qid"][q]
        cx, cy = _correct(x), _correct(y)
        if cx is not None and cy is not None:
            acc_diffs.append((1.0 if cy else 0.0) - (1.0 if cx else 0.0))
            if cx and not cy:
                b += 1
            elif cy and not cx:
                c += 1
            elif cx and cy:
                both_right += 1
            else:
                both_wrong += 1
        fx, fy = _f1(x), _f1(y)
        if fx is not None and fy is not None:
            f1_diffs.append(fy - fx)
        cost_diffs.append(y["cost"] - x["cost"])
        base_cost += x["cost"]
        conf_cost += y["cost"]
    acc = paired_bootstrap(acc_diffs, resamples, rng)
    f1 = paired_bootstrap(f1_diffs, resamples, rng)
    cost = paired_bootstrap(cost_diffs, resamples, rng)
    n_base_right = both_right + b
    n_base_wrong = both_wrong + c
    r2w_lo, r2w_hi = wilson_ci(b, n_base_right)
    w2r_lo, w2r_hi = wilson_ci(c, n_base_wrong)
    return {
        "config": entry["config"],
        "weak_step": entry["weak_step"],
        "weak_model": entry["weak_model"],
        "tier": entry["tier"],
        "n_paired": len(shared),
        "n_paired_scored": len(acc_diffs),
        "delta_accuracy": acc["mean"],
        "delta_accuracy_ci": list(bonett_price_ci(b, c, len(acc_diffs))),
        "delta_accuracy_ci_bootstrap": [acc["ci_low"], acc["ci_high"]],
        "p_accuracy_bootstrap": acc["p"],
        "p_accuracy_mcnemar": mcnemar_exact(b, c) if acc_diffs else None,
        "n_paired_f1": f1["n"],
        "delta_f1": f1["mean"],
        "delta_f1_ci": [f1["ci_low"], f1["ci_high"]],
        "p_f1_bootstrap": f1["p"],
        "delta_cost": cost["mean"],
        "delta_cost_ci": [cost["ci_low"], cost["ci_high"]],
        "p_cost_bootstrap": cost["p"],
        "cost_ratio": (conf_cost / base_cost) if base_cost > 0 else None,
        "baseline_right_config_wrong": b,
        "baseline_wrong_config_right": c,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "n_baseline_right": n_base_right,
        "right_to_wrong_rate": (b / n_base_right) if n_base_right else None,
        "right_to_wrong_ci": [r2w_lo, r2w_hi],
        "n_baseline_wrong": n_base_wrong,
        "wrong_to_right_rate": (c / n_base_wrong) if n_base_wrong else None,
        "wrong_to_right_ci": [w2r_lo, w2r_hi],
    }


def propagation_rate(entry: dict) -> dict:
    """error_propagated True share among judged questions (SQL step rows)."""
    judged = [r for r in entry["by_qid"].values() if agg._scored(r["error_propagated"])]
    k = sum(1 for r in judged if agg._is_true(r["error_propagated"]))
    lo, hi = wilson_ci(k, len(judged))
    return {"n_judged": len(judged), "n_propagated": k,
            "propagation_rate": (k / len(judged)) if judged else None,
            "propagation_ci": [lo, hi]}


def analyse_pipeline(pipeline: str, entries: list[dict], resamples: int, seed: int,
                     margin: float, alpha: float) -> dict:
    rng = random.Random(f"{seed}-{pipeline}")
    baselines = [e for e in entries if e["is_baseline"]]
    base = max(baselines, key=lambda e: (len(e["by_qid"]), e["config"])) if baselines else None
    has_f1 = any(_f1(r) is not None for e in entries for r in e["by_qid"].values())
    primary = "mean_f1" if has_f1 else "accuracy"

    ordered = sorted(entries, key=lambda e: (not e["is_baseline"], e["weak_step"], e["tier"], e["config"]))
    configs = [summarise_config(e, resamples, rng) for e in ordered]

    comparisons: list[dict] = []
    if base is not None:
        for e in ordered:
            if e is base:
                continue
            comp = compare_to_baseline(e, base, resamples, rng)
            comp.update(propagation_rate(e))
            comparisons.append(comp)
        for key in ("p_accuracy_mcnemar", "p_accuracy_bootstrap", "p_f1_bootstrap", "p_cost_bootstrap"):
            for comp, adj in zip(comparisons, holm([c[key] for c in comparisons])):
                comp[key + "_holm"] = adj

    # Pareto frontier on each config's own questions and on the shared ids.
    quality_key = primary
    own = pareto_front(configs, quality_key, "cost_per_question")
    for cfg in configs:
        cfg["pareto_dominated_by"] = own.get(cfg["config"])
        cfg["on_pareto_frontier"] = (own.get(cfg["config"]) == []) if cfg["config"] in own else None
    common_ids: set | None = None
    for e in entries:
        scored = {q for q, r in e["by_qid"].items() if _correct(r) is not None}
        common_ids = scored if common_ids is None else common_ids & scored
    common_ids = common_ids or set()
    common = [summarise_config(e, resamples, rng, ids=common_ids) for e in ordered] \
        if len(common_ids) >= MIN_COMMON else []
    common_front = pareto_front(common, quality_key, "cost_per_question") if common else {}
    for cfg in common:
        cfg["pareto_dominated_by"] = common_front.get(cfg["config"])
        cfg["on_pareto_frontier"] = common_front.get(cfg["config"]) == []

    findings = step_findings(comparisons, primary, margin, alpha)
    return {
        "primary_metric": primary,
        "baseline": base["config"] if base else None,
        "configs": configs,
        "comparisons": comparisons,
        "pareto": {
            "quality_metric": quality_key,
            "own_questions": [{"config": c["config"], "cost_per_question": c["cost_per_question"],
                               quality_key: c[quality_key], "n": c["n_questions"],
                               "on_frontier": c["on_pareto_frontier"],
                               "dominated_by": c["pareto_dominated_by"]} for c in configs],
            "common_questions": {
                "n_common": len(common_ids),
                "used": bool(common),
                "points": [{"config": c["config"], "cost_per_question": c["cost_per_question"],
                            quality_key: c[quality_key], "n": c["n_questions"],
                            "on_frontier": c["on_pareto_frontier"],
                            "dominated_by": c["pareto_dominated_by"]} for c in common],
            },
        },
        "step_findings": findings,
    }


def step_findings(comparisons: list[dict], primary: str, margin: float, alpha: float) -> list[dict]:
    delta_key, ci_key, p_key = (("delta_f1", "delta_f1_ci", "p_f1_bootstrap_holm") if primary == "mean_f1"
                                else ("delta_accuracy", "delta_accuracy_ci", "p_accuracy_mcnemar_holm"))
    out = []
    for comp in comparisons:
        if comp["weak_step"] in ("", "all", "multiple", "unknown"):
            continue
        delta, (lo, hi), p = comp[delta_key], comp[ci_key], comp.get(p_key)
        if delta is None or lo is None:
            verdict = "no data"
        elif comp["n_paired"] < MIN_PAIRED:
            verdict = "too few"
        elif lo > -margin:
            verdict = "non-inferior"
        elif hi < 0 and p is not None and p < alpha:
            verdict = "worse"
        else:
            verdict = "inconclusive"
        out.append({
            "step": comp["weak_step"],
            "model": comp["weak_model"],
            "tier": comp["tier"],
            "config": comp["config"],
            "n_paired": comp["n_paired"],
            "metric": primary,
            "delta": delta,
            "delta_ci": [lo, hi],
            "p_holm": p,
            "cost_ratio": comp["cost_ratio"],
            "right_to_wrong_rate": comp["right_to_wrong_rate"],
            "verdict": verdict,
        })
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _num(value, places: int = 3) -> str:
    return "" if value is None else f"{value:.{places}f}"


def _ci(pair, places: int = 3) -> str:
    lo, hi = pair
    return "" if lo is None else f"[{lo:.{places}f}, {hi:.{places}f}]"


def _p(value) -> str:
    if value is None:
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _label(step: str, model: str) -> str:
    if not step:
        return "all-strong"
    return f"{step.split('.')[-1]} → {MODEL_LABELS.get(model, model)}"


def build_tables(result: dict) -> "OrderedDict[str, tuple[list[str], list[list[str]]]]":
    """Table name -> (header, rows) for both CSV and Markdown output."""
    tables: "OrderedDict[str, tuple[list[str], list[list[str]]]]" = OrderedDict()
    cfg_rows, cmp_rows, prop_rows, par_rows, find_rows = [], [], [], [], []
    for pipeline, res in result["pipelines"].items():
        for c in res["configs"]:
            cfg_rows.append([pipeline, c["config"], _label(c["weak_step"], c["weak_model"]), c["tier"],
                             str(c["n_questions"]), _num(c["accuracy"]), _ci(c["accuracy_ci"]),
                             _num(c["mean_f1"]), _ci(c["mean_f1_ci"]),
                             _num(c["em_rate"]), _num(c["cost_per_question"], 5),
                             _ci(c["cost_per_question_ci"], 5)])
        for c in res["comparisons"]:
            cmp_rows.append([pipeline, c["config"], _label(c["weak_step"], c["weak_model"]),
                             str(c["n_paired"]),
                             _num(c["delta_accuracy"]), _ci(c["delta_accuracy_ci"]),
                             f"{c['baseline_right_config_wrong']}/{c['baseline_wrong_config_right']}",
                             _p(c["p_accuracy_mcnemar"]), _p(c.get("p_accuracy_mcnemar_holm")),
                             _num(c["delta_f1"]), _ci(c["delta_f1_ci"]),
                             _p(c["p_f1_bootstrap"]), _p(c.get("p_f1_bootstrap_holm")),
                             _num(c["cost_ratio"], 2)])
            prop_rows.append([pipeline, c["weak_step"].split(".")[-1] or "-",
                              MODEL_LABELS.get(c["weak_model"], c["weak_model"]), c["tier"], c["config"],
                              str(c["n_baseline_right"]), str(c["baseline_right_config_wrong"]),
                              _num(c["right_to_wrong_rate"]), _ci(c["right_to_wrong_ci"]),
                              str(c["n_baseline_wrong"]), _num(c["wrong_to_right_rate"]),
                              _num(c["propagation_rate"]), _ci(c["propagation_ci"]),
                              str(c["n_judged"])])
        q = res["pareto"]["quality_metric"]
        common = {p["config"]: p for p in res["pareto"]["common_questions"]["points"]}
        for p in res["pareto"]["own_questions"]:
            cp = common.get(p["config"], {})
            par_rows.append([pipeline, p["config"], q, str(p["n"]), _num(p[q]),
                             _num(p["cost_per_question"], 5),
                             "yes" if p["on_frontier"] else "no",
                             ", ".join(p["dominated_by"] or []),
                             _num(cp.get(q)), _num(cp.get("cost_per_question"), 5),
                             ("yes" if cp.get("on_frontier") else "no") if cp else ""])
        for f in res["step_findings"]:
            find_rows.append([pipeline, f["step"].split(".")[-1], MODEL_LABELS.get(f["model"], f["model"]),
                              str(f["n_paired"]), f["metric"], _num(f["delta"]), _ci(f["delta_ci"]),
                              _p(f["p_holm"]), _num(f["cost_ratio"], 2),
                              _num(f["right_to_wrong_rate"]), f["verdict"]])
    tables["config_quality"] = (
        ["pipeline", "config", "swap", "tier", "n", "accuracy", "accuracy 95% CI", "mean F1",
         "F1 95% CI", "EM", "cost/q ($)", "cost/q 95% CI"], cfg_rows)
    tables["paired_tests"] = (
        ["pipeline", "config", "swap", "n paired", "Δ accuracy", "Δ acc 95% CI", "b/c",
         "McNemar p", "McNemar p (Holm)", "Δ F1", "Δ F1 95% CI", "bootstrap p", "bootstrap p (Holm)",
         "cost ratio"], cmp_rows)
    tables["error_propagation"] = (
        ["pipeline", "step", "model", "tier", "config", "baseline right", "became wrong",
         "right→wrong rate", "95% CI", "baseline wrong", "wrong→right rate",
         "error_propagated rate", "95% CI", "n judged"], prop_rows)
    tables["pareto"] = (
        ["pipeline", "config", "quality metric", "n", "quality", "cost/q ($)", "on frontier",
         "dominated by", "quality (common ids)", "cost/q (common ids)", "on frontier (common ids)"],
        par_rows)
    tables["step_findings"] = (
        ["pipeline", "step", "model", "n paired", "metric", "Δ vs all-strong", "95% CI", "p (Holm)",
         "cost ratio", "right→wrong", "verdict"], find_rows)
    return tables


def write_outputs(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    written = []
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    written.append(json_path)
    tables = build_tables(result)
    md = ["# Summary tables", "",
          f"Generated {result['meta']['generated_at']} by `compute_metrics.py` from "
          f"{', '.join(result['meta']['inputs'])}. "
          f"95% CIs: Wilson for proportions, Bonett-Price for paired accuracy differences, "
          f"percentile bootstrap ({result['meta']['bootstrap_resamples']} resamples, "
          f"seed {result['meta']['seed']}) for means and paired F1 and cost differences. "
          f"Non-inferiority margin {result['meta']['margin']}; no verdict below {MIN_PAIRED} "
          f"paired questions.", ""]
    titles = {
        "config_quality": "Per-config quality and cost",
        "paired_tests": "Paired tests against all-strong",
        "error_propagation": "Error propagation by step and model tier",
        "pareto": "Cost-quality Pareto frontier",
        "step_findings": "Step findings",
    }
    for name, (header, rows) in tables.items():
        path = tables_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        written.append(path)
        md += [f"## {titles[name]}", "", "| " + " | ".join(header) + " |",
               "|" + "---|" * len(header)]
        md += ["| " + " | ".join(cell or "" for cell in row) + " |" for row in rows]
        md.append("")
    if result["data_quality"]["warnings"]:
        md += ["## Data quality", ""] + [f"- {w}" for w in result["data_quality"]["warnings"]] + [""]
    md_path = out_dir / "metrics_tables.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    written.append(md_path)
    return written


def compute(inputs: list[Path], *, config_dir: Path = agg.CONFIG_DIR, resamples: int = 10000,
            seed: int = 42, margin: float = 0.05, alpha: float = 0.05) -> dict:
    qrows, warnings = load_questions(inputs)
    groups = config_entries(qrows, config_dir)
    by_pipeline: "OrderedDict[str, list[dict]]" = OrderedDict()
    for entry in groups.values():
        by_pipeline.setdefault(entry["pipeline"], []).append(entry)
    pipelines = OrderedDict()
    for pipeline in sorted(by_pipeline):
        entries = by_pipeline[pipeline]
        if not any(e["is_baseline"] for e in entries):
            warnings.append(f"{pipeline}: no all-strong baseline found; no paired tests or findings")
        res = analyse_pipeline(pipeline, entries, resamples, seed, margin, alpha)
        small = [c["config"] for c in res["configs"] if c["n_questions"] < MIN_COMMON]
        if small:
            warnings.append(f"{pipeline}: fewer than {MIN_COMMON} questions for {', '.join(small)}; "
                            "treat those intervals as indicative only")
        if res["pareto"]["common_questions"]["n_common"] < MIN_COMMON:
            warnings.append(f"{pipeline}: only {res['pareto']['common_questions']['n_common']} question_ids "
                            f"shared by every config (< {MIN_COMMON}); common-ids frontier not computed")
        pipelines[pipeline] = res
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "script": "compute_metrics.py",
            "inputs": [str(p) for p in inputs],
            "n_question_records": len(qrows),
            "bootstrap_resamples": resamples,
            "seed": seed,
            "ci_level": 0.95,
            "margin": margin,
            "alpha": alpha,
            "definitions": {
                "accuracy": "share of scored questions with is_correct True (QA: EM or token F1 >= 0.8)",
                "mean_f1": "mean token F1 over questions with an f1 value (QA only)",
                "right_to_wrong_rate": "of questions the baseline got right, share the config got wrong",
                "propagation_rate": "share of judged questions with error_propagated True (needs step rows "
                                    "run with --baseline)",
                "cost_ratio": "config cost / baseline cost summed over paired questions",
                "p_holm": "Holm-Bonferroni adjusted within pipeline and metric",
                "delta_accuracy_ci": "Bonett-Price adjusted Wald interval for the paired difference",
                "delta_f1_ci": "percentile bootstrap interval of the mean paired F1 difference",
                "verdict": f"too few if fewer than {MIN_PAIRED} paired questions; non-inferior if CI lower "
                           f"bound > -{margin}; worse if CI upper bound < 0 and Holm p < {alpha}; "
                           "otherwise inconclusive",
            },
        },
        "pipelines": pipelines,
        "data_quality": {"warnings": warnings},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Confidence intervals, paired tests, propagation "
                                                 "and Pareto frontier for the routing experiments.")
    parser.add_argument("--inputs", nargs="+", default=agg.DEFAULT_INPUT_GLOBS,
                        help="results CSV paths or globs (default: the experiments/ layouts)")
    parser.add_argument("--out-dir", default=str(REPORTS),
                        help="folder for metrics.json, metrics_tables.md and tables/")
    parser.add_argument("--config-dir", default=str(agg.CONFIG_DIR))
    parser.add_argument("--resamples", type=int, default=10000, help="bootstrap resamples (default 10000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.05,
                        help="non-inferiority margin on the primary metric (default 0.05)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    inputs = agg.expand_inputs(args.inputs)
    if not inputs:
        print("no results files found for: " + ", ".join(args.inputs), file=sys.stderr)
        return 1
    print("reading: " + ", ".join(str(p) for p in inputs))
    result = compute(inputs, config_dir=Path(args.config_dir), resamples=args.resamples,
                     seed=args.seed, margin=args.margin, alpha=args.alpha)
    for path in write_outputs(result, Path(args.out_dir)):
        print(f"wrote {path}")
    for pipeline, res in result["pipelines"].items():
        print(f"{pipeline}: baseline={res['baseline']} primary={res['primary_metric']}")
        for f in res["step_findings"]:
            lo, hi = f["delta_ci"]
            print(f"  {f['step']:16} {f['model']:10} n={f['n_paired']:3} delta={_num(f['delta'])} "
                  f"CI={_ci(f['delta_ci'])} p_holm={_p(f['p_holm'])} cost_ratio={_num(f['cost_ratio'], 2)} "
                  f"-> {f['verdict']}")
    for warning in result["data_quality"]["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
