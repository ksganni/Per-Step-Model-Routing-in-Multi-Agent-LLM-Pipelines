#!/usr/bin/env python3
"""Build Sprint 5 publication figures from experiment CSVs.

Re-run after any update to ``experiments/qa_results.csv`` (and optional SQL
baseline CSVs). Figures are written under ``reports/figures/``.

    python reports/plot_figures.py

QA matrix rows are one summary row per (config, question). Single-step swap
configs map to (step, model) cells for the heatmap / routing map. Error
propagation for QA is estimated by comparing each cheap/qwen config against
``qa-all-strong`` on shared question_ids (baseline correct → mixed wrong).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QA_RESULTS = ROOT / "experiments" / "qa_results.csv"
SQL_BASELINE = ROOT / "experiments" / "results-all-flash.csv"
SQL_CHEAP = ROOT / "experiments" / "results-cheap-selector.csv"
FIG_DIR = ROOT / "reports" / "figures"

STEPS = ["decomposer", "hop_reader", "composer", "formatter"]
STEP_LABELS = {
    "decomposer": "Decomposer",
    "hop_reader": "Hop reader",
    "composer": "Composer",
    "formatter": "Formatter",
}
MODELS = ["flash", "flash-lite", "qwen"]
MODEL_LABELS = {
    "flash": "Flash",
    "flash-lite": "Flash-Lite",
    "qwen": "Qwen",
}

# config name → (step | None for all-strong, model)
CONFIG_STEP_MODEL: dict[str, tuple[str | None, str]] = {
    "qa-all-strong": (None, "flash"),
    "qa-cheap-decomposer": ("decomposer", "flash-lite"),
    "qa-cheap-hop-reader": ("hop_reader", "flash-lite"),
    "qa-cheap-composer": ("composer", "flash-lite"),
    "qa-cheap-formatter": ("formatter", "flash-lite"),
    "qa-qwen-decomposer": ("decomposer", "qwen"),
    "qa-qwen-hop-reader": ("hop_reader", "qwen"),
    "qa-qwen-composer": ("composer", "qwen"),
    "qa-qwen-formatter": ("formatter", "qwen"),
}


def _f(value: str | float | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_qa(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing QA results: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def config_stats(rows: list[dict]) -> dict[str, dict]:
    """Per-config n, mean F1, EM rate, mean cost, mean is_correct."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cfg = row.get("config") or ""
        if cfg:
            buckets[cfg].append(row)
    out: dict[str, dict] = {}
    for cfg, items in buckets.items():
        f1s = [_f(r.get("f1")) for r in items]
        ems = [1.0 if _truthy(r.get("em")) else 0.0 for r in items]
        costs = [_f(r.get("cost")) for r in items]
        correct = [1.0 if _truthy(r.get("is_correct")) else 0.0 for r in items]
        out[cfg] = {
            "n": len(items),
            "f1": float(np.mean(f1s)) if f1s else float("nan"),
            "em": float(np.mean(ems)) if ems else float("nan"),
            "cost": float(np.mean(costs)) if costs else float("nan"),
            "accuracy": float(np.mean(correct)) if correct else float("nan"),
            "rows": items,
        }
    return out


def step_model_matrix(stats: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return accuracy, cost, and n matrices shaped (steps, models)."""
    acc = np.full((len(STEPS), len(MODELS)), np.nan)
    cost = np.full((len(STEPS), len(MODELS)), np.nan)
    nmat = np.zeros((len(STEPS), len(MODELS)), dtype=int)

    strong = stats.get("qa-all-strong")
    for i, step in enumerate(STEPS):
        for j, model in enumerate(MODELS):
            if model == "flash" and strong is not None:
                # All-strong is Flash on every step.
                acc[i, j] = strong["f1"]
                cost[i, j] = strong["cost"]
                nmat[i, j] = strong["n"]
                continue
            cfg = None
            for name, (s, m) in CONFIG_STEP_MODEL.items():
                if s == step and m == model:
                    cfg = name
                    break
            if cfg and cfg in stats:
                acc[i, j] = stats[cfg]["f1"]
                cost[i, j] = stats[cfg]["cost"]
                nmat[i, j] = stats[cfg]["n"]
    return acc, cost, nmat


def recommend_routing(
    acc: np.ndarray,
    cost: np.ndarray,
    strong_f1: float,
    drop_tol: float = 0.05,
) -> dict[str, str]:
    """Cheapest model per step whose F1 is within drop_tol of all-strong F1."""
    routing: dict[str, str] = {}
    for i, step in enumerate(STEPS):
        candidates: list[tuple[float, str]] = []
        for j, model in enumerate(MODELS):
            a, c = acc[i, j], cost[i, j]
            if np.isnan(a) or np.isnan(c):
                continue
            if a + 1e-9 >= strong_f1 - drop_tol:
                candidates.append((c, model))
        if not candidates:
            # Fall back to best accuracy, then cheapest among ties.
            best_j = int(np.nanargmax(acc[i]))
            routing[step] = MODELS[best_j]
        else:
            candidates.sort(key=lambda t: (t[0], 0 if t[1] != "flash" else 1))
            routing[step] = candidates[0][1]
    return routing


def style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_heatmap(acc: np.ndarray, cost: np.ndarray, nmat: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    # Color encodes cost; annotate with F1 and n.
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    finite = cost[np.isfinite(cost)]
    norm = Normalize(
        vmin=float(finite.min()) if len(finite) else 0.0,
        vmax=float(finite.max()) if len(finite) else 1.0,
    )
    # Mask missing cells
    display = np.ma.array(cost, mask=~np.isfinite(cost))
    im = ax.imshow(display, cmap=cmap, norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean question cost (USD)")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_yticks(range(len(STEPS)))
    ax.set_yticklabels([STEP_LABELS[s] for s in STEPS])
    ax.set_xlabel("Model on this step (others = Flash)")
    ax.set_ylabel("Pipeline step")
    ax.set_title("Accuracy × cost heatmap (QA)\ncell text = mean F1 (n); color = mean cost")

    for i in range(len(STEPS)):
        for j in range(len(MODELS)):
            if not np.isfinite(acc[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", color="#666", fontsize=10)
                continue
            # Pick readable text color from background luminance.
            rgba = cmap(norm(cost[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.55 else "black"
            ax.text(
                j,
                i,
                f"{acc[i, j]:.2f}\n(n={nmat[i, j]})",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_routing_map(
    routing: dict[str, str],
    acc: np.ndarray,
    cost: np.ndarray,
    strong_f1: float,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title(
        "Per-step routing map (QA)\n"
        f"Prefer cheapest model within {0.05:.0%} F1 of all-strong "
        f"(baseline F1={strong_f1:.2f})",
        pad=12,
    )

    colors = {
        "flash": "#1f4e79",
        "flash-lite": "#2a9d8f",
        "qwen": "#e9c46a",
    }
    x0 = 0.4
    box_w = 2.1
    gap = 0.35
    for i, step in enumerate(STEPS):
        model = routing[step]
        x = x0 + i * (box_w + gap)
        y = 1.0
        si = STEPS.index(step)
        mi = MODELS.index(model) if model in MODELS else 0
        f1 = acc[si, mi]
        c = cost[si, mi]
        patch = FancyBboxPatch(
            (x, y),
            box_w,
            1.2,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            facecolor=colors.get(model, "#ccc"),
            edgecolor="#222",
            linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(
            x + box_w / 2,
            y + 0.85,
            STEP_LABELS[step],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white" if model == "flash" else "#111",
        )
        ax.text(
            x + box_w / 2,
            y + 0.45,
            MODEL_LABELS.get(model, model),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white" if model == "flash" else "#111",
        )
        ax.text(
            x + box_w / 2,
            y + 0.15,
            f"F1={f1:.2f}  ${c:.3f}" if np.isfinite(f1) and np.isfinite(c) else "",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if model == "flash" else "#222",
        )
        if i < len(STEPS) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap * 0.15, y + 0.6),
                xytext=(x + box_w - 0.02, y + 0.6),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
            )

    # Legend
    for j, model in enumerate(MODELS):
        ax.add_patch(
            FancyBboxPatch(
                (0.4 + j * 2.4, 0.25),
                0.35,
                0.25,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=colors[model],
                edgecolor="#222",
            )
        )
        ax.text(0.85 + j * 2.4, 0.37, MODEL_LABELS[model], va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def estimate_recommended_cost(
    stats: dict[str, dict],
    routing: dict[str, str],
) -> float:
    """Approximate recommended cost without a joint multi-swap run.

    Uses the mean of the observed single-swap (or all-strong) config costs
    for each step's recommended model. This avoids double-counting savings
    from adding independent one-step swaps.
    """
    strong = stats.get("qa-all-strong")
    if strong is None:
        return float("nan")
    vals: list[float] = []
    for step, model in routing.items():
        if model == "flash":
            vals.append(strong["cost"])
            continue
        cfg = None
        for name, (s, m) in CONFIG_STEP_MODEL.items():
            if s == step and m == model:
                cfg = name
                break
        if cfg and cfg in stats and np.isfinite(stats[cfg]["cost"]):
            vals.append(stats[cfg]["cost"])
        else:
            vals.append(strong["cost"])
    return float(np.mean(vals)) if vals else float("nan")


def estimate_all_cheap_cost(stats: dict[str, dict]) -> float:
    """Reprice all-strong token totals as Flash-Lite (proxy for all-cheap)."""
    from harness.pricing import compute_cost

    strong_rows = stats.get("qa-all-strong", {}).get("rows") or []
    if not strong_rows:
        return float("nan")
    costs = []
    for r in strong_rows:
        tin = int(_f(r.get("tokens_in")))
        tout = int(_f(r.get("tokens_out")))
        costs.append(compute_cost("flash-lite", tin, tout))
    return float(np.mean(costs)) if costs else float("nan")


def plot_cost_bars(
    stats: dict[str, dict],
    routing: dict[str, str],
    out: Path,
) -> None:
    strong = stats.get("qa-all-strong")
    if strong is None:
        raise RuntimeError("qa-all-strong required for cost bar chart")

    all_strong_cost = strong["cost"]
    all_strong_f1 = strong["f1"]
    recommended_cost = estimate_recommended_cost(stats, routing)
    all_cheap_cost = estimate_all_cheap_cost(stats)

    # Accuracy for recommended ≈ mean F1 of recommended step configs (proxy)
    rec_f1s = []
    for step, model in routing.items():
        if model == "flash":
            rec_f1s.append(all_strong_f1)
            continue
        for name, (s, m) in CONFIG_STEP_MODEL.items():
            if s == step and m == model and name in stats:
                rec_f1s.append(stats[name]["f1"])
    recommended_f1 = float(np.mean(rec_f1s)) if rec_f1s else all_strong_f1

    # All-cheap F1 proxy: mean of flash-lite single-swap configs
    cheap_f1s = [
        stats[c]["f1"]
        for c, (s, m) in CONFIG_STEP_MODEL.items()
        if m == "flash-lite" and c in stats
    ]
    all_cheap_f1 = float(np.mean(cheap_f1s)) if cheap_f1s else float("nan")

    labels = ["All-strong\n(Flash)", "Recommended\nrouting", "All-cheap\n(Lite proxy)"]
    costs = [all_strong_cost, recommended_cost, all_cheap_cost]
    f1s = [all_strong_f1, recommended_f1, all_cheap_f1]
    bar_colors = ["#1f4e79", "#2a9d8f", "#e76f51"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, costs, color=bar_colors, edgecolor="#222", width=0.65)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean cost per question (USD)")
    ax.set_title("Cost savings — all-strong vs recommended vs all-cheap (QA)")

    for bar, c, f1 in zip(bars, costs, f1s):
        if not np.isfinite(c):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${c:.4f}\nF1≈{f1:.2f}" if np.isfinite(f1) else f"${c:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if np.isfinite(all_strong_cost) and np.isfinite(recommended_cost) and all_strong_cost > 0:
        save = 100.0 * (1.0 - recommended_cost / all_strong_cost)
        ax.text(
            0.98,
            0.98,
            f"Est. savings vs all-strong: {save:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="#ccc"),
        )

    route_txt = " → ".join(
        f"{STEP_LABELS[s]}:{MODEL_LABELS[routing[s]]}" for s in STEPS
    )
    fig.text(0.5, -0.02, f"Recommended: {route_txt}", ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def qa_propagation_flows(stats: dict[str, dict]) -> list[tuple[str, str, float]]:
    """Return (source, target, count) flows for Sankey-style diagram."""
    strong = stats.get("qa-all-strong")
    if strong is None:
        return []
    strong_by_id = {
        r["question_id"]: r
        for r in strong["rows"]
        if r.get("question_id")
    }
    flows: dict[tuple[str, str], float] = defaultdict(float)

    for cfg, (step, model) in CONFIG_STEP_MODEL.items():
        if step is None or cfg not in stats:
            continue
        if model not in {"flash-lite", "qwen"}:
            continue
        label = f"{STEP_LABELS[step]}\n({MODEL_LABELS[model]})"
        for row in stats[cfg]["rows"]:
            qid = row.get("question_id")
            base = strong_by_id.get(qid or "")
            if base is None:
                continue
            # Only count cases where baseline was correct.
            if not (_truthy(base.get("is_correct")) or _truthy(base.get("em"))):
                continue
            mixed_ok = _truthy(row.get("is_correct")) or _truthy(row.get("em"))
            if mixed_ok:
                flows[(label, "Final OK\n(no prop.)")] += 1
            else:
                flows[(label, "Final wrong\n(propagated)")] += 1

    return [(a, b, c) for (a, b), c in flows.items() if c > 0]


def plot_sankey(flows: list[tuple[str, str, float]], out: Path) -> None:
    """Draw a layered flow diagram (Sankey-style) with matplotlib."""
    if not flows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Not enough overlapping baseline/mixed rows yet\n"
            "for an error-propagation diagram.\n"
            "Re-run after more qa-all-strong + cheap/qwen rows.",
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return

    left_nodes = sorted({a for a, _, _ in flows})
    right_nodes = ["Final OK\n(no prop.)", "Final wrong\n(propagated)"]
    # Keep only right nodes that appear
    right_nodes = [r for r in right_nodes if any(b == r for _, b, _ in flows)]

    left_totals = {n: sum(c for a, _, c in flows if a == n) for n in left_nodes}
    right_totals = {n: sum(c for _, b, c in flows if b == n) for n in right_nodes}
    total = sum(left_totals.values()) or 1.0

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "Error propagation (QA proxy)\n"
        "Among questions all-strong got right: does a weak step still yield a wrong final?",
        pad=14,
    )

    def stack_positions(nodes: list[str], totals: dict[str, float], x: float, color_map):
        positions = {}
        gap = 0.25
        usable = 9.0 - gap * max(0, len(nodes) - 1)
        y = 9.5
        for n in nodes:
            h = max(0.35, usable * (totals[n] / total))
            y -= h
            positions[n] = (x, y, h)
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.9, y),
                    1.8,
                    h,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=color_map(n),
                    edgecolor="#222",
                    linewidth=1.0,
                    alpha=0.95,
                )
            )
            ax.text(
                x,
                y + h / 2,
                f"{n}\n(n={int(totals[n])})",
                ha="center",
                va="center",
                fontsize=8,
                color="#111",
            )
            y -= gap
        return positions

    def left_color(name: str) -> str:
        if "Qwen" in name:
            return "#e9c46a"
        if "Lite" in name or "Flash-Lite" in name:
            return "#2a9d8f"
        return "#8ecae6"

    def right_color(name: str) -> str:
        return "#90be6d" if "OK" in name else "#e76f51"

    left_pos = stack_positions(left_nodes, left_totals, 1.6, left_color)
    right_pos = stack_positions(right_nodes, right_totals, 8.4, right_color)

    # Flows as ribbons (simple trapezoids via fill_between)
    for src, dst, count in sorted(flows, key=lambda t: -t[2]):
        if src not in left_pos or dst not in right_pos:
            continue
        x0, y0, h0 = left_pos[src]
        x1, y1, h1 = right_pos[dst]
        # Thickness proportional to count within source
        frac = count / left_totals[src]
        band = h0 * frac
        # Place band from top of remaining - approximate centered
        y0_mid = y0 + h0 / 2
        y1_mid = y1 + h1 / 2
        half = max(0.08, band / 2)
        xs = np.linspace(x0 + 0.9, x1 - 0.9, 40)
        # Smooth vertical blend
        t = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
        top = y0_mid + half + (y1_mid + half - (y0_mid + half)) * (t**2 * (3 - 2 * t))
        bot = y0_mid - half + (y1_mid - half - (y0_mid - half)) * (t**2 * (3 - 2 * t))
        color = "#90be6d" if "OK" in dst else "#e76f51"
        ax.fill_between(xs, bot, top, color=color, alpha=0.35, linewidth=0)

    fig.text(
        0.5,
        0.02,
        "Proxy uses end-to-end correctness (not step raw_output diffs). "
        "Re-run plot_figures.py as CSV grows.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


CONFIG_ORDER = [
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

CONFIG_SHORT = {
    "qa-all-strong": "all-strong",
    "qa-cheap-decomposer": "cheap-decomp",
    "qa-cheap-hop-reader": "cheap-hop",
    "qa-cheap-composer": "cheap-comp",
    "qa-cheap-formatter": "cheap-fmt",
    "qa-qwen-decomposer": "qwen-decomp",
    "qa-qwen-hop-reader": "qwen-hop",
    "qa-qwen-composer": "qwen-comp",
    "qa-qwen-formatter": "qwen-fmt",
}


def _save(fig: plt.Figure, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_config_overview(stats: dict[str, dict], out: Path) -> None:
    """Side-by-side mean F1 and mean cost for every routing config."""
    cfgs = [c for c in CONFIG_ORDER if c in stats]
    if not cfgs:
        return
    labels = [CONFIG_SHORT[c] for c in cfgs]
    f1s = [stats[c]["f1"] for c in cfgs]
    costs = [stats[c]["cost"] for c in cfgs]
    ns = [stats[c]["n"] for c in cfgs]
    colors = []
    for c in cfgs:
        if "all-strong" in c:
            colors.append("#1f4e79")
        elif "cheap" in c:
            colors.append("#2a9d8f")
        else:
            colors.append("#e9c46a")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(len(cfgs))

    bars0 = ax0.bar(x, f1s, color=colors, edgecolor="#222")
    ax0.axhline(stats.get("qa-all-strong", {}).get("f1", np.nan), color="#1f4e79",
                ls="--", lw=1.2, label="all-strong F1")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=35, ha="right")
    ax0.set_ylabel("Mean F1")
    ax0.set_ylim(0, 1.05)
    ax0.set_title("Quality by config")
    ax0.legend(loc="lower right", fontsize=9)
    for bar, n in zip(bars0, ns):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"n={n}", ha="center", va="bottom", fontsize=8)

    ax1.bar(x, costs, color=colors, edgecolor="#222")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("Mean cost per question (USD)")
    ax1.set_title("Cost by config")

    fig.suptitle(
        "Config overview - each bar is one routing experiment "
        "(one weak step, or all Flash)",
        fontsize=13,
        y=1.02,
    )
    _save(fig, out)


def plot_f1_vs_cost_scatter(stats: dict[str, dict], out: Path) -> None:
    """Scatter of mean F1 vs mean cost - easy Pareto-style reading."""
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    markers = {"flash": "o", "flash-lite": "s", "qwen": "D"}
    for cfg, st in stats.items():
        step, model = CONFIG_STEP_MODEL.get(cfg, (None, "flash"))
        ax.scatter(
            st["cost"],
            st["f1"],
            s=80 + 4 * st["n"],
            marker=markers.get(model, "o"),
            c={"flash": "#1f4e79", "flash-lite": "#2a9d8f", "qwen": "#e9c46a"}.get(
                model, "#888"
            ),
            edgecolors="#222",
            zorder=3,
            label=CONFIG_SHORT.get(cfg, cfg),
        )
        ax.annotate(
            CONFIG_SHORT.get(cfg, cfg),
            (st["cost"], st["f1"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlabel("Mean cost per question (USD)  → cheaper is left")
    ax.set_ylabel("Mean F1  → better is up")
    ax.set_title(
        "F1 vs cost tradeoff\n"
        "Best region is up and left. Marker size grows with sample size n."
    )
    # Unique legend entries only for model families
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f4e79",
                   markeredgecolor="#222", markersize=9, label="Flash / all-strong"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2a9d8f",
                   markeredgecolor="#222", markersize=9, label="Flash-Lite swap"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#e9c46a",
                   markeredgecolor="#222", markersize=9, label="Qwen swap"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9)
    _save(fig, out)


def plot_f1_drop_by_step(acc: np.ndarray, out: Path) -> None:
    """F1 change vs all-strong when each step is swapped to Lite or Qwen."""
    strong = acc[:, MODELS.index("flash")]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(STEPS))
    width = 0.36

    lite_j = MODELS.index("flash-lite")
    qwen_j = MODELS.index("qwen")
    lite_drop = acc[:, lite_j] - strong
    qwen_drop = acc[:, qwen_j] - strong

    b0 = ax.bar(x - width / 2, lite_drop, width, label="Flash-Lite swap",
                color="#2a9d8f", edgecolor="#222")
    b1 = ax.bar(x + width / 2, qwen_drop, width, label="Qwen swap",
                color="#e9c46a", edgecolor="#222")
    ax.axhline(0.0, color="#333", lw=1.0)
    ax.axhline(-0.05, color="#e76f51", ls="--", lw=1.0, label="-0.05 F1 tolerance")
    ax.set_xticks(x)
    ax.set_xticklabels([STEP_LABELS[s] for s in STEPS])
    ax.set_ylabel("F1 change vs all-strong (positive = better)")
    ax.set_title(
        "Quality impact of cheapening one step\n"
        "Bars near zero mean that step tolerates a weaker model"
    )
    ax.legend(fontsize=9)
    for bars in (b0, b1):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.008 if h >= 0 else -0.018),
                f"{h:+.2f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
            )
    _save(fig, out)


def plot_sample_coverage(stats: dict[str, dict], out: Path) -> None:
    """How many questions finished per config (progress / confidence)."""
    cfgs = [c for c in CONFIG_ORDER if c in stats]
    labels = [CONFIG_SHORT[c] for c in cfgs]
    ns = [stats[c]["n"] for c in cfgs]
    colors = [
        "#1f4e79" if "all-strong" in c else "#2a9d8f" if "cheap" in c else "#e9c46a"
        for c in cfgs
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(cfgs))
    bars = ax.bar(x, ns, color=colors, edgecolor="#222")
    ax.axhline(150, color="#555", ls="--", lw=1.0, label="Target 150 / config")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Questions completed")
    ax.set_title("Sample coverage - more n means more trustworthy cells in other figures")
    ax.legend(fontsize=9)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(n), ha="center", va="bottom", fontsize=9)
    _save(fig, out)


def plot_propagation_rates(stats: dict[str, dict], out: Path) -> None:
    """Simple percent bars: among baseline-correct questions, share that fail when weak."""
    strong = stats.get("qa-all-strong")
    if strong is None:
        return
    strong_by_id = {r["question_id"]: r for r in strong["rows"] if r.get("question_id")}

    labels: list[str] = []
    rates: list[float] = []
    counts: list[str] = []
    colors: list[str] = []

    for cfg, (step, model) in CONFIG_STEP_MODEL.items():
        if step is None or cfg not in stats or model == "flash":
            continue
        ok_base = 0
        bad_mixed = 0
        for row in stats[cfg]["rows"]:
            base = strong_by_id.get(row.get("question_id") or "")
            if base is None:
                continue
            if not (_truthy(base.get("is_correct")) or _truthy(base.get("em"))):
                continue
            ok_base += 1
            mixed_ok = _truthy(row.get("is_correct")) or _truthy(row.get("em"))
            if not mixed_ok:
                bad_mixed += 1
        if ok_base == 0:
            continue
        labels.append(f"{STEP_LABELS[step]}\n{MODEL_LABELS[model]}")
        rates.append(100.0 * bad_mixed / ok_base)
        counts.append(f"{bad_mixed}/{ok_base}")
        colors.append("#2a9d8f" if model == "flash-lite" else "#e9c46a")

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of baseline-correct questions that become wrong")
    ax.set_title(
        "Error propagation rate by weak step (QA proxy)\n"
        "Lower is safer to cheapen"
    )
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                c, ha="center", va="bottom", fontsize=8)
    _save(fig, out)


def plot_em_f1_compare(stats: dict[str, dict], out: Path) -> None:
    """EM rate vs mean F1 for each config - two accuracy views."""
    cfgs = [c for c in CONFIG_ORDER if c in stats]
    labels = [CONFIG_SHORT[c] for c in cfgs]
    f1s = [stats[c]["f1"] for c in cfgs]
    ems = [stats[c]["em"] for c in cfgs]
    x = np.arange(len(cfgs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.bar(x - width / 2, f1s, width, label="Mean F1", color="#457b9d", edgecolor="#222")
    ax.bar(x + width / 2, ems, width, label="Exact match rate", color="#a8dadc",
           edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(
        "Two ways to score answers\n"
        "F1 allows partial credit; EM requires an exact match"
    )
    ax.legend(fontsize=9)
    _save(fig, out)


def maybe_sql_note() -> str:
    if SQL_CHEAP.is_file() and SQL_BASELINE.is_file():
        with SQL_CHEAP.open(encoding="utf-8", newline="") as f:
            n = sum(1 for _ in csv.DictReader(f))
        return f"SQL cheap-selector rows available: {n} (QA Sankey used above)."
    return "SQL baseline CSVs not found or empty; QA Sankey only."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-results",
        type=Path,
        default=QA_RESULTS,
        help="path to qa_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIG_DIR,
        help="output directory for figures",
    )
    parser.add_argument(
        "--drop-tol",
        type=float,
        default=0.05,
        help="max F1 drop vs all-strong when choosing a cheaper model",
    )
    args = parser.parse_args()

    style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_qa(args.qa_results)
    stats = config_stats(rows)
    if "qa-all-strong" not in stats:
        raise SystemExit("qa-all-strong rows required in the CSV")

    acc, cost, nmat = step_model_matrix(stats)
    strong_f1 = stats["qa-all-strong"]["f1"]
    routing = recommend_routing(acc, cost, strong_f1, drop_tol=args.drop_tol)

    core = {
        "accuracy_cost_heatmap.png": lambda p: plot_heatmap(acc, cost, nmat, p),
        "error_propagation_sankey.png": lambda p: plot_sankey(
            qa_propagation_flows(stats), p
        ),
        "routing_map.png": lambda p: plot_routing_map(
            routing, acc, cost, strong_f1, p
        ),
        "cost_savings_bars.png": lambda p: plot_cost_bars(stats, routing, p),
    }
    extra = {
        "config_overview.png": lambda p: plot_config_overview(stats, p),
        "f1_vs_cost_scatter.png": lambda p: plot_f1_vs_cost_scatter(stats, p),
        "f1_drop_by_step.png": lambda p: plot_f1_drop_by_step(acc, p),
        "sample_coverage.png": lambda p: plot_sample_coverage(stats, p),
        "propagation_rate_bars.png": lambda p: plot_propagation_rates(stats, p),
        "em_vs_f1_by_config.png": lambda p: plot_em_f1_compare(stats, p),
    }

    written: list[str] = []
    for name, fn in {**core, **extra}.items():
        path = args.out_dir / name
        fn(path)
        written.append(name)

    # Sidecar summary for teammates / report captions
    summary_path = args.out_dir / "figure_summary.txt"
    lines = [
        f"source: {args.qa_results}",
        f"total QA rows: {len(rows)}",
        f"all-strong n={stats['qa-all-strong']['n']} "
        f"F1={stats['qa-all-strong']['f1']:.3f} "
        f"cost=${stats['qa-all-strong']['cost']:.4f}",
        "recommended routing: "
        + ", ".join(f"{s}={routing[s]}" for s in STEPS),
        maybe_sql_note(),
        "figures:",
        *[f"  {name}" for name in written],
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote figures to", args.out_dir)
    for line in lines:
        print(" ", line)


if __name__ == "__main__":
    main()
