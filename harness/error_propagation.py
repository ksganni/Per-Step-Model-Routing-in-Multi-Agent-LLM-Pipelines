"""Decide whether a weak step's error propagated to a wrong final answer.

A routing config with one weak step (for example Flash-Lite as the Selector
and Flash for every other step) is compared question by question against an
all-strong baseline run of the same questions.

error_propagated is True when all three hold:

1. the all-strong baseline answered the question correctly,
2. the mixed config answered it incorrectly, and
3. the weak step's output differs from the baseline's output for that step.

It is False when any of them fails, including when the weak model never ran
for the question (for example a weak refiner that was never triggered).

It is blank ("") when the question cannot be judged: no baseline rows, a
baseline that is not all one model, more than one weak step, or a baseline
or mixed answer that could not be scored.

The inputs are CSV-style rows as written by experiment_runner, so the same
function works on rows in memory and on rows read back from a results file.
"""

from __future__ import annotations

from pipelines.sql.steps import STEP_SELECTOR, parse_selector


def _is_true(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _is_false(value: object) -> bool:
    return value is False or str(value).strip().lower() == "false"


def _first_output(rows: list[dict], step: str) -> str | None:
    """raw_output of the first call to ``step``, or None if the step never ran."""
    for row in rows:
        if row.get("step") == step:
            return row.get("raw_output") or ""
    return None


def _normalise_selection(raw: str) -> object:
    """Parse a Selector reply so equal table choices compare equal.

    Column order inside a table does not matter, and neither does JSON
    formatting. A reply that cannot be parsed falls back to its text.
    """
    selection, ok = parse_selector(raw)
    if not ok or selection is None:
        return " ".join(raw.split())
    return {
        table: choice if isinstance(choice, str) else tuple(sorted(choice))
        for table, choice in selection.items()
    }


def _normalise_output(step: str, raw: str) -> object:
    if step == STEP_SELECTOR:
        return _normalise_selection(raw)
    return " ".join(raw.split())


def step_output_changed(step: str, mixed_rows: list[dict], baseline_rows: list[dict]) -> bool:
    """True when the mixed run's output for ``step`` differs from the baseline's.

    A step that ran in only one of the two runs counts as changed.
    """
    mixed = _first_output(mixed_rows, step)
    baseline = _first_output(baseline_rows, step)
    if mixed is None or baseline is None:
        return (mixed is None) != (baseline is None)
    return _normalise_output(step, mixed) != _normalise_output(step, baseline)


def error_propagated(mixed_rows: list[dict], baseline_rows: list[dict]) -> bool | str:
    """Return True, False, or "" for one question (see the module docstring).

    Args:
        mixed_rows: The question's step rows from the one-weak-step config.
        baseline_rows: The same question's step rows from the all-strong run.
    """
    baseline_steps = [r for r in baseline_rows if r.get("step")]
    mixed_steps = [r for r in mixed_rows if r.get("step")]
    if not baseline_steps or not mixed_steps:
        return ""

    baseline_models = {r.get("model") for r in baseline_steps}
    if len(baseline_models) != 1:
        return ""
    strong = baseline_models.pop()

    weak_steps = sorted({r["step"] for r in mixed_steps if r.get("model") != strong})
    if len(weak_steps) > 1:
        return ""

    baseline_correct = baseline_steps[0].get("is_correct")
    mixed_correct = mixed_steps[0].get("is_correct")
    if not (_is_true(baseline_correct) or _is_false(baseline_correct)):
        return ""
    if not (_is_true(mixed_correct) or _is_false(mixed_correct)):
        return ""

    if not weak_steps:
        return False
    if not (_is_true(baseline_correct) and _is_false(mixed_correct)):
        return False
    return step_output_changed(weak_steps[0], mixed_steps, baseline_steps)
