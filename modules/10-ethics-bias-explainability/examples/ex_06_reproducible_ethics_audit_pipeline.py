"""Reproducible ethics audit pipeline with deterministic sampling.

Run:
    python modules/10-ethics-bias-explainability/examples/ex_06_reproducible_ethics_audit_pipeline.py
"""

from __future__ import annotations

import random


def sample_predictions(seed: int) -> list[tuple[str, int]]:
    """Generate deterministic group predictions."""
    random.seed(seed)
    groups = ["group_a", "group_b", "group_c"]
    rows: list[tuple[str, int]] = []
    for group in groups:
        for _ in range(4):
            rows.append((group, int(random.random() > 0.4)))
    return rows


def group_positive_rates(rows: list[tuple[str, int]]) -> dict[str, float]:
    """Compute positive prediction rates by group."""
    totals: dict[str, int] = {}
    positives: dict[str, int] = {}
    for group, pred in rows:
        totals[group] = totals.get(group, 0) + 1
        positives[group] = positives.get(group, 0) + pred
    return {group: positives[group] / totals[group] for group in totals}


def main() -> None:
    """Run deterministic audit twice and verify reproducibility."""
    rows_a = sample_predictions(seed=17)
    rows_b = sample_predictions(seed=17)

    rates_a = group_positive_rates(rows_a)
    rates_b = group_positive_rates(rows_b)

    print(f"rates_a={rates_a}")
    print(f"rates_b={rates_b}")
    print(f"same_result={rates_a == rates_b}")


if __name__ == "__main__":
    main()
