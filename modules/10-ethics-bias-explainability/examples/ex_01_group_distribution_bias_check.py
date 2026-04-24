"""Group distribution check for bias diagnostics.

Run:
    python modules/10-ethics-bias-explainability/examples/ex_01_group_distribution_bias_check.py
"""

from __future__ import annotations


def distribution(records: list[tuple[str, int]]) -> dict[str, float]:
    """Return positive-rate distribution by group."""
    totals: dict[str, int] = {}
    positives: dict[str, int] = {}

    for group, outcome in records:
        totals[group] = totals.get(group, 0) + 1
        positives[group] = positives.get(group, 0) + outcome

    return {group: positives[group] / totals[group] for group in totals}


def main() -> None:
    """Print positive-rate per group and max disparity."""
    records = [
        ("group_a", 1),
        ("group_a", 0),
        ("group_a", 1),
        ("group_b", 1),
        ("group_b", 0),
        ("group_b", 0),
        ("group_c", 1),
        ("group_c", 1),
    ]

    rates = distribution(records)
    max_gap = max(rates.values()) - min(rates.values())

    print(f"positive_rates={rates}")
    print(f"max_rate_gap={max_gap:.4f}")


if __name__ == "__main__":
    main()
