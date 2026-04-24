"""Demographic parity difference calculation.

Run:
    python modules/10-ethics-bias-explainability/examples/ex_02_demographic_parity_metric.py
"""

from __future__ import annotations


def positive_rate(predictions: list[int]) -> float:
    """Compute positive prediction rate."""
    return sum(predictions) / len(predictions) if predictions else 0.0


def demographic_parity_difference(group_a: list[int], group_b: list[int]) -> float:
    """Absolute gap in positive rates between two groups."""
    return abs(positive_rate(group_a) - positive_rate(group_b))


def main() -> None:
    """Report demographic parity metrics for two groups."""
    pred_group_a = [1, 1, 0, 1, 0, 1]
    pred_group_b = [1, 0, 0, 0, 0, 1]

    rate_a = positive_rate(pred_group_a)
    rate_b = positive_rate(pred_group_b)
    dp_diff = demographic_parity_difference(pred_group_a, pred_group_b)

    print(f"rate_group_a={rate_a:.4f}")
    print(f"rate_group_b={rate_b:.4f}")
    print(f"dp_difference={dp_diff:.4f}")


if __name__ == "__main__":
    main()
