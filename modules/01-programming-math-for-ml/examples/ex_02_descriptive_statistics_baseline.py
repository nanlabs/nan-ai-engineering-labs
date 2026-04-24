"""Compute baseline descriptive statistics on a small numeric dataset.

Run:
    python modules/01-programming-math-for-ml/examples/ex_02_descriptive_statistics_baseline.py
"""

from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    """Return arithmetic mean."""
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    """Return median for odd/even length lists."""
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def variance(values: list[float]) -> float:
    """Return population variance."""
    mu = mean(values)
    return sum((x - mu) ** 2 for x in values) / len(values)


def std_dev(values: list[float]) -> float:
    """Return population standard deviation."""
    return math.sqrt(variance(values))


def main() -> None:
    """Print deterministic descriptive statistics."""
    data = [10, 12, 11, 13, 15, 18, 17, 16, 14, 13]

    print("Descriptive statistics baseline")
    print(f"count: {len(data)}")
    print(f"mean: {mean(data):.4f}")
    print(f"median: {median(data):.4f}")
    print(f"variance: {variance(data):.4f}")
    print(f"std_dev: {std_dev(data):.4f}")


if __name__ == "__main__":
    main()
