"""
Compare min-max scaling and z-score normalization.

Run:
    python modules/01-programming-math-for-ml/examples/ex_03_feature_scaling_normalization.py
"""

from __future__ import annotations


def min_max_scale(values: list[float]) -> list[float]:
    """Scale values to [0, 1]."""
    low = min(values)
    high = max(values)
    return [(x - low) / (high - low) for x in values]


def z_score_normalize(values: list[float]) -> list[float]:
    """Normalize values to zero mean and unit variance."""
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    sigma = var ** 0.5
    return [(x - mu) / sigma for x in values]


def main() -> None:
    """Show two baseline scaling strategies."""
    feature = [150, 160, 170, 180, 190, 200]

    scaled = min_max_scale(feature)
    normalized = z_score_normalize(feature)

    print("Feature scaling baseline")
    print(f"original:   {feature}")
    print(f"min_max:    {[round(x, 4) for x in scaled]}")
    print(f"z_score:    {[round(x, 4) for x in normalized]}")


if __name__ == "__main__":
    main()
