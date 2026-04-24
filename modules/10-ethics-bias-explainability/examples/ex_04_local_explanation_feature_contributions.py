"""Local explanation via linear feature contribution breakdown.

Run:
    python modules/10-ethics-bias-explainability/examples/ex_04_local_explanation_feature_contributions.py
"""

from __future__ import annotations


def local_contributions(features: dict[str, float], weights: dict[str, float], bias: float) -> tuple[dict[str, float], float]:
    """Compute per-feature contributions and final prediction."""
    contribs = {name: features[name] * weights[name] for name in features}
    prediction = sum(contribs.values()) + bias
    return contribs, prediction


def main() -> None:
    """Print local explanation for one synthetic prediction."""
    features = {"income": 7.0, "experience": 4.0, "debt_ratio": 2.0}
    weights = {"income": 0.4, "experience": 0.3, "debt_ratio": -0.5}
    bias = 0.2

    contribs, prediction = local_contributions(features, weights, bias)

    print(f"feature_contributions={contribs}")
    print(f"prediction={prediction:.4f}")


if __name__ == "__main__":
    main()
