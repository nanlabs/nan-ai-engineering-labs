"""Simple feature drift detection demo.

Run:
    python modules/12-mlops-ai-in-production/examples/ex_04_drift_detection_demo.py
"""

from __future__ import annotations


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def drift_score(reference: list[float], production: list[float]) -> float:
    """Use difference of means as a lightweight drift score."""
    return abs(mean(reference) - mean(production))


def main() -> None:
    """Compare reference and production distributions."""
    reference = [0.10, 0.12, 0.11, 0.10, 0.13]
    production = [0.20, 0.18, 0.22, 0.19, 0.21]

    score = drift_score(reference, production)
    print(f"drift_score={score:.4f}")
    print(f"drift_alert={score > 0.05}")


if __name__ == "__main__":
    main()
