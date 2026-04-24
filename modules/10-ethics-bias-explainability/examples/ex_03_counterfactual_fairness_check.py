"""Counterfactual fairness check by feature perturbation.

Run:
    python modules/10-ethics-bias-explainability/examples/ex_03_counterfactual_fairness_check.py
"""

from __future__ import annotations


def score_model(income: float, years_experience: float, protected_flag: int) -> float:
    """Toy model score including protected attribute influence."""
    return 0.3 * income + 0.7 * years_experience + 0.2 * protected_flag


def main() -> None:
    """Compare score with only protected attribute changed."""
    income = 7.5
    years_experience = 4.0

    factual = score_model(income, years_experience, protected_flag=0)
    counterfactual = score_model(income, years_experience, protected_flag=1)

    delta = abs(counterfactual - factual)

    print(f"factual_score={factual:.4f}")
    print(f"counterfactual_score={counterfactual:.4f}")
    print(f"counterfactual_delta={delta:.4f}")


if __name__ == "__main__":
    main()
