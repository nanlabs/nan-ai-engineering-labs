"""Generate a concise model risk report from metrics.

Run:
    python modules/10-ethics-bias-explainability/examples/ex_05_risk_report_generator.py
"""

from __future__ import annotations


def risk_level(dp_difference: float, counterfactual_delta: float) -> str:
    """Assign qualitative risk level from fairness indicators."""
    if dp_difference > 0.25 or counterfactual_delta > 0.20:
        return "high"
    if dp_difference > 0.10 or counterfactual_delta > 0.10:
        return "medium"
    return "low"


def main() -> None:
    """Print compact risk summary for governance review."""
    dp_diff = 0.18
    cf_delta = 0.12
    level = risk_level(dp_diff, cf_delta)

    report = {
        "model": "credit_decision_v1",
        "demographic_parity_difference": dp_diff,
        "counterfactual_delta": cf_delta,
        "risk_level": level,
        "recommended_action": "review thresholds and retrain with debiasing constraints",
    }

    print(report)


if __name__ == "__main__":
    main()
