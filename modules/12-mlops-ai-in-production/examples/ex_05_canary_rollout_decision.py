"""Canary rollout decision demo.

Run:
    python modules/12-mlops-ai-in-production/examples/ex_05_canary_rollout_decision.py
"""

from __future__ import annotations


def should_promote(canary_error: float, baseline_error: float, canary_latency: int, baseline_latency: int) -> bool:
    """Promote canary if error does not regress and latency stays bounded."""
    return canary_error <= baseline_error and canary_latency <= baseline_latency * 1.1


def main() -> None:
    """Evaluate canary promotion decision."""
    baseline_error = 0.12
    canary_error = 0.10
    baseline_latency = 120
    canary_latency = 126

    promote = should_promote(canary_error, baseline_error, canary_latency, baseline_latency)
    print(f"promote_canary={promote}")


if __name__ == "__main__":
    main()
