"""Reproducible MLOps runbook execution demo.

Run:
    python modules/12-mlops-ai-in-production/examples/ex_06_reproducible_mlops_runbook.py
"""

from __future__ import annotations

import random


def generate_run(seed: int) -> dict[str, str]:
    """Generate deterministic run metadata."""
    random.seed(seed)
    return {
        "run_id": f"run-{random.randint(1000, 9999)}",
        "status": "success",
        "registered_model": f"model-v{random.randint(1, 3)}.{random.randint(0, 9)}",
    }


def main() -> None:
    """Verify deterministic runbook metadata with a fixed seed."""
    run_a = generate_run(seed=33)
    run_b = generate_run(seed=33)

    print(f"run_a={run_a}")
    print(f"run_b={run_b}")
    print(f"same_result={run_a == run_b}")


if __name__ == "__main__":
    main()
