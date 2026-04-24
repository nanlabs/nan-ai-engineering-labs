"""Run a deterministic mini math pipeline with fixed seed.

Run:
    python modules/01-programming-math-for-ml/examples/ex_06_reproducible_math_pipeline.py
"""

from __future__ import annotations

import random


def run_pipeline(seed: int) -> dict[str, float]:
    """Generate synthetic values and compute stable summary metrics."""
    random.seed(seed)
    values = [round(random.uniform(0, 1), 6) for _ in range(8)]

    avg = sum(values) / len(values)
    centered = [v - avg for v in values]
    energy = sum(v * v for v in centered)

    return {
        "mean": round(avg, 6),
        "energy": round(energy, 6),
    }


def main() -> None:
    """Check reproducibility with same seed and compare with a different seed."""
    run_a = run_pipeline(seed=42)
    run_b = run_pipeline(seed=42)
    run_c = run_pipeline(seed=43)

    print("Reproducible math pipeline")
    print(f"run_a: {run_a}")
    print(f"run_b: {run_b}")
    print(f"run_c: {run_c}")
    print(f"same_result: {run_a == run_b}")
    print(f"different_seed_changes_output: {run_a != run_c}")


if __name__ == "__main__":
    main()
