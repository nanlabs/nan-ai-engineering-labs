"""Run a deterministic mini cleaning pipeline with a fixed random seed.

Run:
    python modules/02-data-collection-cleaning-visualization/examples/ex_06_reproducible_cleaning_pipeline.py
"""

from __future__ import annotations

import random


def run_pipeline(seed: int) -> dict[str, object]:
    """Generate synthetic rows and compute stable cleaning metrics."""
    random.seed(seed)

    rows: list[dict[str, float | None]] = []
    for idx in range(8):
        value = round(random.uniform(10, 50), 2)
        maybe_missing = None if idx in {2, 6} else value
        rows.append({"id": idx + 1, "metric": maybe_missing})

    observed = [row["metric"] for row in rows if row["metric"] is not None]
    fill_value = round(sum(observed) / len(observed), 2)

    imputed = [fill_value if row["metric"] is None else row["metric"] for row in rows]
    checksum = round(sum(imputed), 4)

    return {
        "fill_value": fill_value,
        "checksum": checksum,
        "missing_count": sum(1 for row in rows if row["metric"] is None),
    }


def main() -> None:
    """Verify reproducibility across repeated runs with same seed."""
    run_a = run_pipeline(seed=7)
    run_b = run_pipeline(seed=7)
    run_c = run_pipeline(seed=8)

    print("Reproducible cleaning pipeline")
    print(f"run_a: {run_a}")
    print(f"run_b: {run_b}")
    print(f"run_c: {run_c}")
    print(f"same_result: {run_a == run_b}")
    print(f"different_seed_changes_output: {run_a != run_c}")


if __name__ == "__main__":
    main()
