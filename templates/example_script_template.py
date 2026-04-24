"""Example template for training-ai executable labs.

Fill this script with module-specific logic while keeping the structure.
"""

from __future__ import annotations


def build_sample_input() -> list[float]:
    """Return small deterministic input data for reproducible runs."""
    return [1.0, 2.0, 3.0, 4.0]


def run_example(values: list[float], *, scale: float = 1.0) -> dict[str, float]:
    """Run the core example logic and return result metrics."""
    if not values:
        msg = "values must not be empty"
        raise ValueError(msg)

    scaled = [value * scale for value in values]
    total = sum(scaled)
    mean = total / len(scaled)
    return {"total": total, "mean": mean}


def main() -> None:
    """Execute the happy-path run and print deterministic output."""
    values = build_sample_input()
    result = run_example(values, scale=2.0)
    print(f"Input size: {len(values)}")
    print(f"Result total: {result['total']:.2f}")
    print(f"Result mean: {result['mean']:.2f}")


if __name__ == "__main__":
    main()
