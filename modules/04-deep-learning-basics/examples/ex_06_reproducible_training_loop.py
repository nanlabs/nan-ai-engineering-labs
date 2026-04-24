"""Reproducible mini training loop with deterministic updates.

Run:
    python modules/04-deep-learning-basics/examples/ex_06_reproducible_training_loop.py
"""

from __future__ import annotations

import random


def run_training(seed: int) -> tuple[float, float]:
    """Return final weight and loss after deterministic pseudo-training."""
    random.seed(seed)
    weight = random.uniform(-0.5, 0.5)
    x_values = [1.0, 2.0, 3.0]
    y_values = [2.1, 3.9, 6.2]

    lr = 0.03
    loss = 0.0
    for _ in range(20):
        errors = [(weight * x - y) for x, y in zip(x_values, y_values, strict=True)]
        loss = sum(err * err for err in errors) / len(errors)
        grad = 2 * sum(err * x for err, x in zip(errors, x_values, strict=True)) / len(errors)
        weight -= lr * grad

    return weight, loss


def main() -> None:
    """Show same outputs when using same seed."""
    weight_a, loss_a = run_training(seed=7)
    weight_b, loss_b = run_training(seed=7)

    print(f"run_a weight={weight_a:.6f} loss={loss_a:.6f}")
    print(f"run_b weight={weight_b:.6f} loss={loss_b:.6f}")
    print(f"same_result={weight_a == weight_b and loss_a == loss_b}")


if __name__ == "__main__":
    main()
