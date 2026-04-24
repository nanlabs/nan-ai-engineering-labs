"""Gradient descent baseline for y = w*x regression.

Run:
    python modules/04-deep-learning-basics/examples/ex_02_gradient_descent_baseline.py
"""

from __future__ import annotations


def mse_and_grad(weight: float, x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    """Return MSE and gradient dMSE/dw."""
    errors = [(weight * x - y) for x, y in zip(x_values, y_values, strict=True)]
    mse = sum(err * err for err in errors) / len(errors)
    grad = 2 * sum(err * x for err, x in zip(errors, x_values, strict=True)) / len(errors)
    return mse, grad


def main() -> None:
    """Train one weight with deterministic gradient descent."""
    x_values = [1.0, 2.0, 3.0, 4.0]
    y_values = [2.0, 4.1, 5.9, 8.2]

    weight = 0.0
    lr = 0.05

    for epoch in range(1, 11):
        mse, grad = mse_and_grad(weight, x_values, y_values)
        weight -= lr * grad
        print(f"epoch={epoch:02d} mse={mse:.4f} grad={grad:.4f} w={weight:.4f}")


if __name__ == "__main__":
    main()
