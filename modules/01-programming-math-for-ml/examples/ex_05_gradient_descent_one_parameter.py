"""
Optimize one parameter with gradient descent on mean squared error.

Run:
    python modules/01-programming-math-for-ml/examples/ex_05_gradient_descent_one_parameter.py
"""

from __future__ import annotations


def mse(w: float, x: list[float], y: list[float]) -> float:
    """Compute mean squared error for y_hat = w*x."""
    errors = [(w * xi - yi) ** 2 for xi, yi in zip(x, y, strict=True)]
    return sum(errors) / len(errors)


def gradient(w: float, x: list[float], y: list[float]) -> float:
    """Compute derivative of MSE with respect to w."""
    terms = [2 * xi * (w * xi - yi) for xi, yi in zip(x, y, strict=True)]
    return sum(terms) / len(terms)


def main() -> None:
    """Run a deterministic gradient-descent optimization."""
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]

    w = 0.0
    lr = 0.05

    print("Gradient descent baseline")
    for step in range(1, 11):
        grad = gradient(w, x, y)
        w = w - lr * grad
        loss = mse(w, x, y)
        print(f"step={step:02d} w={w:.4f} mse={loss:.6f}")


if __name__ == "__main__":
    main()
