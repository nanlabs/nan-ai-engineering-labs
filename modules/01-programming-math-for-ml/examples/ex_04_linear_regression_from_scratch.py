"""Fit one-feature linear regression with closed-form least squares.

Run:
    python modules/01-programming-math-for-ml/examples/ex_04_linear_regression_from_scratch.py
"""

from __future__ import annotations


def fit_linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return slope and intercept from closed-form solution."""
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=True))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def predict(x: list[float], slope: float, intercept: float) -> list[float]:
    """Predict y values using y = m*x + b."""
    return [slope * xi + intercept for xi in x]


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Return mean absolute error."""
    return sum(abs(a - b) for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)


def main() -> None:
    """Train and evaluate a deterministic linear baseline."""
    x = [1, 2, 3, 4, 5, 6]
    y = [2.1, 4.2, 5.8, 8.1, 10.2, 12.1]

    slope, intercept = fit_linear_regression(x, y)
    y_hat = predict(x, slope, intercept)

    print("Linear regression from scratch")
    print(f"slope: {slope:.4f}")
    print(f"intercept: {intercept:.4f}")
    print(f"mae: {mae(y, y_hat):.4f}")


if __name__ == "__main__":
    main()
