"""Show the effect of L2 regularization on linear model weights.

Run:
    python modules/03-machine-learning-fundamentals/examples/ex_03_regularization_tradeoffs.py
"""

from __future__ import annotations


def fit_weight(x_values: list[float], y_values: list[float], l2_lambda: float) -> float:
    """Fit a one-parameter model y = w*x using closed-form ridge estimate."""
    numerator = sum(x * y for x, y in zip(x_values, y_values, strict=True))
    denominator = sum(x * x for x in x_values) + l2_lambda
    return numerator / denominator


def mse(x_values: list[float], y_values: list[float], weight: float) -> float:
    """Compute mean squared error."""
    errors = [(weight * x - y) ** 2 for x, y in zip(x_values, y_values, strict=True)]
    return sum(errors) / len(errors)


def main() -> None:
    """Compare train/test behavior as regularization strength increases."""
    x_train = [1, 2, 3, 4, 5]
    y_train = [2.2, 3.9, 6.1, 8.2, 9.8]
    x_test = [6, 7]
    y_test = [12.1, 13.7]

    for l2_lambda in [0.0, 0.5, 5.0]:
        weight = fit_weight(x_train, y_train, l2_lambda)
        train_mse = mse(x_train, y_train, weight)
        test_mse = mse(x_test, y_test, weight)

        print(f"lambda={l2_lambda:.1f} | w={weight:.4f} | train_mse={train_mse:.4f} | test_mse={test_mse:.4f}")


if __name__ == "__main__":
    main()
