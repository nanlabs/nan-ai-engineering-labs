"""Compare simple regression and classification baselines without external libs.

Run:
    python modules/03-machine-learning-fundamentals/examples/ex_02_linear_vs_logistic_baseline.py
"""

from __future__ import annotations


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    """Compute MAE metric."""
    errors = [abs(a - b) for a, b in zip(y_true, y_pred, strict=True)]
    return sum(errors) / len(errors)


def regression_baseline() -> float:
    """Train a tiny linear model y = a*x + b with closed-form slope/intercept."""
    x_values = [1, 2, 3, 4, 5, 6, 7, 8]
    y_values = [3, 5, 7, 9, 11, 13, 15, 17]

    x_train, x_test = x_values[:6], x_values[6:]
    y_train, y_test = y_values[:6], y_values[6:]

    x_mean = sum(x_train) / len(x_train)
    y_mean = sum(y_train) / len(y_train)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_train, y_train, strict=True))
    denominator = sum((x - x_mean) ** 2 for x in x_train)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    y_pred = [slope * x + intercept for x in x_test]
    return mean_absolute_error(y_test, y_pred)


def classification_baseline() -> float:
    """Classify with a linear score threshold as a lightweight baseline."""
    samples = [
        (0.1, 0.2, 0),
        (0.2, 0.1, 0),
        (0.3, 0.4, 0),
        (1.0, 0.9, 1),
        (0.9, 1.1, 1),
        (1.2, 0.8, 1),
        (0.4, 0.3, 0),
        (1.1, 1.0, 1),
    ]

    train = samples[:6]
    test = samples[6:]

    threshold = sum((x1 + x2) for x1, x2, _ in train) / len(train)
    predictions = [1 if (x1 + x2) >= threshold else 0 for x1, x2, _ in test]
    labels = [label for _, _, label in test]

    correct = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == pred)
    return correct / len(labels)


def main() -> None:
    """Print baseline metrics for both tasks."""
    mae = regression_baseline()
    accuracy = classification_baseline()

    print("Regression baseline: synthetic_linear")
    print(f"MAE: {mae:.4f}")
    print()
    print("Classification baseline: synthetic_binary")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
