"""Simulate how model depth can overfit tiny datasets.

Run:
    python modules/03-machine-learning-fundamentals/examples/ex_04_tree_depth_vs_overfitting.py
"""

from __future__ import annotations


def baseline_rule(x_value: float) -> int:
    """Low-depth rule equivalent to a simple split."""
    return 1 if x_value >= 0.5 else 0


def overfit_rule(index: int, x_value: float) -> int:
    """Artificially memorize train indexes (high-depth behavior)."""
    memorize = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}
    if index in memorize:
        return memorize[index]
    return 1 if x_value >= 0.5 else 0


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Compute accuracy."""
    hits = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return hits / len(y_true)


def main() -> None:
    """Compare train vs test metrics for a simple and overfit strategy."""
    x_train = [0.1, 0.7, 0.2, 0.9, 0.3, 0.8]
    y_train = [0, 1, 0, 1, 0, 1]
    x_test = [0.15, 0.55, 0.85]
    y_test = [0, 1, 1]

    baseline_train = [baseline_rule(x) for x in x_train]
    baseline_test = [baseline_rule(x) for x in x_test]

    overfit_train = [overfit_rule(i, x) for i, x in enumerate(x_train)]
    overfit_test = [overfit_rule(i + len(x_train), x) for i, x in enumerate(x_test)]

    print("Baseline-like depth")
    print(f"train_acc={accuracy(y_train, baseline_train):.4f} | test_acc={accuracy(y_test, baseline_test):.4f}")
    print()
    print("Overfit-like depth")
    print(f"train_acc={accuracy(y_train, overfit_train):.4f} | test_acc={accuracy(y_test, overfit_test):.4f}")


if __name__ == "__main__":
    main()
