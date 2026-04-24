"""Baseline binary classification with deterministic train/test split.

Run:
    python modules/03-machine-learning-fundamentals/examples/ex_01_train_test_split_baseline.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    """Simple 2D sample with binary target."""

    x1: float
    x2: float
    label: int


def build_dataset() -> list[Sample]:
    """Return a deterministic synthetic binary classification dataset."""
    return [
        Sample(0.1, 0.2, 0),
        Sample(0.2, 0.1, 0),
        Sample(0.3, 0.4, 0),
        Sample(1.0, 0.9, 1),
        Sample(0.9, 1.1, 1),
        Sample(1.2, 0.8, 1),
        Sample(0.4, 0.3, 0),
        Sample(1.1, 1.0, 1),
        Sample(0.25, 0.35, 0),
        Sample(0.95, 0.85, 1),
    ]


def deterministic_split(dataset: list[Sample], test_size: float = 0.2) -> tuple[list[Sample], list[Sample]]:
    """Split dataset preserving order for reproducibility."""
    test_count = max(1, int(len(dataset) * test_size))
    train = dataset[:-test_count]
    test = dataset[-test_count:]
    return train, test


def centroid(train: list[Sample], label: int) -> tuple[float, float]:
    """Compute centroid of samples for a given label."""
    subset = [sample for sample in train if sample.label == label]
    x1_mean = sum(sample.x1 for sample in subset) / len(subset)
    x2_mean = sum(sample.x2 for sample in subset) / len(subset)
    return x1_mean, x2_mean


def predict(sample: Sample, c0: tuple[float, float], c1: tuple[float, float]) -> int:
    """Predict class by closest centroid in euclidean space."""
    d0 = (sample.x1 - c0[0]) ** 2 + (sample.x2 - c0[1]) ** 2
    d1 = (sample.x1 - c1[0]) ** 2 + (sample.x2 - c1[1]) ** 2
    return 0 if d0 <= d1 else 1


def f1_score(y_true: list[int], y_pred: list[int]) -> float:
    """Compute binary F1 score."""
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    """Train a baseline model and print reproducible metrics."""
    dataset = build_dataset()
    train, test = deterministic_split(dataset, test_size=0.2)

    c0 = centroid(train, label=0)
    c1 = centroid(train, label=1)

    y_true = [sample.label for sample in test]
    y_pred = [predict(sample, c0, c1) for sample in test]

    accuracy = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(y_true)
    f1 = f1_score(y_true, y_pred)

    print("Dataset: synthetic_binary")
    print(f"Train size: {len(train)} | Test size: {len(test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
