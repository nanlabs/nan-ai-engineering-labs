"""Minimal reproducible ML pipeline using deterministic synthetic data.

Run:
    python modules/03-machine-learning-fundamentals/examples/ex_06_reproducible_ml_pipeline.py
"""

from __future__ import annotations

import random


def generate_data(seed: int) -> list[tuple[float, int]]:
    """Generate deterministic synthetic samples."""
    random.seed(seed)
    samples: list[tuple[float, int]] = []
    for _ in range(20):
        x = random.random()
        label = 1 if x >= 0.5 else 0
        samples.append((x, label))
    return samples


def split(samples: list[tuple[float, int]], ratio: float = 0.8) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    cut = int(len(samples) * ratio)
    return samples[:cut], samples[cut:]


def train_threshold_model(train: list[tuple[float, int]]) -> float:
    """Fit a threshold from train means per class."""
    class0 = [x for x, label in train if label == 0]
    class1 = [x for x, label in train if label == 1]
    return (sum(class0) / len(class0) + sum(class1) / len(class1)) / 2


def evaluate(test: list[tuple[float, int]], threshold: float) -> float:
    """Evaluate threshold classifier accuracy."""
    preds = [1 if x >= threshold else 0 for x, _ in test]
    labels = [label for _, label in test]
    hits = sum(1 for label, pred in zip(labels, preds, strict=True) if label == pred)
    return hits / len(labels)


def main() -> None:
    """Run a deterministic train/eval loop and print reproducible outputs."""
    samples = generate_data(seed=42)
    train, test = split(samples)
    threshold = train_threshold_model(train)
    acc = evaluate(test, threshold)

    print(f"samples={len(samples)} train={len(train)} test={len(test)}")
    print(f"threshold={threshold:.4f}")
    print(f"accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
