"""Reproducible CV mini pipeline with deterministic augmentation.

Run:
    python modules/06-computer-vision/examples/ex_06_cv_pipeline_reproducible.py
"""

from __future__ import annotations

import random


Image = list[list[int]]


def augment_flip(image: Image, seed: int) -> Image:
    """Flip image horizontally with deterministic randomness."""
    random.seed(seed)
    do_flip = random.random() > 0.5
    if not do_flip:
        return [row[:] for row in image]
    return [list(reversed(row)) for row in image]


def average_intensity(image: Image) -> float:
    """Compute mean pixel intensity."""
    flat = [pixel for row in image for pixel in row]
    return sum(flat) / len(flat)


def run(seed: int) -> tuple[bool, float]:
    """Run deterministic augmentation and scoring."""
    image = [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
    ]
    augmented = augment_flip(image, seed=seed)
    return augmented != image, average_intensity(augmented)


def main() -> None:
    """Show reproducibility with same seed."""
    flip_a, avg_a = run(seed=11)
    flip_b, avg_b = run(seed=11)

    print(f"run_a flipped={flip_a} avg={avg_a:.2f}")
    print(f"run_b flipped={flip_b} avg={avg_b:.2f}")
    print(f"same_result={flip_a == flip_b and avg_a == avg_b}")


if __name__ == "__main__":
    main()
