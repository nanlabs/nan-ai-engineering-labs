"""Image matrix basics: grayscale image stats and normalization.

Run:
    python modules/06-computer-vision/examples/ex_01_image_matrix_basics.py
"""

from __future__ import annotations


def normalize(image: list[list[int]]) -> list[list[float]]:
    """Normalize 0-255 image into 0-1 range."""
    return [[pixel / 255.0 for pixel in row] for row in image]


def image_stats(image: list[list[int]]) -> tuple[int, int, float]:
    """Return min, max, mean for grayscale image."""
    flat = [pixel for row in image for pixel in row]
    return min(flat), max(flat), sum(flat) / len(flat)


def main() -> None:
    """Run stats and normalization on a tiny grayscale image."""
    image = [
        [10, 20, 30],
        [40, 120, 220],
        [60, 90, 255],
    ]

    min_v, max_v, mean_v = image_stats(image)
    normalized = normalize(image)

    print(f"stats min={min_v} max={max_v} mean={mean_v:.2f}")
    print(f"normalized_center={normalized[1][1]:.4f}")


if __name__ == "__main__":
    main()
