"""Compute dot product and cosine similarity with pure Python vectors.

Run:
    python modules/01-programming-math-for-ml/examples/ex_01_vector_dot_product_baseline.py
"""

from __future__ import annotations

import math


def dot_product(a: list[float], b: list[float]) -> float:
    """Return dot product for two vectors of the same size."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(v: list[float]) -> float:
    """Return Euclidean norm."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity in [-1, 1]."""
    return dot_product(a, b) / (norm(a) * norm(b))


def main() -> None:
    """Run a deterministic vector similarity baseline."""
    v1 = [2.0, 1.0, 0.0]
    v2 = [1.5, 1.0, 0.5]
    v3 = [0.0, 1.0, 2.0]

    print("Vector baseline")
    print(f"dot(v1, v2): {dot_product(v1, v2):.4f}")
    print(f"cos(v1, v2): {cosine_similarity(v1, v2):.4f}")
    print(f"cos(v1, v3): {cosine_similarity(v1, v3):.4f}")


if __name__ == "__main__":
    main()
