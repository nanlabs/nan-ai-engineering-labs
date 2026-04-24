"""Item-similarity recommender using cosine similarity.

Run:
    python modules/08-recommender-systems/examples/ex_02_item_similarity_recommender.py
"""

from __future__ import annotations

import math


def cosine(a_values: list[float], b_values: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(a_values, b_values, strict=True))
    norm_a = math.sqrt(sum(a * a for a in a_values))
    norm_b = math.sqrt(sum(b * b for b in b_values))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    """Compute similarities and recommend items close to item_a."""
    item_vectors = {
        "item_a": [5, 4, 0, 0],
        "item_b": [4, 5, 0, 1],
        "item_c": [0, 0, 5, 4],
        "item_d": [1, 0, 4, 5],
    }

    target = "item_a"
    similarities: list[tuple[str, float]] = []
    for item, vector in item_vectors.items():
        if item == target:
            continue
        score = cosine(item_vectors[target], vector)
        similarities.append((item, score))

    ranked = sorted(similarities, key=lambda pair: pair[1], reverse=True)
    print(f"target={target}")
    print(f"ranked_neighbors={ranked}")


if __name__ == "__main__":
    main()
