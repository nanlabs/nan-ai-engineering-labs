"""User-similarity recommender using nearest-neighbor idea.

Run:
    python modules/08-recommender-systems/examples/ex_03_user_similarity_recommender.py
"""

from __future__ import annotations

import math


def cosine(a_values: list[float], b_values: list[float]) -> float:
    dot = sum(a * b for a, b in zip(a_values, b_values, strict=True))
    norm_a = math.sqrt(sum(a * a for a in a_values))
    norm_b = math.sqrt(sum(b * b for b in b_values))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def recommend_from_neighbor(target: str, user_vectors: dict[str, list[float]], items: list[str]) -> list[str]:
    """Recommend unseen items from most similar user."""
    similarities: list[tuple[str, float]] = []
    for user, vector in user_vectors.items():
        if user == target:
            continue
        similarities.append((user, cosine(user_vectors[target], vector)))

    nearest_user = sorted(similarities, key=lambda pair: pair[1], reverse=True)[0][0]
    target_vector = user_vectors[target]
    neighbor_vector = user_vectors[nearest_user]

    recommendations: list[str] = []
    for idx, item in enumerate(items):
        if target_vector[idx] == 0 and neighbor_vector[idx] > 0:
            recommendations.append(item)
    return recommendations


def main() -> None:
    """Run user-based recommendation for one target user."""
    items = ["item_a", "item_b", "item_c", "item_d"]
    user_vectors = {
        "u1": [5, 4, 0, 0],
        "u2": [4, 5, 1, 0],
        "u3": [0, 1, 5, 4],
    }

    recs = recommend_from_neighbor("u1", user_vectors, items)
    print(f"target=u1 recommendations={recs}")


if __name__ == "__main__":
    main()
