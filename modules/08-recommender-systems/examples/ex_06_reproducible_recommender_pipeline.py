"""Reproducible mini recommender pipeline.

Run:
    python modules/08-recommender-systems/examples/ex_06_reproducible_recommender_pipeline.py
"""

from __future__ import annotations

import random


def generate_interactions(seed: int) -> list[tuple[str, str, float]]:
    """Generate deterministic user-item interactions."""
    random.seed(seed)
    users = ["u1", "u2", "u3"]
    items = ["item_a", "item_b", "item_c", "item_d"]

    interactions: list[tuple[str, str, float]] = []
    for user in users:
        for item in items[:3]:
            rating = round(3.0 + random.random() * 2.0, 2)
            interactions.append((user, item, rating))
    return interactions


def popularity_scores(interactions: list[tuple[str, str, float]]) -> dict[str, float]:
    """Compute average score per item."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _, item, score in interactions:
        totals[item] = totals.get(item, 0.0) + score
        counts[item] = counts.get(item, 0) + 1
    return {item: totals[item] / counts[item] for item in totals}


def main() -> None:
    """Run deterministic recommendation pipeline twice."""
    interactions_a = generate_interactions(seed=9)
    scores_a = popularity_scores(interactions_a)

    interactions_b = generate_interactions(seed=9)
    scores_b = popularity_scores(interactions_b)

    print(f"scores_a={scores_a}")
    print(f"scores_b={scores_b}")
    print(f"same_result={scores_a == scores_b}")


if __name__ == "__main__":
    main()
