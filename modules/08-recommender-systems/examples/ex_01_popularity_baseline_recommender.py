"""Popularity-based top-N recommender baseline.

Run:
    python modules/08-recommender-systems/examples/ex_01_popularity_baseline_recommender.py
"""

from __future__ import annotations


def build_popularity(ratings: list[tuple[str, str, float]]) -> dict[str, float]:
    """Aggregate item scores by average rating."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _, item, rating in ratings:
        totals[item] = totals.get(item, 0.0) + rating
        counts[item] = counts.get(item, 0) + 1
    return {item: totals[item] / counts[item] for item in totals}


def recommend_top_n(popularity: dict[str, float], n: int = 3) -> list[tuple[str, float]]:
    """Return top-N items by popularity score."""
    return sorted(popularity.items(), key=lambda pair: pair[1], reverse=True)[:n]


def main() -> None:
    """Run popularity baseline on a tiny rating set."""
    ratings = [
        ("u1", "item_a", 4.0),
        ("u2", "item_a", 5.0),
        ("u1", "item_b", 3.0),
        ("u3", "item_c", 5.0),
        ("u2", "item_c", 4.5),
        ("u3", "item_d", 2.5),
    ]

    popularity = build_popularity(ratings)
    top_items = recommend_top_n(popularity, n=3)

    print(f"popularity_scores={popularity}")
    print(f"top_3={top_items}")


if __name__ == "__main__":
    main()
