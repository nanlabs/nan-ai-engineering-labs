"""Cold-start strategy combining popularity and simple content tags.

Run:
    python modules/08-recommender-systems/examples/ex_05_cold_start_fallbacks.py
"""

from __future__ import annotations


def popularity_fallback(scores: dict[str, float], top_n: int = 3) -> list[str]:
    """Recommend most popular items."""
    return [item for item, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_n]]


def tag_based_fallback(user_tags: set[str], item_tags: dict[str, set[str]]) -> list[str]:
    """Recommend items with highest tag overlap."""
    ranking: list[tuple[str, int]] = []
    for item, tags in item_tags.items():
        ranking.append((item, len(user_tags & tags)))
    return [item for item, _ in sorted(ranking, key=lambda pair: pair[1], reverse=True)]


def main() -> None:
    """Show cold-start recommendation options."""
    popularity = {"item_a": 4.8, "item_b": 4.6, "item_c": 4.7, "item_d": 4.1}
    item_tags = {
        "item_a": {"ml", "python"},
        "item_b": {"llm", "nlp"},
        "item_c": {"ml", "timeseries"},
        "item_d": {"cv", "vision"},
    }

    new_user_tags = {"ml", "python", "timeseries"}

    print(f"popularity_fallback={popularity_fallback(popularity, top_n=3)}")
    print(f"tag_fallback={tag_based_fallback(new_user_tags, item_tags)[:3]}")


if __name__ == "__main__":
    main()
