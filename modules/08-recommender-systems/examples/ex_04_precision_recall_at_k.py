"""Evaluate recommendations with Precision@K and Recall@K.

Run:
    python modules/08-recommender-systems/examples/ex_04_precision_recall_at_k.py
"""

from __future__ import annotations


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def main() -> None:
    """Compute metrics on a tiny recommendation scenario."""
    recommended = ["item_c", "item_b", "item_a", "item_d"]
    relevant = {"item_b", "item_d"}

    for k in [1, 2, 3]:
        p_at_k = precision_at_k(recommended, relevant, k)
        r_at_k = recall_at_k(recommended, relevant, k)
        print(f"k={k} precision@k={p_at_k:.4f} recall@k={r_at_k:.4f}")


if __name__ == "__main__":
    main()
