# Examples — Recommender Systems

## Example 1 — Popularity baseline

Build a simple recommender with a global top-N list.

## Example 2 — Item-based collaborative filtering

Compute item similarity and recommend neighbors.

## Example 3 — User-based collaborative filtering

Find similar users and suggest items.

## Example 4 — Evaluation at K

Compute Precision@K and recall@K on a test set.

## Rules

- Each example must include the input, expected output, and conclusion.
- Record findings in `notes/`.

## Available examples

### Executable scripts (phase-2 continuation)

1. `ex_01_popularity_baseline_recommender.py`

   - Builds a top-N baseline using item popularity.
   - Expected output: popularity dictionary and top-3 recommendations.

1. `ex_02_item_similarity_recommender.py`

   - Recommends neighbors from item-to-item cosine similarity.
   - Expected output: ranked neighbor list for a target item.

1. `ex_03_user_similarity_recommender.py`

   - Uses nearest-user logic for user-based recommendations.
   - Expected output: recommendation list for target user.

1. `ex_04_precision_recall_at_k.py`

   - Evaluates recommendation quality at different cutoff values.
   - Expected output: precision@k and recall@k for k in 1..3.

1. `ex_05_cold_start_fallbacks.py`

   - Shows fallback strategies for users with no history.
   - Expected output: popularity-based and tag-based fallback lists.

1. `ex_06_reproducible_recommender_pipeline.py`

   - Runs a deterministic mini pipeline with fixed seed.
   - Expected output: `same_result=True` across repeated runs.

## How to use these examples

```bash
python modules/08-recommender-systems/examples/ex_01_popularity_baseline_recommender.py
python modules/08-recommender-systems/examples/ex_02_item_similarity_recommender.py
python modules/08-recommender-systems/examples/ex_03_user_similarity_recommender.py
python modules/08-recommender-systems/examples/ex_04_precision_recall_at_k.py
python modules/08-recommender-systems/examples/ex_05_cold_start_fallbacks.py
python modules/08-recommender-systems/examples/ex_06_reproducible_recommender_pipeline.py
```

Recommended order: run from baseline (`01`) to evaluation (`04`) and then fallback/reproducibility (`05-06`).

## Next steps

1. Add ranking-focused metrics such as NDCG for deeper evaluation.
1. Add a lightweight hybrid recommender demo (content + collaborative).
1. Document practical trade-offs in `notes/README.md`.
