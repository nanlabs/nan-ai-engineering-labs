# Executable Example Backlog (Phase-2 Pilot)

## Pilot objective

Validate real effort and instructional clarity with two high-impact modules:

1. `modules/03-machine-learning-fundamentals`
1. `modules/05-nlp-large-language-models`

Per-module target in the pilot:

- 6 executable `.py` examples.
- An example README with expected output.
- 1 suggested variation plus 1 common-errors block per example.

## Module 03 - Machine Learning Fundamentals

### L1 (guided baseline)

1. `examples/ex_01_train_test_split_baseline.py`
1. `examples/ex_02_linear_vs_logistic_baseline.py`

### L2 (controlled variations)

1. `examples/ex_03_regularization_tradeoffs.py`
1. `examples/ex_04_tree_depth_vs_overfitting.py`

### L3 (realistic case plus metrics)

1. `examples/ex_05_model_comparison_metrics.py`

### L4 (maintainability extension)

1. `examples/ex_06_reproducible_ml_pipeline.py`

## Module 05 - NLP & Large Language Models

### L1 (guided baseline)

1. `examples/ex_01_text_preprocessing_baseline.py`
1. `examples/ex_02_embeddings_similarity_baseline.py`

### L2 (controlled variations)

1. `examples/ex_03_prompt_variants_quality.py`
1. `examples/ex_04_chunking_strategies_rag.py`

### L3 (realistic case plus metrics)

1. `examples/ex_05_rag_minimal_eval.py`

### L4 (maintainability extension)

1. `examples/ex_06_llm_workflow_with_guardrails.py`

## Pilot definition of done

1. Each example runs with a single command.
1. Each example documents a minimal expected output.
1. Each example includes a controlled variation.
1. Each example includes common errors and a quick debug guide.
1. Pre-commit is green for new files.
1. Contract validation is green.
