# Examples — NLP & Large Language Models

## Example 1 — Cleaning and tokenization

Preprocess text for classification tasks.

## Example 2 — TF-IDF plus classifier

Train a text-classification baseline.

## Example 3 — Embeddings and similarity

Compare sentences by semantic proximity.

## Example 4 — Basic LLM prompting

Design a structured prompt and evaluate the output.

## Rules

- Define the objective and success criteria for each example.
- Record errors and improvements in `notes/`.

## Available examples

### Executable scripts (phase-2 pilot)

1. `ex_01_text_preprocessing_baseline.py`

   - Minimal normalization and tokenization pipeline.
   - Expected output: cleaned tokens per sentence.

1. `ex_02_embeddings_similarity_baseline.py`

   - Semantic similarity with TF-IDF and cosine similarity.
   - Expected output: similarity A-B higher than similarity A-C.

1. `ex_03_prompt_variants_quality.py`

   - Comparison of weak/medium/strong prompts with a simple rubric.
   - Expected output: prompt-quality score increases with prompt strength.

1. `ex_04_chunking_strategies_rag.py`

   - Comparison of fixed-size versus sentence-based chunking for retrieval.
   - Expected output: score differences and a different top retrieved chunk.

1. `ex_05_rag_minimal_eval.py`

   - Minimal exact-match evaluation over a simplified RAG flow.
   - Expected output: run-level `exact_match_avg`.

1. `ex_06_llm_workflow_with_guardrails.py`

   - Workflow with input and output guardrails for sensitive prompts.
   - Expected output: one allowed case and one blocked case.

## How to use these examples

```bash
python modules/05-nlp-large-language-models/examples/ex_01_text_preprocessing_baseline.py
python modules/05-nlp-large-language-models/examples/ex_02_embeddings_similarity_baseline.py
python modules/05-nlp-large-language-models/examples/ex_03_prompt_variants_quality.py
python modules/05-nlp-large-language-models/examples/ex_04_chunking_strategies_rag.py
python modules/05-nlp-large-language-models/examples/ex_05_rag_minimal_eval.py
python modules/05-nlp-large-language-models/examples/ex_06_llm_workflow_with_guardrails.py
```

Recommendation: change the input sentences and compare how the similarity shifts.

## Next steps

1. Add an L4 example for output evaluation with an expanded rubric.
1. Integrate these examples with `guardrails` and `llm-evals` practices.
1. Document common tokenization and vocabulary issues in `notes/README.md`.
