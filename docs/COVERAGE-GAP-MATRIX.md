# Coverage Gap Matrix (Phase 1)

## Objective

Establish an objective topic-subtopic-module coverage matrix and prioritize P0/P1/P2 gaps to guide plan execution.

## Method and evidence

Primary curriculum sources:

- `docs/init-path/content/*.md` (12 core topics).
- `docs/LEARNING-PATH.md` (official study sequence).

Technical snapshot (2026-04-23):

- Core modules with executable Python examples: 0 across 12/12 modules.
- Core modules with Markdown examples: 3 per module.
- Placeholders in core (`Pendiente de completar este apartado.`): 121.
- Placeholders in extras: 36.
- Python examples per extra:
  - `agents`: 3
  - `ai-observability`: 4
  - `guardrails`: 3
  - `llm-evals`: 4
  - `multimodal`: 4
  - `synthetic-data`: 3

## Topic -> Module -> Status matrix

| Topic (init-path)                         | Target module                                       | Current status | Priority | Main action                                                             |
| ----------------------------------------- | --------------------------------------------------- | -------------- | -------- | ----------------------------------------------------------------------- |
| Programming & Math for ML                 | `modules/01-programming-math-for-ml`                | Partial        | P0       | Migrate key examples to executable `.py` files with expected output     |
| Data Collection, Cleaning & Visualization | `modules/02-data-collection-cleaning-visualization` | Partial        | P0       | Add executable cleaning examples plus reproducible EDA                  |
| Machine Learning Fundamentals             | `modules/03-machine-learning-fundamentals`          | Partial        | P0       | Add regression/classification baselines with metrics and comparison     |
| Deep Learning Basics                      | `modules/04-deep-learning-basics`                   | Partial        | P0       | Add minimal reproducible training (MLP/CNN) with overfit checks         |
| NLP & Large Language Models               | `modules/05-nlp-large-language-models`              | Partial        | P0       | Add executable NLP/LLM pipeline (prompting plus minimal eval)           |
| Computer Vision                           | `modules/06-computer-vision`                        | Partial        | P0       | Add executable CV examples with inference and basic validation          |
| Time Series & Anomaly Detection           | `modules/07-time-series-anomaly-detection`          | Partial        | P0       | Add forecast plus anomaly detection with temporal metrics               |
| Recommender Systems                       | `modules/08-recommender-systems`                    | Partial        | P0       | Add collaborative-filtering baselines with reproducible ranking metrics |
| Generative AI & Prompt Engineering        | `modules/09-generative-ai-prompt-engineering`       | Partial        | P0       | Add prompting examples and quality evaluation by scenario               |
| Ethics, Bias & Explainability             | `modules/10-ethics-bias-explainability`             | Partial        | P1       | Add guided fairness and SHAP/LIME-style examples                        |
| Data Privacy & Security                   | `modules/11-data-privacy-security`                  | Partial        | P1       | Add redaction, anonymization, and minimal control examples              |
| MLOps & AI in Production                  | `modules/12-mlops-ai-in-production`                 | Partial        | P0       | Add executable serving, monitoring, and rollback examples               |

## Prioritized cross-cutting gaps

### P0 (blocking executable learning)

1. Core modules without executable `.py` scripts in `examples/` (12/12 modules).
1. High volume of placeholders in core modules (121).
1. Missing explicit expected output for most core examples.

### P1 (high impact, not immediately blocking)

1. Extras below the density target (4-6): `agents`, `guardrails`, and `synthetic-data` have 3.
1. Residual placeholders in extras (36).
1. Narrative misalignment: documents claiming 100% completion versus evidence of partial executable coverage.

### P2 (advanced coverage expansion)

1. Applied RL for product use cases.
1. Introductory causal ML.
1. Introductory graph ML.
1. Advanced evals and systematic red-teaming.
1. Inference optimization (quantization/distillation/serving).

## Recommended initial backlog (execution order)

1. Complete the P0 placeholders in modules 03 and 05 (pilot).
1. Publish 6-8 executable examples in module 03.
1. Publish 6-8 executable examples in module 05.
1. Define an expected-output and common-errors rubric for every new example.
1. Scale the pattern across core waves A/B/C.

## Phase-1 exit criteria

1. Matrix published and versioned (this document).
1. P0/P1/P2 gaps agreed upon.
1. Execution pilot defined (M03 and M05).
