# Coverage Gap Matrix (Current Snapshot)

## Objective

Provide an up-to-date coverage map for core modules and trends-extras, and define the remaining priorities after structural completion.

## Current Evidence

Source of truth:

- `scripts/validate_learning_labs.py --strict-core`
- `scripts/audit_english_conversion.py`

Latest snapshot:

- Core modules scanned: 12
- Extra units scanned: 6
- Missing required paths: 0
- Heading gaps: 0
- Placeholder markers in trends-extras README files: 0
- Language distribution: 266 English, 0 Spanish, 8 Mixed (274 total files)

## Topic to Module Status

| Topic | Module Path | Status | Priority | Next Action |
| --- | --- | --- | --- | --- |
| Programming & Math for ML | `modules/01-programming-math-for-ml` | Complete | P2 | Optional pedagogical refinements |
| Data Collection, Cleaning & Visualization | `modules/02-data-collection-cleaning-visualization` | Complete | P2 | Optional dataset expansion |
| Machine Learning Fundamentals | `modules/03-machine-learning-fundamentals` | Complete | P2 | Optional metric/report depth |
| Deep Learning Basics | `modules/04-deep-learning-basics` | Complete | P2 | Optional additional diagnostics |
| NLP & Large Language Models | `modules/05-nlp-large-language-models` | Complete | P1 | Resolve remaining mixed-language example files |
| Computer Vision | `modules/06-computer-vision` | Complete | P2 | Optional benchmark extensions |
| Time Series & Anomaly Detection | `modules/07-time-series-anomaly-detection` | Complete | P1 | Resolve remaining mixed-language example file |
| Recommender Systems | `modules/08-recommender-systems` | Complete | P1 | Resolve remaining mixed-language example file |
| Generative AI & Prompt Engineering | `modules/09-generative-ai-prompt-engineering` | Complete | P1 | Resolve remaining mixed-language example file |
| Ethics, Bias & Explainability | `modules/10-ethics-bias-explainability` | Complete | P2 | Optional practice depth improvements |
| Data Privacy & Security | `modules/11-data-privacy-security` | Complete | P1 | Resolve remaining mixed-language example files |
| MLOps & AI in Production | `modules/12-mlops-ai-in-production` | Complete | P2 | Optional production runbook expansion |

## Trends-Extras Status

| Unit | Status | Priority | Next Action |
| --- | --- | --- | --- |
| `trends-extras/agents` | Complete | P2 | Optional additional example density |
| `trends-extras/ai-observability` | Complete | P2 | Optional monitoring case studies |
| `trends-extras/guardrails` | Complete | P2 | Optional red-team scenario expansion |
| `trends-extras/llm-evals` | Complete | P2 | Optional benchmark automation depth |
| `trends-extras/multimodal` | Complete | P2 | Optional model comparison matrix |
| `trends-extras/synthetic-data` | Complete | P2 | Optional utility/privacy benchmark pack |

## Remaining Gaps

### P1 (recommended next)

1. Convert the remaining 8 mixed-language files to fully English content.
2. Standardize wording and naming in mixed legacy filenames where needed.

### P2 (optional quality expansion)

1. Add deeper advanced examples in selected modules.
2. Add stronger expected-output rubrics for all practice paths.
3. Add CI-level checks for language purity if strict English-only is required.

## Exit Criteria

This matrix can be considered fully closed when:

1. Mixed files count reaches zero.
2. Optional CI checks are in place for ongoing enforcement.
