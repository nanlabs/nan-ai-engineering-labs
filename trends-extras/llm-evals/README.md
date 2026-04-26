# LLM Evals — Systematic Evaluation of LLMs

## 🎯 Objective

Implement rigorous evaluations for LLMs and AI systems: benchmarks, automatic metrics, human feedback, and evaluation frameworks.

## 💡 What will you learn

- Evaluation frameworks (LangChain Evals, PromptFoo, OpenAI Evals)
- Automatic metrics (BLEU, ROUGE, BERTScore, perplexity)
- Human evaluation (RLHF basics, pairwise comparison, Likert scales)
- Benchmark datasets (MMLU, HellaSwag, TruthfulQA, HumanEval)
- A/B testing for prompt variations
- Regression testing in CI/CD
- Cost-quality trade-offs

## 📂 Content

### Examples

- **ex_01_automatic_metrics.py**: BLEU, ROUGE, BERTScore for text generation
- **ex_02_prompt_evaluation.py**: Compare multiple prompts with Metrics
- **ex_03_benchmark_evaluation.py**: Evaluate LLM on standard benchmarks
- **ex_04_regression_testing.py**: Automatic tests to detect regressions

## 🔑 Concepts Clave

### Evaluation Pyramid

```
┌────────────────────────┐
│   Human Evaluation     │ ← Gold standard, caro, lento
├────────────────────────┤
│   LLM-as-Judge         │ ← GPT-4 evaluate outputs
├────────────────────────┤
│  Reference-based       │ ← BLEU, ROUGE (need ground truth)
├────────────────────────┤
│  Reference-free        │ ← Perplexity, coherence
└────────────────────────┘
```

### Metrics by Task Type

**Text Generation:**

- BLEU (n-gram overlap with reference)
- ROUGE (recall-oriented for summary)
- BERTScore (semantic similarity with embeddings)
- METEOR (considers synonyms and stemming)

**Question Answering:**

- Exact Match (EM)
- f1 Score (token-level)
- Answer accuracy

**Classification:**

- accuracy, Precision, recall, f1
- Confusion matrix

**Code Generation:**

- Pass@k (% tests passed)
- Execution accuracy

## 📊 Frameworks Comparison

| Framework | Pros | Cons | Best For |
| ------------------- | -------------------------- | ---------------------- | ------------------ |
| **LangChain Evals** | Integrated with LangChain | Scattered documentation | LangChain apps |
| **PromptFoo** | CLI friendly, YAML configs | Less programmatic | Prompt engineering |
| **OpenAI Evals** | Community benchmarks | Requires OpenAI | Comparing models |
| **Evidently AI** | Drift detection | More for monitoring | Production |

## 💻 LLM-as-Judge Pattern

Instead of automatic Metrics, use GPT-4 as the evaluator:

```python
evaluation_prompt = f"""
Evaluate la next response en escala 1-5:

Pregunta: {question}
Answer: {llm_response}

Criterios:
- Factualidad (1-5)
- Relevancia (1-5)
- Completitud (1-5)

Formato JSON:
{{"factuality": X, "relevance": Y, "completeness": Z, "reasoning": "..."}}
"""

judge_response = gpt4(evaluation_prompt)
scores = json.loads(judge_response)
```

## 🧪 A/B Testing Prompts

```python
# Prompt A
prompt_a = "Respond, Response, Responds, Responded, Responder en Spanish: {question}"

# Prompt B
prompt_b = "Eres un asistente useful. Respond, Response, Responds, Responded, Responder en Spanish de forma concisa: {question}"

# Evaluate ambos
results_a = evaluate(prompt_a, test_set)
results_b = evaluate(prompt_b, test_set)

# Compare
print(f"Prompt A: {results_a['avg_score']}")
print(f"Prompt B: {results_b['avg_score']}")
print(f"Winner: {'B' if results_b['avg_score'] > results_a['avg_score'] else 'A'}")
```

## 📈 Benchmark Datasets

| Benchmark | Task | Evaluate | Datasets |
| -------------- | --------------- | -------------- | ------------- |
| **MMLU** | Multi-choice QA | Knowledge | 57 subjects |
| **HellaSwag** | Completion | Common sense | 10k scenarios |
| **TruthfulQA** | QA | Truthfulness | 817 questions |
| **HumanEval** | Code gen | Coding ability | 164 problems |
| **GSM8K** | Math | Reasoning | 8.5k problems |

## 🔬 Regression Testing

```python
# tests/test_llm_responses.py
import pytest

def test_greeting():
    response = llm("Hello")
    assert "hi" in response.lower() or "hello" in response.lower()

def test_no_pii_leakage():
    response = llm("What's my email?")
    assert not contains_email(response)

def test_response_length():
    response = llm("Explain AI in one sentence")
    assert len(response.split()) <= 30  # Brief response

def test_factuality():
    response = llm("Capital of France?")
    assert "paris" in response.lower()
```

## 🧪 Quick Exercise

1. **Setup**: `pip install rouge-score bert-score`
1. **Ground truth**: Create 10 Q&A pairs manualmente
1. **Generate**: Get responses from LLM
1. **Evaluate**: Calculate ROUGE and BERTScore
1. **Iterate**: Improve prompt, re-evaluate

## 📚 Resources Curados

**Frameworks:**

- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
- [PromptFoo](https://github.com/promptfoo/promptfoo)
- [OpenAI Evals](https://github.com/openai/evals)
- [Ragas (RAG evaluation)](https://github.com/explodinggradients/ragas)

**Metrics Libraries:**

- [ROUGE](https://github.com/google-research/google-research/tree/master/rouge)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)

**Benchmarks:**

- [HELM (Stanford)](https://crfm.stanford.edu/helm/)
- [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

**Papers:**

- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)

## ✅ Learning Checklist

- [ ] Automatic Calculate Metrics (BLEU, ROUGE, BERTScore)
- [ ] Implement LLM-as-Judge evaluation
- [ ] A/B testing of prompts with test set
- [ ] Evaluate in public benchmark (MMLU subset)
- [ ] Setup regression tests in CI/CD
- [ ] Human evaluation with pairwise comparison
- [ ] Cost-quality analysis (GPT-4 vs GPT-3.5 precision/cost)

## 🎯 Impacto Real

- **Prompt Engineering**: Data-driven prompt optimization
- **Model Selection**: Compare Models objetivamente (GPT-4 vs Claude vs Llama)
- **Quality Assurance**: Detect regressions before deploying
- **Research**: Quantify improvements in papers/experiments

## 🚀 Next Steps

Combine with:

- **ai-observability** for assess in production (online metrics)
- **agents** for assess agents (task success rate, costs)
- **guardrails** to measure effectiveness of safety filters

## Module objective

Develop a reproducible evaluation workflow for LLM systems that combines automatic metrics, benchmark tests, and human review.

## What you will achieve

- Compare prompts, models, and configurations with quantitative evidence.
- Build regression tests to prevent quality degradation.
- Interpret metric trade-offs (quality, latency, cost).
- Establish a repeatable evaluation protocol for release decisions.

## Internal structure

- `README.md`: evaluation strategy, metric taxonomy, and decision criteria.
- `examples/`: metric computation, benchmark runs, and regression testing.
- `practices/`: guided experiments and evaluation reports.

## Level path (L1-L4)

- L1: Run baseline automatic metrics on sample outputs.
- L2: Compare prompt variants with controlled experiments.
- L3: Add benchmark suites and human pairwise review.
- L4: Automate regression gates in CI/CD.

## Recommended plan (by progress, not by weeks)

Begin with a small, representative dataset and baseline metrics. Then add prompt/model comparisons, benchmark checks, and finally regression gates in your delivery pipeline.

## Module completion criteria

- You can produce an evaluation report with reproducible methodology.
- You can justify model/prompt choices using metrics and error analysis.
- You can run regression checks before deployment.
