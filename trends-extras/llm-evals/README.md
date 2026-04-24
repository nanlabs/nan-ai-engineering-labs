# LLM Evals — Evaluación Sistemática de LLMs

## 🎯 Objetivo

Implementar evaluaciones rigurosas para LLMs y AI systems: benchmarks, métricas automáticas, human feedback, y frameworks de evaluation.

## 💡 Qué aprenderás

- Evaluation frameworks (LangChain Evals, PromptFoo, OpenAI Evals)
- Métricas automáticas (BLEU, ROUGE, BERTScore, perplexity)
- Human evaluation (RLHF basics, pairwise comparison, Likert scales)
- Benchmark datasets (MMLU, HellaSwag, TruthfulQA, HumanEval)
- A/B testing para prompt variations
- Regression testing en CI/CD
- Cost-quality trade-offs

## 📂 Contenido

### Examples

- **ex_01_automatic_metrics.py**: BLEU, ROUGE, BERTScore para text generation
- **ex_02_prompt_evaluation.py**: Comparar múltiples prompts con métricas
- **ex_03_benchmark_evaluation.py**: Evaluar LLM en benchmarks estándar
- **ex_04_regression_testing.py**: Tests automáticos para detectar regressions

## 🔑 Conceptos Clave

### Evaluation Pyramid

```
┌────────────────────────┐
│   Human Evaluation     │ ← Gold standard, caro, lento
├────────────────────────┤
│   LLM-as-Judge         │ ← GPT-4 evalúa outputs
├────────────────────────┤
│  Reference-based       │ ← BLEU, ROUGE (need ground truth)
├────────────────────────┤
│  Reference-free        │ ← Perplexity, coherence
└────────────────────────┘
```

### Métricas por Tipo de Tarea

**Text Generation:**

- BLEU (n-gram overlap con referencia)
- ROUGE (recall-oriented para summarization)
- BERTScore (semantic similarity con embeddings)
- METEOR (considera synonyms y stemming)

**Question Answering:**

- Exact Match (EM)
- F1 Score (token-level)
- Answer accuracy

**Classification:**

- Accuracy, Precision, Recall, F1
- Confusion matrix

**Code Generation:**

- Pass@k (% tests passed)
- Execution accuracy

## 📊 Frameworks Comparison

| Framework           | Pros                       | Cons                   | Best For           |
| ------------------- | -------------------------- | ---------------------- | ------------------ |
| **LangChain Evals** | Integrado con LangChain    | Documentación dispersa | LangChain apps     |
| **PromptFoo**       | CLI friendly, configs YAML | Menos programático     | Prompt engineering |
| **OpenAI Evals**    | Community benchmarks       | Requiere OpenAI        | Comparing models   |
| **Evidently AI**    | Drift detection            | Más para monitoring    | Production         |

## 💻 LLM-as-Judge Pattern

En lugar de métricas automáticas, usa GPT-4 como evaluador:

```python
evaluation_prompt = f"""
Evalúa la siguiente respuesta en escala 1-5:

Pregunta: {question}
Respuesta: {llm_response}

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
prompt_a = "Responde en español: {question}"

# Prompt B
prompt_b = "Eres un asistente útil. Responde en español de forma concisa: {question}"

# Evaluar ambos
results_a = evaluate(prompt_a, test_set)
results_b = evaluate(prompt_b, test_set)

# Comparar
print(f"Prompt A: {results_a['avg_score']}")
print(f"Prompt B: {results_b['avg_score']}")
print(f"Winner: {'B' if results_b['avg_score'] > results_a['avg_score'] else 'A'}")
```

## 📈 Benchmark Datasets

| Benchmark      | Task            | Evalúa         | Datasets      |
| -------------- | --------------- | -------------- | ------------- |
| **MMLU**       | Multi-choice QA | Knowledge      | 57 subjects   |
| **HellaSwag**  | Completion      | Common sense   | 10k scenarios |
| **TruthfulQA** | QA              | Truthfulness   | 817 questions |
| **HumanEval**  | Code gen        | Coding ability | 164 problems  |
| **GSM8K**      | Math            | Reasoning      | 8.5k problems |

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

## 🧪 Ejercicio Rápido

1. **Setup**: `pip install rouge-score bert-score`
1. **Ground truth**: Crea 10 Q&A pairs manualmente
1. **Generate**: Obtén respuestas de LLM
1. **Evaluate**: Calcula ROUGE y BERTScore
1. **Iterate**: Mejora prompt, re-evalúa

## 📚 Recursos Curados

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

## ✅ Checklist de Aprendizaje

- [ ] Calcular métricas automáticas (BLEU, ROUGE, BERTScore)
- [ ] Implementar LLM-as-Judge evaluation
- [ ] A/B testing de prompts con test set
- [ ] Evaluar en benchmark público (MMLU subset)
- [ ] Setup regression tests en CI/CD
- [ ] Human evaluation con pairwise comparison
- [ ] Cost-quality analysis (GPT-4 vs GPT-3.5 precision/cost)

## 🎯 Impacto Real

- **Prompt Engineering**: Data-driven prompt optimization
- **Model Selection**: Comparar modelos objetivamente (GPT-4 vs Claude vs Llama)
- **Quality Assurance**: Detectar regressions antes de deploy
- **Research**: Cuantificar mejoras en papers/experiments

## 🚀 Próximos Pasos

Combina con:

- **ai-observability** para evaluar en producción (online metrics)
- **agents** para evaluar agentes (task success rate, costs)
- **guardrails** para medir efectividad de safety filters

## Module objective

Pendiente de completar este apartado.

## What you will achieve

Pendiente de completar este apartado.

## Internal structure

Pendiente de completar este apartado.

## Level path (L1-L4)

Pendiente de completar este apartado.

## Recommended plan (by progress, not by weeks)

Pendiente de completar este apartado.

## Module completion criteria

Pendiente de completar este apartado.
