# AI Observability — Monitoring de LLM Applications

## 🎯 Objetivo

Implementar observability completa para aplicaciones con LLMs: tracing, logging de prompts/completions, latency monitoring, cost tracking, y error analysis.

## 💡 Qué aprenderás

- Distributed tracing para LLM calls (LangSmith, Weights & Biases)
- Logging estructurado (prompts, completions, metadata)
- Latency y performance monitoring
- Cost tracking (tokens, API calls, $ per request)
- Error analysis y debugging
- A/B testing in production
- User feedback collection (thumbs up/down, corrections)

## 📂 Contenido

### Examples

- **ex_01_basic_logging.py**: Logging de prompts/completions con metadata
- **ex_02_langsmith_tracing.py**: Tracing distribuido con LangSmith
- **ex_03_cost_tracking.py**: Calcular costos por request/user/día
- **ex_04_metrics_dashboard.py**: Dashboard con Prometheus + Grafana

## 🔑 Conceptos Clave

### Observability Pillars para LLMs

```
┌──────────────────────────────────────┐
│            OBSERVABILITY             │
├──────────────────────────────────────┤
│  Logs        │  Traces    │  Metrics │
├──────────────┼────────────┼──────────┤
│ - Prompts    │ - Spans    │ - Latency│
│ - Outputs    │ - Parent   │ - Tokens │
│ - Errors     │ - Children │ - Costs  │
│ - Metadata   │ - Duration │ - Errors │
└──────────────┴────────────┴──────────┘
```

### Key Metrics to Track

**Performance:**

- Latency (p50, p95, p99)
- Tokens per second
- Time to first token (TTFT)
- Requests per minute

**Cost:**

- Tokens used (prompt + completion)
- Cost per request ($ per 1K tokens)
- Daily/monthly spend by user/endpoint

**Quality:**

- User feedback rate (👍/👎)
- Error rate (API failures, timeouts)
- Hallucination detection score
- Response length distribution

## 🔍 Distributed Tracing Example

```python
from langsmith import traceable

@traceable(name="retrieval")
def retrieve_docs(query: str):
    # Vector search
    docs = vector_db.search(query, top_k=5)
    return docs

@traceable(name="generation")
def generate_response(query: str, context: str):
    prompt = f"Context: {context}\n\nQuestion: {query}"
    response = llm(prompt)
    return response

@traceable(name="rag_pipeline")
def rag_pipeline(user_query: str):
    docs = retrieve_docs(user_query)
    context = "\n".join([d.text for d in docs])
    response = generate_response(user_query, context)
    return response

# En LangSmith dashboard verás:
# rag_pipeline (2.3s)
#   ├─ retrieval (0.8s)
#   └─ generation (1.5s)
```

## 📊 Cost Tracking Pattern

```python
import tiktoken

def calculate_cost(prompt: str, completion: str, model: str = "gpt-4"):
    # Pricing (ejemplo)
    pricing = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},  # $ per 1K tokens
        "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002}
    }

    # Count tokens
    enc = tiktoken.encoding_for_model(model)
    prompt_tokens = len(enc.encode(prompt))
    completion_tokens = len(enc.encode(completion))

    # Calculate cost
    cost = (
        (prompt_tokens / 1000) * pricing[model]["prompt"] +
        (completion_tokens / 1000) * pricing[model]["completion"]
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost
    }
```

## 📈 Logging Best Practices

**Structured Logging:**

```python
import logging
import json
from datetime import datetime

logger = logging.getLogger("llm_app")

def log_llm_call(
    user_id: str,
    prompt: str,
    completion: str,
    model: str,
    latency_ms: float,
    cost: float,
    metadata: dict = None
):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "model": model,
        "prompt": prompt[:200],  # Truncate for privacy
        "completion": completion[:200],
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "metadata": metadata or {}
    }

    logger.info(json.dumps(log_entry))
```

## 🛠️ Tools Comparison

| Tool                 | Type                | Pros                       | Cons              | Price |
| -------------------- | ------------------- | -------------------------- | ----------------- | ----- |
| **LangSmith**        | Tracing             | LangChain native, powerful | Tied to LangChain | $$    |
| **Weights & Biases** | Experiment tracking | Great for ML               | Learning curve    | $$$   |
| **Helicone**         | Proxy logger        | Easy setup                 | Limited features  | $     |
| **Custom Logging**   | DIY                 | Full control               | More work         | Free  |
| **OpenTelemetry**    | Standard            | Vendor agnostic            | Setup complex     | Free  |

## 🧪 Ejercicio Rápido

1. **Setup**: `pip install langsmith opentelemetry`
1. **Log LLM calls**: Implementa wrapper que loggea cada llamada
1. **Track costs**: Calcula gasto total de 100 requests
1. **Visualize**: Crea gráfico de latency vs tokens
1. **Alert**: Setup alerta si cost/day > $100

## 📚 Recursos Curados

**Plataformas:**

- [LangSmith](https://www.langchain.com/langsmith)
- [Weights & Biases](https://wandb.ai/)
- [Helicone](https://www.helicone.ai/)
- [Arize AI](https://arize.com/)

**Open Source:**

- [OpenTelemetry](https://opentelemetry.io/)
- [OpenLLMetry](https://github.com/traceloop/openllmetry)
- [Langfuse](https://github.com/langfuse/langfuse)

**Dashboards:**

- [Grafana](https://grafana.com/)
- [Datadog LLM Observability](https://www.datadoghq.com/product/llm-observability/)

**Guides:**

- [LLM Observability Best Practices](https://www.langchain.com/blog/observability)

## ✅ Checklist de Aprendizaje

- [ ] Log structured prompts/completions
- [ ] Implement distributed tracing (LangSmith o OpenTelemetry)
- [ ] Track tokens y costs por request
- [ ] Monitor latency metrics (p50/p95/p99)
- [ ] Error tracking y alerting
- [ ] User feedback collection (thumbs up/down)
- [ ] A/B testing con feature flags
- [ ] Cost dashboards por usuario/endpoint/día

## 🎯 Impacto Real

- **Cost Control**: Detectar "runaway costs" antes de que sea tarde
- **Performance**: Identificar bottlenecks (retrieval lento, LLM slow)
- **Quality**: Correlacionar user feedback con prompt versions
- **Debugging**: Reproducir issues con logs completos
- **Compliance**: Auditoría de prompts/outputs para regulaciones

## 🚨 Common Pitfalls

**Privacy**: Loggear PII sin consent

- **Solución**: Redactar PII antes de logging

**Volume**: Logs masivos impactan performance

- **Solución**: Sampling, async logging

**Cost**: Tracing tools pueden ser caros

- **Solución**: Start con open source, upgrade si necesario

## 🚀 Próximos Pasos

Combina con:

- **llm-evals** para correlacionar metrics con quality
- **agents** para tracear pasos de agentes autónomos
- **guardrails** para loggear intentos bloqueados

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
