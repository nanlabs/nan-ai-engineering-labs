# AI Observability — Monitoring of LLM Applications

## 🎯 Objective

Implement complete observability for applications with LLMs: tracing, prompt/completion logging, latency monitoring, cost tracking, and error analysis.

## 💡 What will you learn

- Distributed tracing for LLM calls (LangSmith, Weights & Biases)
- Logging structure (prompts, completions, metadata)
- Latency and performance monitoring
- Cost tracking (tokens, API calls, $ per request)
- error analysis and debugging
- A/B testing in production
- User feedback collection (thumbs up/down, corrections)

## 📂 Content

### Examples

- **ex_01_basic_logging.py**: Logging prompts/completions with metadata
- **ex_02_langsmith_tracing.py**: Distributed tracing with LangSmith
- **ex_03_cost_tracking.py**: Calculate costs per request/user/day
- **ex_04_metrics_dashboard.py**: Dashboard with Prometheus + Grafana

## 🔑 Concepts Clave

### Observability Pillars for LLMs

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
- tokens per second
- Time to first token (TTFT)
- Requests per minute

**Cost:**

- tokens used (prompt + completion)
- Cost per request ($ per 1K tokens)
- Daily/monthly spend by user/endpoint

**Quality:**

- User feedback rate (👍/👎)
- error rate (API failures, timeouts)
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

# En LangSmith dashboard you will see:
# rag_pipeline (2.3s)
#   ├─ retrieval (0.8s)
#   └─ generation (1.5s)
```

## 📊 Cost Tracking Pattern

```python
import tiktoken

def calculate_cost(prompt: str, completion: str, model: str = "gpt-4"):
    # Pricing (example)
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

| Tools                | Type                | Pros                       | Cons              | Price |
| ---------------------| ------------------- | -------------------------- | ----------------- | ----- |
| **LangSmith**        | Tracing             | LangChain native, powerful | Tied to LangChain | $$    |
| **Weights & Biases** | Experiment tracking | Great for ML               | Learning curve    | $$$   |
| **Helicone**         | Proxy logger        | Easy setup                 | Limited features  | $     |
| **Custom Logging**   | DIY                 | Full control               | More work         | Free  |
| **OpenTelemetry**    | Standard            | Vendor agnostic            | Setup complex     | Free  |

## 🧪 Quick Exercise

1. **Setup**: `pip install langsmith opentelemetry`
2. **Log LLM calls**: Implement wrapper that logs each call
3. **Track costs**: Calculate total cost of 100 requests
4. **Visualize**: Create latency vs tokens graph
5. **Alert**: Alert Setup if cost/day > $100

## 📚 Health Resources

**Platforms:**

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

## ✅ Learning Checklist

- [ ] Log structured prompts/completions
- [ ] Implement distributed tracing (LangSmith o OpenTelemetry)
- [ ] Track tokens and costs per request
- [ ] Monitor latency metrics (p50/p95/p99)
- [ ] error tracking and alerting
- [ ] User feedback collection (thumbs up/down)
- [ ] A/B testing with feature flags
- [ ] Cost dashboards per user/endpoint/day

## 🎯 Real impact

- **Cost Control**: Detect "runaway costs" before it's too late
- **Performance**: Identify bottlenecks (retrieval lento, LLM slow)
- **Quality**: Correlate user feedback with prompt versions
- **Debugging**: Reproduce issues with complete logs
- **Compliance**: Audit of prompts/outputs for regulations

## 🚨 Common Pitfalls

**Privacy**: logging PII without consent

- **Solution**: Redact PII before logging

**Volume**: Massive logs impact performance

- **Solution**: Sampling, async logging

**Cost**: Tracing tools can be expensive

- **Solution**: Start with open source, upgrade if necessary

## 🚀 Next Steps

Combine with:

- **llm-evals** to correlate metrics with quality
- **agents** to trace autonomous agent steps
- **guardrails** to log blocked attempts

## Module objective

Learn to instrument, monitor, and troubleshoot LLM-powered systems in production using actionable metrics, traces, and cost signals.

## What you will achieve

- Configure structured logs and useful operational metadata.
- Add tracing for multi-step LLM pipelines.
- Track token usage and cost budgets.
- Define alerting thresholds for reliability and quality risks.

## Internal structure

- `README.md`: observability principles, pitfalls, and deployment guidance.
- `examples/`: logging, tracing, cost tracking, and dashboard code.
- `practices/`: hands-on monitoring design and incident response drills.

## Level path (L1-L4)

- L1: Enable baseline logging and request metadata.
- L2: Add traces and correlation IDs across components.
- L3: Define metrics and alerts for latency, errors, and drift.
- L4: Build an observability runbook for production incidents.

## Recommended plan (by progress, not by weeks)

Start by instrumenting one critical user flow, then expand coverage to traces and cost controls. After baseline visibility is stable, add alerts and on-call playbooks.

## Module completion criteria

- You can diagnose failures from logs and traces.
- You can report latency, error rate, and cost for a full workflow.
- You can justify alert thresholds and escalation rules.
