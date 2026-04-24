"""
Basic LLM Logging
=================
Logging estructurado para LLM applications.
Track prompts, completions, metadata, errors.

Requirements:
    pip install python-json-logger
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any

# ============================================================================
# STRUCTURED LOGGING SETUP
# ============================================================================

class LLMLogger:
    """
    Logger estructurado para LLM calls.
    """

    def __init__(self, log_file: str = "llm_logs.jsonl"):
        self.log_file = log_file

        # Setup logger
        self.logger = logging.getLogger("llm_app")
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(console)

    def log_llm_call(
        self,
        user_id: str,
        session_id: str,
        prompt: str,
        completion: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
        metadata: Dict = None
    ):
        """
        Loggea una llamada al LLM.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "prompt": prompt[:200],  # Truncate for privacy
            "completion": completion[:200],
            "model": model,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "metadata": metadata or {}
        }

        self.logger.info(json.dumps(log_entry))

    def log_error(
        self,
        user_id: str,
        error_type: str,
        error_message: str,
        prompt: str = None,
        metadata: Dict = None
    ):
        """
        Loggea un error.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "user_id": user_id,
            "error_type": error_type,
            "error_message": error_message,
            "prompt": prompt[:200] if prompt else None,
            "metadata": metadata or {}
        }

        self.logger.error(json.dumps(log_entry))


# ============================================================================
# MOCK LLM WITH LOGGING
# ============================================================================

def mock_llm_with_logging(prompt: str, model: str = "gpt-3.5-turbo") -> Dict:
    """
    Mock LLM que simula latency y retorna metadata.
    """
    start_time = time.time()

    # Simulate API call
    time.sleep(0.1)  # Simulate latency

    # Mock response
    completion = "This is a mock response for: " + prompt[:30]

    # Calculate metadata
    input_tokens = len(prompt.split()) * 1.3  # Aproximación
    output_tokens = len(completion.split()) * 1.3
    latency_ms = (time.time() - start_time) * 1000

    # Cost (simplified)
    cost_per_1k_tokens = 0.002 if model == "gpt-3.5-turbo" else 0.03
    cost_usd = ((input_tokens + output_tokens) / 1000) * cost_per_1k_tokens

    return {
        "completion": completion,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "latency_ms": latency_ms,
        "cost_usd": cost_usd
    }


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_basic_logging():
    """Logging básico."""
    print("="*70)
    print("DEMO 1: Basic LLM Logging")
    print("="*70 + "\n")

    logger = LLMLogger("demo_logs.jsonl")

    # Simular varias llamadas
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a haiku about AI",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"📞 Call {i}: {prompt[:40]}...")

        # Call LLM
        result = mock_llm_with_logging(prompt)

        # Log it
        logger.log_llm_call(
            user_id="user_123",
            session_id="session_abc",
            prompt=prompt,
            completion=result["completion"],
            model=result["model"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            latency_ms=result["latency_ms"],
            cost_usd=result["cost_usd"],
            metadata={"endpoint": "/chat"}
        )

        print(f"   ✅ Logged: {result['latency_ms']:.0f}ms, ${result['cost_usd']:.4f}\n")


def demo_error_logging():
    """Logging de errores."""
    print("="*70)
    print("DEMO 2: Error Logging")
    print("="*70 + "\n")

    logger = LLMLogger("demo_logs.jsonl")

    # Simular errores
    errors = [
        ("rate_limit", "Rate limit exceeded: 60 requests/min", "Hello"),
        ("timeout", "Request timed out after 30s", "Explain universe"),
        ("invalid_prompt", "Prompt contains prohibited content", "How to hack"),
    ]

    for error_type, error_msg, prompt in errors:
        print(f"❌ Error: {error_type}")
        print(f"   Message: {error_msg}\n")

        logger.log_error(
            user_id="user_456",
            error_type=error_type,
            error_message=error_msg,
            prompt=prompt,
            metadata={"retry_count": 0}
        )


def demo_aggregation_queries():
    """Queries de agregación en logs."""
    print("="*70)
    print("DEMO 3: Log Aggregation Queries")
    print("="*70 + "\n")

    print("💡 Queries útiles con jq o SQL:\n")

    print("1️⃣ Total de requests por usuario:")
    print("   jq -r '.user_id' llm_logs.jsonl | sort | uniq -c\n")

    print("2️⃣ Costo total por día:")
    print("   jq -r '[.timestamp, .cost_usd] | @csv' llm_logs.jsonl | \\\n")
    print("     awk -F, '{sum[$1]+=$2} END {for(d in sum) print d, sum[d]}'\n")

    print("3️⃣ P95 latency:")
    print("   jq -r '.latency_ms' llm_logs.jsonl | sort -n | \\\n")
    print("     awk '{a[NR]=$1} END {print a[int(NR*0.95)]}'\n")

    print("4️⃣ Errores más comunes:")
    print("   jq -r 'select(.level==\"ERROR\") | .error_type' llm_logs.jsonl | \\\n")
    print("     sort | uniq -c | sort -rn\n")

    print("5️⃣ Average tokens por modelo:")
    print("   jq -r 'select(.model) | [.model, .tokens.total] | @csv' llm_logs.jsonl | \\\n")
    print("     awk -F, '{sum[$1]+=$2; cnt[$1]++} END {for(m in sum) print m, sum[m]/cnt[m]}'\n")


def demo_log_analysis_script():
    """Script de análisis de logs."""
    print("="*70)
    print("DEMO 4: Log Analysis Script")
    print("="*70 + "\n")

    print("📊 analyze_logs.py:\n")
    print("""
import json
from collections import defaultdict
from datetime import datetime

def analyze_logs(log_file: str):
    '''
    Analiza logs de LLM.
    '''
    stats = {
        'total_requests': 0,
        'total_cost': 0.0,
        'total_tokens': 0,
        'errors': 0,
        'latencies': [],
        'users': set(),
        'models': defaultdict(int),
    }

    with open(log_file) as f:
        for line in f:
            entry = json.loads(line)

            if entry.get('level') == 'ERROR':
                stats['errors'] += 1
                continue

            stats['total_requests'] += 1
            stats['total_cost'] += entry.get('cost_usd', 0)
            stats['total_tokens'] += entry.get('tokens', {}).get('total', 0)
            stats['latencies'].append(entry.get('latency_ms', 0))
            stats['users'].add(entry.get('user_id'))
            stats['models'][entry.get('model')] += 1

    # Calculate percentiles
    latencies_sorted = sorted(stats['latencies'])
    p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

    # Print report
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Total Cost: ${stats['total_cost']:.2f}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Unique Users: {len(stats['users'])}")
    print(f"Errors: {stats['errors']}")
    print(f"\\nLatency:")
    print(f"  P50: {p50:.0f}ms")
    print(f"  P95: {p95:.0f}ms")
    print(f"  P99: {p99:.0f}ms")
    print(f"\\nModels:")
    for model, count in stats['models'].items():
        print(f"  {model}: {count} requests")

if __name__ == '__main__':
    analyze_logs('llm_logs.jsonl')
""")


def demo_best_practices():
    """Best practices para logging."""
    print("\n" + "="*70)
    print("DEMO 5: Logging Best Practices")
    print("="*70 + "\n")

    print("✅ DO:")
    print("  • Log EVERY LLM call")
    print("  • Include timestamp, user_id, cost, latency")
    print("  • Use structured logging (JSON)")
    print("  • Truncate prompts/completions (privacy)")
    print("  • Log errors with context")
    print("  • Async logging (don't block requests)")
    print("  • Rotate log files (daily/weekly)\n")

    print("❌ DON'T:")
    print("  • Log full PII (emails, passwords, SSNs)")
    print("  • Log API keys")
    print("  • Ignore errors")
    print("  • Use unstructured logs (hard to query)")
    print("  • Log synchronously (adds latency)\n")

    print("📁 LOG RETENTION:")
    print("  • Hot: Last 7 days (fast access)")
    print("  • Warm: 8-30 days (slower access)")
    print("  • Cold: 30+ days (archive, compliance)\n")

    print("🔍 QUERYABLE STORAGE:")
    print("  • Files: jq, grep (simple, cheap)")
    print("  • Elasticsearch: Full-text search")
    print("  • BigQuery/Snowflake: SQL analytics")
    print("  • ClickHouse: Time-series analytics")


if __name__ == "__main__":
    print("\n🎯 BASIC LLM LOGGING")
    print("📝 Track every LLM call for observability\n")

    demo_basic_logging()
    demo_error_logging()
    demo_aggregation_queries()
    demo_log_analysis_script()
    demo_best_practices()

    print("\n" + "="*70)
    print("🚀 NEXT STEPS:")
    print("="*70)
    print("  1. Setup structured logging (JSON)")
    print("  2. Log to file + cloud (S3, GCS)")
    print("  3. Create analysis scripts")
    print("  4. Set up alerts (cost > $100/day)")
    print("  5. Build dashboards (Grafana, Datadog)")
    print("  6. Implement log rotation")
    print("  7. Ensure compliance (GDPR, HIPAA)")
