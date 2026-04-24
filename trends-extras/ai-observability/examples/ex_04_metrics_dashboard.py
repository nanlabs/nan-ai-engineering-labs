"""
Metrics Dashboard with Prometheus
==================================
Expo metrics para monitoring con Prometheus + Grafana.
Track latency, throughput, errors, costs en production.

Requirements:
    pip install prometheus-client
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import time
import random

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Counters: Monotonically increasing (total requests, errors)
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'endpoint', 'status']
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens processed',
    ['model', 'type']  # type: input/output
)

llm_cost_total = Counter(
    'llm_cost_usd_total',
    'Total cost in USD',
    ['model']
)

# Histograms: Distribution of values (latency, response length)
llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM call latency in seconds',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Buckets for percentiles
)

llm_response_length = Histogram(
    'llm_response_length_tokens',
    'LLM response length in tokens',
    ['model'],
    buckets=[10, 50, 100, 500, 1000, 5000]
)

# Gauges: Current value (active requests, cache hit rate)
llm_active_requests = Gauge(
    'llm_active_requests',
    'Number of active LLM requests',
    ['model']
)

llm_cache_hit_rate = Gauge(
    'llm_cache_hit_rate',
    'Cache hit rate (0-1)',
    ['endpoint']
)


# ============================================================================
# INSTRUMENTED LLM WRAPPER
# ============================================================================

class InstrumentedLLM:
    """
    LLM wrapper que expone Prometheus metrics.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _call_llm(self, prompt: str) -> dict:
        """
        Simula llamada al LLM (mock).
        En producción: llamar a OpenAI/Anthropic/etc.
        """
        time.sleep(random.uniform(0.1, 2.0))  # Simulate latency

        response = f"Mock response for: {prompt[:30]}..."
        tokens = len(response.split())

        return {
            "response": response,
            "tokens": tokens,
            "cost": tokens * 0.00002  # Mock cost
        }

    def generate(self, prompt: str) -> dict:
        """
        Genera respuesta con metrics.
        """
        # Increment active requests
        llm_active_requests.labels(model=self.model).inc()

        try:
            # Measure latency
            with llm_latency_seconds.labels(model=self.model).time():
                # Check cache
                if prompt in self.cache:
                    self.cache_hits += 1
                    result = self.cache[prompt]
                    status = "success"
                else:
                    self.cache_misses += 1
                    result = self._call_llm(prompt)
                    self.cache[prompt] = result
                    status = "success"

            # Update metrics
            llm_requests_total.labels(
                model=self.model,
                endpoint="/generate",
                status=status
            ).inc()

            llm_tokens_total.labels(model=self.model, type="output").inc(result["tokens"])
            llm_cost_total.labels(model=self.model).inc(result["cost"])
            llm_response_length.labels(model=self.model).observe(result["tokens"])

            # Update cache hit rate
            total = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total if total > 0 else 0
            llm_cache_hit_rate.labels(endpoint="/generate").set(hit_rate)

            return result

        except Exception as e:
            # Log error
            llm_requests_total.labels(
                model=self.model,
                endpoint="/generate",
                status="error"
            ).inc()
            raise

        finally:
            # Decrement active requests
            llm_active_requests.labels(model=self.model).dec()


# ============================================================================
# PROMETHEUS EXPORTER
# ============================================================================

def expose_metrics(port: int = 8000):
    """
    Expone metrics en /metrics endpoint.

    En producción:
        from prometheus_client import start_http_server
        start_http_server(8000)
    """
    print(f"📊 Metrics exposed at http://localhost:{port}/metrics")
    print("   Configure Prometheus to scrape this endpoint\n")


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_basic_metrics():
    """Metrics básicas."""
    print("="*70)
    print("DEMO 1: Basic Metrics Collection")
    print("="*70 + "\n")

    llm = InstrumentedLLM("gpt-3.5-turbo")

    # Simulate requests
    prompts = [
        "What is AI?",
        "Explain quantum computing",
        "What is AI?",  # Cache hit
        "Summarize this article",
    ]

    print("📞 Simulating LLM calls...\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"Call {i}: {prompt}")
        result = llm.generate(prompt)
        print(f"   ✅ {result['tokens']} tokens, ${result['cost']:.5f}\n")

    # Print metrics
    print("\n📊 Collected Metrics:")
    print(generate_latest(REGISTRY).decode('utf-8'))


def demo_grafana_dashboard():
    """Grafana dashboard."""
    print("\n" + "="*70)
    print("DEMO 2: Grafana Dashboard Example")
    print("="*70 + "\n")

    print("📊 GRAFANA DASHBOARD PANELS:\n")
    print("""
Panel 1: Request Rate
─────────────────────────────────────────────────
Query: rate(llm_requests_total[5m])

   150 req/s ┤                                    ╭─
   100 req/s ┤                             ╭──────╯
    50 req/s ┤                   ╭─────────╯
     0 req/s └───────────────────┘
              12:00  12:05  12:10  12:15  12:20


Panel 2: P95 Latency
─────────────────────────────────────────────────
Query: histogram_quantile(0.95, llm_latency_seconds_bucket)

    5.0s ┤
    2.5s ┤                                      ╭─
    1.0s ┤                            ╭─────────╯
    0.5s ┤              ╭─────────────╯
      0s └──────────────┘
          12:00  12:05  12:10  12:15  12:20


Panel 3: Error Rate
─────────────────────────────────────────────────
Query: rate(llm_requests_total{status="error"}[5m])

    5% ┤
    2% ┤                                        ╭╮
    0% ┤────────────────────────────────────────╯╰─
       12:00  12:05  12:10  12:15  12:20


Panel 4: Cost per Hour
─────────────────────────────────────────────────
Query: rate(llm_cost_usd_total[1h])

   $10/hr ┤                                    ╭─
    $5/hr ┤                             ╭──────╯
    $0/hr └─────────────────────────────╯
           12:00  12:05  12:10  12:15  12:20


Panel 5: Cache Hit Rate
─────────────────────────────────────────────────
Query: llm_cache_hit_rate

   100% ┤                               ╭───────
    75% ┤                       ╭───────╯
    50% ┤               ╭───────╯
    25% ┤       ╭───────╯
     0% └───────╯
         12:00  12:05  12:10  12:15  12:20
    """)


def demo_prometheus_queries():
    """PromQL queries."""
    print("\n" + "="*70)
    print("DEMO 3: Useful PromQL Queries")
    print("="*70 + "\n")

    queries = [
        ("Request rate (per second)", "rate(llm_requests_total[5m])"),
        ("Success rate", "rate(llm_requests_total{status='success'}[5m]) / rate(llm_requests_total[5m])"),
        ("P50 latency", "histogram_quantile(0.50, llm_latency_seconds_bucket)"),
        ("P95 latency", "histogram_quantile(0.95, llm_latency_seconds_bucket)"),
        ("P99 latency", "histogram_quantile(0.99, llm_latency_seconds_bucket)"),
        ("Cost per hour", "rate(llm_cost_usd_total[1h])"),
        ("Average tokens per request", "rate(llm_tokens_total[5m]) / rate(llm_requests_total[5m])"),
        ("Active requests", "llm_active_requests"),
        ("Error rate", "rate(llm_requests_total{status='error'}[5m])"),
    ]

    for i, (description, query) in enumerate(queries, 1):
        print(f"{i}. {description}:")
        print(f"   {query}\n")


def demo_alerts():
    """Alerting rules."""
    print("="*70)
    print("DEMO 4: Prometheus Alerting Rules")
    print("="*70 + "\n")

    print("📋 alerting_rules.yml:\n")
    print("""
groups:
  - name: llm_alerts
    interval: 30s
    rules:

    # High error rate
    - alert: HighLLMErrorRate
      expr: rate(llm_requests_total{status="error"}[5m]) > 0.05
      for: 5m
      labels:
        severity: warning
      annotations:
        description: "LLM error rate is {{ $value }}% (threshold: 5%)"

    # High latency
    - alert: HighLLMLatency
      expr: histogram_quantile(0.95, llm_latency_seconds_bucket) > 5.0
      for: 10m
      labels:
        severity: warning
      annotations:
        description: "P95 latency is {{ $value }}s (threshold: 5s)"

    # Cost spike
    - alert: LLMCostSpike
      expr: rate(llm_cost_usd_total[1h]) > 50
      for: 5m
      labels:
        severity: critical
      annotations:
        description: "Cost is ${{ $value }}/hr (threshold: $50/hr)"

    # Low cache hit rate
    - alert: LowCacheHitRate
      expr: llm_cache_hit_rate < 0.3
      for: 30m
      labels:
        severity: info
      annotations:
        description: "Cache hit rate is {{ $value }} (threshold: 0.3)"
    """)


def demo_setup_guide():
    """Setup guide."""
    print("\n" + "="*70)
    print("DEMO 5: Setup Guide")
    print("="*70 + "\n")

    print("🚀 SETUP PROMETHEUS + GRAFANA:\n")
    print("""
1. Start Prometheus:

   prometheus.yml:
   ───────────────────────────────────────
   scrape_configs:
     - job_name: 'llm_app'
       static_configs:
         - targets: ['localhost:8000']
   ───────────────────────────────────────

   $ docker run -p 9090:9090 \\
       -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \\
       prom/prometheus

2. Instrument your app:

   from prometheus_client import start_http_server, Counter

   requests_total = Counter('requests_total', 'Total requests')

   start_http_server(8000)  # Expose metrics

   # Your app code
   requests_total.inc()  # Increment counter

3. Start Grafana:

   $ docker run -p 3000:3000 grafana/grafana

   • Visit http://localhost:3000
   • Add Prometheus datasource (http://prometheus:9090)
   • Import dashboard

4. Create alerts:

   • Configure alerting_rules.yml
   • Set up notification channels (Slack, PagerDuty, Email)
   • Test alerts
    """)


if __name__ == "__main__":
    print("\n🎯 METRICS DASHBOARD WITH PROMETHEUS")
    print("📊 Monitor LLM applications in production\n")

    demo_basic_metrics()
    demo_grafana_dashboard()
    demo_prometheus_queries()
    demo_alerts()
    demo_setup_guide()

    print("\n" + "="*70)
    print("💡 BEST PRACTICES:")
    print("="*70)
    print("  ✅ Expose metrics at /metrics endpoint")
    print("  ✅ Use Counter for totals (requests, errors)")
    print("  ✅ Use Histogram for distributions (latency)")
    print("  ✅ Use Gauge for current values (active requests)")
    print("  ✅ Set appropriate bucket sizes")
    print("  ✅ Create alerts for key metrics")
    print("  ✅ Review dashboards weekly")
    print("  ✅ Set up PagerDuty for critical alerts")

    print("\n📚 Resources:")
    print("  • Prometheus: https://prometheus.io/")
    print("  • Grafana: https://grafana.com/")
    print("  • prometheus-client: https://github.com/prometheus/client_python")
