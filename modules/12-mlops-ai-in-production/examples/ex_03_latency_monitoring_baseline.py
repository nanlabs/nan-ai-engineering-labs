"""Latency monitoring baseline with alert threshold.

Run:
    python modules/12-mlops-ai-in-production/examples/ex_03_latency_monitoring_baseline.py
"""

from __future__ import annotations


def summary(latencies: list[int]) -> tuple[float, int, int]:
    """Return average, min, and max latency."""
    avg = sum(latencies) / len(latencies)
    return avg, min(latencies), max(latencies)


def main() -> None:
    """Print latency summary and alert status."""
    latencies = [80, 95, 110, 130, 125, 145, 90]
    avg, min_v, max_v = summary(latencies)
    alert = max_v > 140

    print(f"avg_latency={avg:.2f}")
    print(f"min_latency={min_v} max_latency={max_v}")
    print(f"alert_triggered={alert}")


if __name__ == "__main__":
    main()
