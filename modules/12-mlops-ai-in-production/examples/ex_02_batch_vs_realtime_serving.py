"""Batch vs realtime serving trade-off demo.

Run:
    python modules/12-mlops-ai-in-production/examples/ex_02_batch_vs_realtime_serving.py
"""

from __future__ import annotations


def choose_serving_mode(latency_ms: int, daily_volume: int) -> str:
    """Choose serving mode using simple operational heuristics."""
    if latency_ms <= 200:
        return "realtime"
    if daily_volume >= 10000:
        return "batch"
    return "hybrid"


def main() -> None:
    """Print serving mode decisions for sample workloads."""
    scenarios = [
        (80, 500),
        (1500, 50000),
        (400, 3000),
    ]

    for latency_ms, daily_volume in scenarios:
        mode = choose_serving_mode(latency_ms, daily_volume)
        print(f"latency_ms={latency_ms} daily_volume={daily_volume} mode={mode}")


if __name__ == "__main__":
    main()
