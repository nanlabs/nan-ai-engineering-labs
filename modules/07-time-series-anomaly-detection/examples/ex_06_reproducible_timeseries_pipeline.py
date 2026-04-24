"""Reproducible end-to-end mini pipeline for forecasting + anomaly check.

Run:
    python modules/07-time-series-anomaly-detection/examples/ex_06_reproducible_timeseries_pipeline.py
"""

from __future__ import annotations

import random


def generate_series(seed: int, length: int = 12) -> list[float]:
    """Generate deterministic synthetic series with mild trend and noise."""
    random.seed(seed)
    values: list[float] = []
    base = 50.0
    for step in range(length):
        noise = random.uniform(-1.5, 1.5)
        values.append(base + step * 0.8 + noise)
    return values


def forecast_next(series: list[float], window: int = 3) -> float:
    """Forecast next value using trailing moving average."""
    return sum(series[-window:]) / window


def main() -> None:
    """Run deterministic pipeline twice and verify same outputs."""
    series_a = generate_series(seed=24)
    pred_a = forecast_next(series_a)

    series_b = generate_series(seed=24)
    pred_b = forecast_next(series_b)

    print(f"last_3_a={series_a[-3:]}")
    print(f"prediction_a={pred_a:.4f}")
    print(f"prediction_b={pred_b:.4f}")
    print(f"same_result={series_a == series_b and pred_a == pred_b}")


if __name__ == "__main__":
    main()
