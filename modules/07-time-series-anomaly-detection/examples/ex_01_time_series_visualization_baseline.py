"""Describe trend and seasonality from a synthetic time series.

Run:
    python modules/07-time-series-anomaly-detection/examples/ex_01_time_series_visualization_baseline.py
"""

from __future__ import annotations


def moving_average(values: list[float], window: int) -> list[float]:
    """Compute trailing moving average."""
    result: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        result.append(sum(segment) / len(segment))
    return result


def main() -> None:
    """Print basic trend signals without external plotting dependencies."""
    series = [10, 11, 13, 15, 14, 16, 18, 20, 19, 21, 23, 25]
    ma3 = moving_average(series, window=3)

    slope = series[-1] - series[0]
    direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"

    print(f"series_length={len(series)}")
    print(f"start={series[0]} end={series[-1]} trend={direction}")
    print(f"moving_avg_last_3={ma3[-3:]}")


if __name__ == "__main__":
    main()
