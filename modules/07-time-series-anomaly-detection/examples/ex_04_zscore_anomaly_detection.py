"""Detect anomalies in a time series using z-score.

Run:
    python modules/07-time-series-anomaly-detection/examples/ex_04_zscore_anomaly_detection.py
"""

from __future__ import annotations


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float], avg: float) -> float:
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return variance**0.5


def detect_anomalies(values: list[float], threshold: float = 2.0) -> list[tuple[int, float, float]]:
    """Return index, value, and z-score for points above threshold."""
    avg = mean(values)
    deviation = std(values, avg)
    anomalies: list[tuple[int, float, float]] = []

    for idx, value in enumerate(values):
        if deviation == 0:
            continue
        z_value = abs((value - avg) / deviation)
        if z_value >= threshold:
            anomalies.append((idx, value, z_value))

    return anomalies


def main() -> None:
    """Run anomaly detection on synthetic metric data."""
    metric = [50, 51, 49, 52, 50, 120, 51, 49, 50, 48]
    anomalies = detect_anomalies(metric, threshold=2.0)

    print(f"series={metric}")
    print(f"anomalies_count={len(anomalies)}")
    for idx, value, z_value in anomalies:
        print(f"index={idx} value={value} z={z_value:.4f}")


if __name__ == "__main__":
    main()
