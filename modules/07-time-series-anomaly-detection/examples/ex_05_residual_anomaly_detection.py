"""Residual-based anomaly detection for one-step forecasts.

Run:
    python modules/07-time-series-anomaly-detection/examples/ex_05_residual_anomaly_detection.py
"""

from __future__ import annotations


def one_step_naive_forecast(series: list[float]) -> list[float]:
    """Forecast each point using the previous observed value."""
    return [series[idx - 1] for idx in range(1, len(series))]


def detect_residual_anomalies(actual: list[float], predicted: list[float], threshold: float) -> list[tuple[int, float]]:
    """Return index and residual for anomalies above threshold."""
    anomalies: list[tuple[int, float]] = []
    for idx, (a_value, p_value) in enumerate(zip(actual, predicted, strict=True), start=1):
        residual = abs(a_value - p_value)
        if residual >= threshold:
            anomalies.append((idx, residual))
    return anomalies


def main() -> None:
    """Run residual anomaly detection on synthetic latency metric."""
    series = [100, 101, 99, 100, 102, 140, 103, 101, 100]
    predicted = one_step_naive_forecast(series)
    actual = series[1:]

    anomalies = detect_residual_anomalies(actual, predicted, threshold=20)

    print(f"residual_points={len(actual)}")
    print(f"anomalies_count={len(anomalies)}")
    for idx, residual in anomalies:
        print(f"index={idx} residual={residual:.2f}")


if __name__ == "__main__":
    main()
