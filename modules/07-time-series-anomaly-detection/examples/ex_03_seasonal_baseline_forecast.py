"""Seasonal baseline forecasting demo.

Run:
    python modules/07-time-series-anomaly-detection/examples/ex_03_seasonal_baseline_forecast.py
"""

from __future__ import annotations


def seasonal_forecast(train: list[float], season_length: int, horizon: int) -> list[float]:
    """Repeat the last observed season."""
    season = train[-season_length:]
    forecast: list[float] = []
    while len(forecast) < horizon:
        forecast.extend(season)
    return forecast[:horizon]


def mape(actual: list[float], predicted: list[float]) -> float:
    """Compute mean absolute percentage error."""
    parts = [abs((a - p) / a) for a, p in zip(actual, predicted, strict=True) if a != 0]
    return 100 * sum(parts) / len(parts)


def main() -> None:
    """Evaluate seasonal baseline on synthetic monthly demand."""
    train = [30, 40, 45, 35, 30, 42, 48, 37]
    test = [31, 41, 47, 36]

    pred = seasonal_forecast(train, season_length=4, horizon=4)
    print(f"seasonal_pred={pred}")
    print(f"mape={mape(test, pred):.2f}%")


if __name__ == "__main__":
    main()
