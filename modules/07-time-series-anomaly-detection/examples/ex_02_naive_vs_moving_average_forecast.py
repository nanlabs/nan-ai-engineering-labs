"""Compare naive and moving-average forecasts.

Run:
    python modules/07-time-series-anomaly-detection/examples/ex_02_naive_vs_moving_average_forecast.py
"""

from __future__ import annotations


def mae(actual: list[float], predicted: list[float]) -> float:
    """Compute mean absolute error."""
    errors = [abs(a - p) for a, p in zip(actual, predicted, strict=True)]
    return sum(errors) / len(errors)


def naive_forecast(train: list[float], horizon: int) -> list[float]:
    """Forecast with the last observed value."""
    return [train[-1]] * horizon


def moving_average_forecast(train: list[float], horizon: int, window: int = 3) -> list[float]:
    """Forecast with trailing moving average value."""
    baseline = sum(train[-window:]) / window
    return [baseline] * horizon


def main() -> None:
    """Report MAE for two lightweight baselines."""
    train = [100, 103, 101, 106, 108, 110, 109, 113]
    test = [114, 116, 115]

    naive_pred = naive_forecast(train, horizon=len(test))
    ma_pred = moving_average_forecast(train, horizon=len(test), window=3)

    print(f"naive_pred={naive_pred} mae={mae(test, naive_pred):.4f}")
    print(f"ma_pred={ma_pred} mae={mae(test, ma_pred):.4f}")


if __name__ == "__main__":
    main()
