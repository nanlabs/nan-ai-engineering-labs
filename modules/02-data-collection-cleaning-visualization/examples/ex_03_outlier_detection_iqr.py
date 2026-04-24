"""Detect outliers with interquartile range (IQR) fences.

Run:
    python modules/02-data-collection-cleaning-visualization/examples/ex_03_outlier_detection_iqr.py
"""

from __future__ import annotations


def percentile(sorted_values: list[float], q: float) -> float:
    """Return percentile using linear interpolation."""
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def detect_iqr_outliers(values: list[float]) -> tuple[float, float, list[float]]:
    """Return lower/upper fences and outlier list."""
    sorted_values = sorted(values)
    q1 = percentile(sorted_values, 0.25)
    q3 = percentile(sorted_values, 0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = [value for value in values if value < lower or value > upper]
    return lower, upper, outliers


def main() -> None:
    """Run IQR outlier detection baseline."""
    response_times = [110, 125, 118, 130, 127, 119, 600, 122, 128, 121]
    lower, upper, outliers = detect_iqr_outliers(response_times)

    print("IQR outlier detection baseline")
    print(f"lower_fence: {lower:.2f}")
    print(f"upper_fence: {upper:.2f}")
    print(f"outliers: {outliers}")


if __name__ == "__main__":
    main()
