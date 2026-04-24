"""Compare mean and median imputation for missing numeric values.

Run:
    python modules/02-data-collection-cleaning-visualization/examples/ex_02_missing_values_imputation.py
"""

from __future__ import annotations


def mean(values: list[float]) -> float:
    """Return arithmetic mean."""
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    """Return median value for sorted numeric list."""
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def impute(values: list[float | None], strategy: str) -> list[float]:
    """Impute missing values with a selected strategy."""
    observed = [value for value in values if value is not None]
    fill_value = mean(observed) if strategy == "mean" else median(observed)
    return [value if value is not None else fill_value for value in values]


def main() -> None:
    """Run imputation baseline and print strategy comparison."""
    income = [41000.0, 39000.0, None, None, 78000.0, 52000.0]

    mean_imputed = impute(income, strategy="mean")
    median_imputed = impute(income, strategy="median")

    print("Missing-value imputation baseline")
    print(f"mean_imputed:   {[round(value, 2) for value in mean_imputed]}")
    print(f"median_imputed: {[round(value, 2) for value in median_imputed]}")
    print(f"mean(mean_imputed): {mean(mean_imputed):.2f}")
    print(f"mean(median_imputed): {mean(median_imputed):.2f}")


if __name__ == "__main__":
    main()
