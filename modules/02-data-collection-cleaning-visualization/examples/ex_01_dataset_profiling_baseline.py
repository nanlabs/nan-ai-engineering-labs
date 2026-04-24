"""Profile a small tabular dataset with practical quality indicators.

Run:
    python modules/02-data-collection-cleaning-visualization/examples/ex_01_dataset_profiling_baseline.py
"""

from __future__ import annotations


def build_dataset() -> list[dict[str, object | None]]:
    """Return a deterministic in-memory table with intentional quality issues."""
    return [
        {"id": 1, "age": 25, "income": 41000, "city": "Cordoba"},
        {"id": 2, "age": None, "income": 39000, "city": "Mendoza"},
        {"id": 3, "age": 33, "income": None, "city": "Rosario"},
        {"id": 3, "age": 33, "income": None, "city": "Rosario"},
        {"id": 4, "age": 45, "income": 78000, "city": None},
    ]


def infer_type(values: list[object | None]) -> str:
    """Infer a simple type label from non-null values."""
    non_null = [value for value in values if value is not None]
    if not non_null:
        return "unknown"
    if all(isinstance(value, int | float) for value in non_null):
        return "numeric"
    return "categorical"


def profile_dataset(rows: list[dict[str, object | None]]) -> dict[str, object]:
    """Compute shape, missing values, duplicates, and type hints."""
    columns = list(rows[0].keys()) if rows else []

    missing_by_column: dict[str, int] = {}
    type_by_column: dict[str, str] = {}

    for column in columns:
        values = [row[column] for row in rows]
        missing_by_column[column] = sum(1 for value in values if value is None)
        type_by_column[column] = infer_type(values)

    duplicate_count = len(rows) - len({tuple(row.items()) for row in rows})

    return {
        "row_count": len(rows),
        "column_count": len(columns),
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_count,
        "type_by_column": type_by_column,
    }


def main() -> None:
    """Run baseline profiling and print deterministic summary."""
    dataset = build_dataset()
    summary = profile_dataset(dataset)

    print("Dataset profiling baseline")
    print(f"rows: {summary['row_count']} | columns: {summary['column_count']}")
    print(f"missing_by_column: {summary['missing_by_column']}")
    print(f"duplicate_rows: {summary['duplicate_rows']}")
    print(f"type_by_column: {summary['type_by_column']}")


if __name__ == "__main__":
    main()
