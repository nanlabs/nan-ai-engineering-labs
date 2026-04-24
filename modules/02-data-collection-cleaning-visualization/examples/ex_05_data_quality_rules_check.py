"""Validate simple data-quality rules over tabular rows.

Run:
    python modules/02-data-collection-cleaning-visualization/examples/ex_05_data_quality_rules_check.py
"""

from __future__ import annotations


def build_rows() -> list[dict[str, object | None]]:
    """Return deterministic sample records with mixed quality."""
    return [
        {"id": 1, "age": 24, "email": "ana@example.com"},
        {"id": 2, "age": 31, "email": "bob@example.com"},
        {"id": 2, "age": 31, "email": "bob@example.com"},
        {"id": 3, "age": -4, "email": "carla@example.com"},
        {"id": 4, "age": 40, "email": None},
    ]


def check_rules(rows: list[dict[str, object | None]]) -> dict[str, dict[str, object]]:
    """Evaluate common practical quality checks."""
    duplicate_rows = len(rows) - len({tuple(row.items()) for row in rows})
    invalid_age = sum(
        1 for row in rows if not isinstance(row["age"], int) or row["age"] < 0 or row["age"] > 120
    )
    missing_email = sum(1 for row in rows if not row["email"])

    return {
        "no_duplicate_rows": {"passed": duplicate_rows == 0, "issues": duplicate_rows},
        "age_in_valid_range": {"passed": invalid_age == 0, "issues": invalid_age},
        "required_email": {"passed": missing_email == 0, "issues": missing_email},
    }


def main() -> None:
    """Run quality-rule checks and print pass/fail report."""
    rules = check_rules(build_rows())

    print("Data quality rules check")
    for rule, result in rules.items():
        print(f"{rule}: passed={result['passed']} issues={result['issues']}")


if __name__ == "__main__":
    main()
