"""Classify columns by data sensitivity.

Run:
    python modules/11-data-privacy-security/examples/ex_01_data_classification_matrix.py
"""

from __future__ import annotations


def classify_column(name: str) -> str:
    """Assign simple sensitivity label based on column name."""
    lowered = name.lower()
    if any(token in lowered for token in ["password", "token", "secret"]):
        return "restricted"
    if any(token in lowered for token in ["email", "phone", "address", "ssn"]):
        return "confidential"
    if any(token in lowered for token in ["name", "city", "country"]):
        return "internal"
    return "public"


def main() -> None:
    """Print a classification matrix for sample columns."""
    columns = ["user_id", "email", "country", "password_hash", "phone_number"]
    matrix = {column: classify_column(column) for column in columns}
    print(matrix)


if __name__ == "__main__":
    main()
