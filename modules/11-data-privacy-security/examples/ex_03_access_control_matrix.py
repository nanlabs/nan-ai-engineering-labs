"""Role-based access control matrix demo.

Run:
    python modules/11-data-privacy-security/examples/ex_03_access_control_matrix.py
"""

from __future__ import annotations


def can_access(role: str, resource: str) -> bool:
    """Check minimal RBAC policy."""
    policy = {
        "analyst": {"aggregated_metrics", "masked_dataset"},
        "engineer": {"aggregated_metrics", "masked_dataset", "feature_store"},
        "admin": {"aggregated_metrics", "masked_dataset", "feature_store", "raw_dataset"},
    }
    return resource in policy.get(role, set())


def main() -> None:
    """Print access results for sample role/resource pairs."""
    checks = [
        ("analyst", "raw_dataset"),
        ("engineer", "feature_store"),
        ("admin", "raw_dataset"),
    ]

    for role, resource in checks:
        print(f"role={role} resource={resource} allowed={can_access(role, resource)}")


if __name__ == "__main__":
    main()
