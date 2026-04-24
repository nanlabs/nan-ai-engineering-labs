"""Pipeline security checklist scoring demo.

Run:
    python modules/11-data-privacy-security/examples/ex_04_pipeline_security_checklist.py
"""

from __future__ import annotations


def checklist_score(checks: dict[str, bool]) -> tuple[int, int]:
    """Return passed checks and total checks."""
    passed = sum(1 for value in checks.values() if value)
    return passed, len(checks)


def main() -> None:
    """Evaluate a basic security checklist."""
    checks = {
        "encrypted_storage": True,
        "masked_exports": True,
        "least_privilege": False,
        "audit_logs": True,
        "secret_rotation": False,
    }

    passed, total = checklist_score(checks)
    print(f"checks={checks}")
    print(f"passed={passed}/{total}")


if __name__ == "__main__":
    main()
