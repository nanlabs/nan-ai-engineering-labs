"""Reproducible privacy audit pipeline demo.

Run:
    python modules/11-data-privacy-security/examples/ex_06_reproducible_privacy_audit_pipeline.py
"""

from __future__ import annotations

import random


def generate_audit_findings(seed: int) -> dict[str, int]:
    """Generate deterministic audit counts by category."""
    random.seed(seed)
    return {
        "access_control": random.randint(0, 3),
        "data_exposure": random.randint(0, 3),
        "logging_gaps": random.randint(0, 3),
    }


def main() -> None:
    """Run deterministic audit twice and verify stable output."""
    findings_a = generate_audit_findings(seed=29)
    findings_b = generate_audit_findings(seed=29)

    print(f"findings_a={findings_a}")
    print(f"findings_b={findings_b}")
    print(f"same_result={findings_a == findings_b}")


if __name__ == "__main__":
    main()
