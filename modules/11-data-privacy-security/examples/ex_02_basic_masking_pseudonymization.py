"""Basic masking and pseudonymization demo.

Run:
    python modules/11-data-privacy-security/examples/ex_02_basic_masking_pseudonymization.py
"""

from __future__ import annotations

import hashlib


def mask_email(email: str) -> str:
    """Mask local part while preserving domain."""
    local, domain = email.split("@", maxsplit=1)
    return f"{local[:1]}***@{domain}"


def pseudonymize(value: str) -> str:
    """Hash value into deterministic pseudonym."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    """Show masking and pseudonymization on sample records."""
    email = "ana.garcia@example.com"
    customer_id = "customer-001"

    print(f"masked_email={mask_email(email)}")
    print(f"pseudonymized_id={pseudonymize(customer_id)}")


if __name__ == "__main__":
    main()
