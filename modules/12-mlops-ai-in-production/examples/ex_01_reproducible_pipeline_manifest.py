"""Reproducible ML pipeline manifest demo.

Run:
    python modules/12-mlops-ai-in-production/examples/ex_01_reproducible_pipeline_manifest.py
"""

from __future__ import annotations


def build_manifest() -> dict[str, str]:
    """Return a lightweight manifest for reproducibility."""
    return {
        "dataset_version": "customers-v3",
        "feature_config": "feature-set-a",
        "training_code": "commit-abc123",
        "model_version": "model-v1.0.0",
    }


def main() -> None:
    """Print a deterministic manifest."""
    print(build_manifest())


if __name__ == "__main__":
    main()
