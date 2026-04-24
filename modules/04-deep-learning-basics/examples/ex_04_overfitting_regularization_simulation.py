"""Simulate overfitting and regularization impact.

Run:
    python modules/04-deep-learning-basics/examples/ex_04_overfitting_regularization_simulation.py
"""

from __future__ import annotations


def score(train_error: float, val_error: float, alpha: float) -> float:
    """Combine train and validation error with regularization alpha."""
    gap_penalty = abs(val_error - train_error)
    return val_error + alpha * gap_penalty


def main() -> None:
    """Compare model candidates under different regularization strengths."""
    candidates = {
        "small_model": (0.35, 0.38),
        "complex_model": (0.10, 0.55),
        "balanced_model": (0.22, 0.28),
    }

    for alpha in [0.0, 0.5, 1.0]:
        print(f"alpha={alpha:.1f}")
        best_name = ""
        best_value = float("inf")
        for name, (train_error, val_error) in candidates.items():
            value = score(train_error, val_error, alpha)
            print(f"  {name}: train={train_error:.2f} val={val_error:.2f} score={value:.4f}")
            if value < best_value:
                best_name, best_value = name, value
        print(f"  selected={best_name}\n")


if __name__ == "__main__":
    main()
