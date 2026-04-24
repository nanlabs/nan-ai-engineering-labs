"""Compare two classification models with multiple metrics.

Run:
    python modules/03-machine-learning-fundamentals/examples/ex_05_model_comparison_metrics.py
"""

from __future__ import annotations


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    total = tp + tn + fp + fn
    return (tp + tn) / total if total else 0.0


def report(name: str, *, tp: int, tn: int, fp: int, fn: int) -> None:
    print(name)
    print(f"accuracy={accuracy(tp, tn, fp, fn):.4f}")
    print(f"precision={precision(tp, fp):.4f}")
    print(f"recall={recall(tp, fn):.4f}")
    print(f"f1={f1_score(tp, fp, fn):.4f}")
    print()


def main() -> None:
    """Show how metrics can change model ranking."""
    report("Model A", tp=42, tn=50, fp=8, fn=20)
    report("Model B", tp=34, tn=56, fp=2, fn=28)


if __name__ == "__main__":
    main()
