"""Compare common activation functions on sample inputs.

Run:
    python modules/04-deep-learning-basics/examples/ex_03_activation_functions_comparison.py
"""

from __future__ import annotations


def relu(value: float) -> float:
    return value if value > 0 else 0.0


def tanh_approx(value: float) -> float:
    exp_pos = 2.718281828 ** value
    exp_neg = 2.718281828 ** (-value)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + (2.718281828 ** (-value)))


def main() -> None:
    """Print side-by-side activation values."""
    inputs = [-2.0, -0.5, 0.0, 0.5, 2.0]
    print("x | relu | tanh | sigmoid")
    for x_value in inputs:
        print(f"{x_value:>4.1f} | {relu(x_value):>4.2f} | {tanh_approx(x_value):>5.2f} | {sigmoid(x_value):>7.4f}")


if __name__ == "__main__":
    main()
