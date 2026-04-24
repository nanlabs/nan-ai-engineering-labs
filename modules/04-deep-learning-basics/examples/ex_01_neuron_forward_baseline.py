"""Single-neuron forward pass baseline.

Run:
    python modules/04-deep-learning-basics/examples/ex_01_neuron_forward_baseline.py
"""

from __future__ import annotations


def sigmoid(value: float) -> float:
    """Compute sigmoid activation."""
    return 1.0 / (1.0 + (2.718281828 ** (-value)))


def neuron_forward(inputs: list[float], weights: list[float], bias: float) -> float:
    """Compute weighted sum followed by sigmoid."""
    z_value = sum(x * w for x, w in zip(inputs, weights, strict=True)) + bias
    return sigmoid(z_value)


def main() -> None:
    """Run deterministic forward examples."""
    weights = [0.9, -0.4, 0.7]
    bias = -0.1

    sample_a = [1.0, 0.5, 0.2]
    sample_b = [0.2, 1.2, -0.1]

    out_a = neuron_forward(sample_a, weights, bias)
    out_b = neuron_forward(sample_b, weights, bias)

    print(f"output_a={out_a:.4f}")
    print(f"output_b={out_b:.4f}")


if __name__ == "__main__":
    main()
