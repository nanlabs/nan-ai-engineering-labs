"""Tiny one-hidden-layer MLP forward pass from scratch.

Run:
    python modules/04-deep-learning-basics/examples/ex_05_simple_mlp_from_scratch.py
"""

from __future__ import annotations


def relu(value: float) -> float:
    return value if value > 0 else 0.0


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + (2.718281828 ** (-value)))


def dense(inputs: list[float], weights: list[list[float]], biases: list[float]) -> list[float]:
    """Compute dense layer output."""
    outputs: list[float] = []
    for neuron_weights, bias in zip(weights, biases, strict=True):
        outputs.append(sum(x * w for x, w in zip(inputs, neuron_weights, strict=True)) + bias)
    return outputs


def main() -> None:
    """Run a deterministic MLP forward pass for two samples."""
    hidden_w = [[0.6, -0.2], [0.1, 0.8], [-0.5, 0.4]]
    hidden_b = [0.1, -0.1, 0.2]
    out_w = [[0.7, -0.3, 0.5]]
    out_b = [-0.2]

    for sample in ([1.0, 0.2], [0.1, 0.9]):
        hidden_raw = dense(list(sample), hidden_w, hidden_b)
        hidden_act = [relu(value) for value in hidden_raw]
        output_raw = dense(hidden_act, out_w, out_b)[0]
        prediction = sigmoid(output_raw)
        print(f"sample={sample} prediction={prediction:.4f}")


if __name__ == "__main__":
    main()
