# Examples — Deep Learning Basics

## Example 1 — First neural network

Train a simple MLP for binary classification.

## Example 2 — Training curves

Visualize loss and accuracy on train/validation across epochs.

## Example 3 — Dropout and regularization

Compare results with and without regularization.

## Example 4 — Hyperparameter tuning

Evaluate the impact of learning rate, batch size, and number of layers.

## Rules

- Include the objective, input, expected output, and common errors.
- Keep notebooks/scripts reproducible.

## Available examples

### Executable scripts (phase-2 pilot)

1. `ex_01_neuron_forward_baseline.py`

   - Forward pass of a neuron with sigmoid activation.
   - Expected output: `output_a` and `output_b` between 0 and 1.

1. `ex_02_gradient_descent_baseline.py`

   - Single-parameter training with gradient descent.
   - Expected output: progressive decrease in `mse`.

1. `ex_03_activation_functions_comparison.py`

   - Comparison of ReLU, tanh, and sigmoid on common inputs.
   - Expected output: a table showing behavioral differences.

1. `ex_04_overfitting_regularization_simulation.py`

   - Simulation of overfitting and model selection with penalization.
   - Expected output: selected model changes based on `alpha`.

1. `ex_05_simple_mlp_from_scratch.py`

   - Forward pass of a small MLP without frameworks.
   - Expected output: reproducible predictions per sample.

1. `ex_06_reproducible_training_loop.py`

   - Reproducible training loop with a fixed seed.
   - Expected output: two identical runs (`same_result=True`).

## How to use these examples

```bash
python modules/04-deep-learning-basics/examples/ex_01_neuron_forward_baseline.py
python modules/04-deep-learning-basics/examples/ex_02_gradient_descent_baseline.py
python modules/04-deep-learning-basics/examples/ex_03_activation_functions_comparison.py
python modules/04-deep-learning-basics/examples/ex_04_overfitting_regularization_simulation.py
python modules/04-deep-learning-basics/examples/ex_05_simple_mlp_from_scratch.py
python modules/04-deep-learning-basics/examples/ex_06_reproducible_training_loop.py
```

Recommendation: run `01` through `06` in order to preserve the conceptual progression.

## Next steps

1. Add an L4 variant with mini-batch training and early stopping.
1. Connect examples 04 and 06 with generalization practices.
1. Document stability observations in `notes/README.md`.
