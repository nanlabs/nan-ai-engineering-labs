# Examples — Machine Learning Fundamentals

## Example 1 — Regression baseline

Train a linear regression baseline and measure MAE/MSE.

## Example 2 — Binary classification

Train a simple classifier and evaluate a confusion matrix.

## Example 3 — Cross-validation

Compare performance with `KFold`.

## Example 4 — Basic tuning

Apply `GridSearchCV` to a minimal hyperparameter set.

## Rules

- Each example must explain the input, output, and success criteria.
- Store observations in `notes/`.

## Available examples

### Executable scripts (phase-2 pilot)

1. `ex_01_train_test_split_baseline.py`

   - Reproducible binary-classification baseline with `train_test_split`.
   - Expected output: `Accuracy` and `F1 score` above 0.90.

1. `ex_02_linear_vs_logistic_baseline.py`

   - Comparison between regression and classification baselines.
   - Expected output: `MAE` for diabetes plus `Accuracy` for breast_cancer.

1. `ex_03_regularization_tradeoffs.py`

   - Effect of L2 regularization on weights and error.
   - Expected output: changes in `w`, `train_mse`, and `test_mse` by `lambda`.

1. `ex_04_tree_depth_vs_overfitting.py`

   - Simulation of overfitting behavior as model complexity increases.
   - Expected output: high `train_acc` with lower generalization.

1. `ex_05_model_comparison_metrics.py`

   - Comparison of models with `accuracy`, `precision`, `recall`, and `f1`.
   - Expected output: ranking differences by metric.

1. `ex_06_reproducible_ml_pipeline.py`

   - Minimal reproducible pipeline with a fixed seed.
   - Expected output: stable `threshold` and `accuracy` across runs.

## How to use these examples

```bash
python modules/03-machine-learning-fundamentals/examples/ex_01_train_test_split_baseline.py
python modules/03-machine-learning-fundamentals/examples/ex_02_linear_vs_logistic_baseline.py
python modules/03-machine-learning-fundamentals/examples/ex_03_regularization_tradeoffs.py
python modules/03-machine-learning-fundamentals/examples/ex_04_tree_depth_vs_overfitting.py
python modules/03-machine-learning-fundamentals/examples/ex_05_model_comparison_metrics.py
python modules/03-machine-learning-fundamentals/examples/ex_06_reproducible_ml_pipeline.py
```

Recommendation: run example 01 first, then modify `test_size` to observe
metric stability.

## Next steps

1. Connect these examples to model-comparison practices.
1. Add an L4 example with automated cross-validation.
1. Record results and observations in `notes/README.md`.
