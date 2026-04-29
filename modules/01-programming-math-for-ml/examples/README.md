# Examples — Programming and Math for ML

This folder contains practical examples with complete code, step-by-step logic,
and expected outputs.

## Available examples

### Existing guided markdown examples

1. [01 - Vectors and dot product](./01-vectors-dot-product.md)
1. [02 - Descriptive statistics](./02-descriptive-statistics.md)

### Executable scripts (phase-2 continuation)

1. `ex_01_vector_dot_product_baseline.py`

   - Computes dot product and cosine similarity with pure Python vectors.
   - Expected output: positive similarity for aligned vectors and lower for orthogonal-ish vectors.

1. `ex_02_descriptive_statistics_baseline.py`

   - Computes mean, median, variance, and standard deviation on a small dataset.
   - Expected output: deterministic summary statistics.

1. `ex_03_feature_scaling_normalization.py`

   - Compares min-max scaling and z-score normalization.
   - Expected output: transformed values in stable ranges.

1. `ex_04_linear_regression_from_scratch.py`

   - Fits a one-feature linear model with a closed-form solution.
   - Expected output: slope/intercept close to the synthetic data trend.

1. `ex_05_gradient_descent_one_parameter.py`

   - Optimizes one parameter with gradient descent on MSE.
   - Expected output: monotonically decreasing loss.

1. `ex_06_reproducible_math_pipeline.py`

   - Runs a deterministic mini pipeline using a fixed random seed.
   - Expected output: repeated runs produce `same_result=True`.

## How to use these examples

```bash
python modules/01-programming-math-for-ml/examples/ex_01_vector_dot_product_baseline.py
python modules/01-programming-math-for-ml/examples/ex_02_descriptive_statistics_baseline.py
python modules/01-programming-math-for-ml/examples/ex_03_feature_scaling_normalization.py
python modules/01-programming-math-for-ml/examples/ex_04_linear_regression_from_scratch.py
python modules/01-programming-math-for-ml/examples/ex_05_gradient_descent_one_parameter.py
python modules/01-programming-math-for-ml/examples/ex_06_reproducible_math_pipeline.py
```

Recommended order: fundamentals (`01-03`) first, then modeling (`04-05`), and
finally reproducibility (`06`).

## Next steps

1. Connect these scripts with the module `practices/` exercises.
1. Add one geometric-intuition script for projections.
1. Record common mistakes and fixes in `notes/README.md`.
