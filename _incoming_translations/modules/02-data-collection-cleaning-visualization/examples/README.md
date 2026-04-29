# Examples — Data Collection, Cleaning & Visualization

## Example 1 — Initial dataset profiling

Objective: read data with Pandas and generate a quick profile (types, nulls, duplicates).

## Example 2 — Missing-value cleaning

Objective: compare imputation strategies and justify the choice.

## Example 3 — outlier detection

Objective: apply IQR and visualize impact before/after treatment.

## Example 4 — Business-oriented visualization

Objective: create 3 charts that answer concrete analysis questions.

## Rules

- Each example must include input, expected output, and explanation.
- Every step must be reproducible in a notebook or script.

## Available examples

### Existing guided markdown examples

1. [01 - Cleaning a dirty dataset](./01-limpieza-dataset-sucio.md)
1. [02 - Full EDA on wine dataset](./02-eda-completo-dataset-vinos.md)

### Executable scripts (phase-2 continuation)

1. `ex_01_dataset_profiling_baseline.py`

   - Profiles a small tabular dataset (shape, nulls, duplicates, and simple type hints).
   - Expected output: deterministic data quality summary.

1. `ex_02_missing_values_imputation.py`

   - Compares mean and median imputation for numeric missing values.
   - Expected output: two imputed datasets with consistent summary stats.

1. `ex_03_outlier_detection_iqr.py`

   - Detects outliers using IQR fences and reports candidate rows.
   - Expected output: lower/upper bounds and flagged values.

1. `ex_04_business_visualization_ascii.py`

   - Builds simple business-friendly ASCII charts from grouped data.
   - Expected output: sorted categories and proportional bars.

1. `ex_05_data_quality_rules_check.py`

   - Validates practical quality rules (ranges, required fields, duplicates).
   - Expected output: pass/fail report by rule.

1. `ex_06_reproducible_cleaning_pipeline.py`

   - Runs a deterministic mini cleaning pipeline with fixed random seed.
   - Expected output: repeated runs produce `same_result=True`.

## How to use these examples

```bash
python modules/02-data-collection-cleaning-visualization/examples/ex_01_dataset_profiling_baseline.py
python modules/02-data-collection-cleaning-visualization/examples/ex_02_missing_values_imputation.py
python modules/02-data-collection-cleaning-visualization/examples/ex_03_outlier_detection_iqr.py
python modules/02-data-collection-cleaning-visualization/examples/ex_04_business_visualization_ascii.py
python modules/02-data-collection-cleaning-visualization/examples/ex_05_data_quality_rules_check.py
python modules/02-data-collection-cleaning-visualization/examples/ex_06_reproducible_cleaning_pipeline.py
```

Recommended order: profiling (`01`) -> cleaning (`02-03`) -> communication and
controls (`04-05`) -> reproducibility (`06`).

## Next steps

1. Connect these examples to `practices/` with a single realistic dataset.
1. Add a variant with categorical encoding and leakage checks.
1. Document common cleaning pitfalls in `notes/README.md`.
