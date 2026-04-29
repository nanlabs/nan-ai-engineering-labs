# Examples — Ethics, Bias & Explainability

## Example 1 — Basic bias detection

Review group distributions and potential imbalances.

## Example 2 — Fairness metric

Calculate a fairness metric and compare it across groups.

## Example 3 — Local explanation

Use SHAP/LIME-style ideas to explain an individual prediction.

## Example 4 — Risk report

Document risks and mitigations in a short format.

## Rules

- Avoid conclusions without quantitative evidence.
- Record assumptions explicitly.

## Available examples

### Executable scripts (phase-2 continuation)

1. `ex_01_group_distribution_bias_check.py`

   - Checks positive-rate distribution by group.
   - Expected output: group rates and maximum disparity gap.

1. `ex_02_demographic_parity_metric.py`

   - Computes demographic parity difference between groups.
   - Expected output: positive rate per group and DP gap.

1. `ex_03_counterfactual_fairness_check.py`

   - Measures score sensitivity under protected-attribute perturbation.
   - Expected output: factual vs counterfactual score delta.

1. `ex_04_local_explanation_feature_contributions.py`

   - Builds a local explanation via feature contribution breakdown.
   - Expected output: per-feature contributions and final prediction.

1. `ex_05_risk_report_generator.py`

   - Produces a concise model risk report from fairness indicators.
   - Expected output: dictionary-style risk summary.

1. `ex_06_reproducible_ethics_audit_pipeline.py`

   - Runs deterministic ethics audit pipeline with fixed seed.
   - Expected output: identical group rates across repeated runs.

## How to use these examples

```bash
python modules/10-ethics-bias-explainability/examples/ex_01_group_distribution_bias_check.py
python modules/10-ethics-bias-explainability/examples/ex_02_demographic_parity_metric.py
python modules/10-ethics-bias-explainability/examples/ex_03_counterfactual_fairness_check.py
python modules/10-ethics-bias-explainability/examples/ex_04_local_explanation_feature_contributions.py
python modules/10-ethics-bias-explainability/examples/ex_05_risk_report_generator.py
python modules/10-ethics-bias-explainability/examples/ex_06_reproducible_ethics_audit_pipeline.py
```

Recommended order: distribution -> fairness metric -> counterfactual -> explanation -> risk report -> reproducibility.

## Next steps

1. Add equalized-odds style checks for classification pipelines.
1. Add a lightweight mitigation demo (thresholding or reweighting).
1. Document governance checklist outcomes in `notes/README.md`.
