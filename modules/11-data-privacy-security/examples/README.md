# Examples — Data Privacy & Security

## Example 1 — Data classification

Label columns by sensitivity level.

## Example 2 — Basic anonymization

Apply simple masking or pseudonymization techniques.

## Example 3 — Access matrix

Define roles and minimum permissions.

## Example 4 — Pipeline security checklist

Review critical points of data exposure.

## Rules

- Prioritize practical and verifiable controls.
- Document assumptions and limits.

## Available examples

### Executable scripts (phase-2 continuation)

1. `ex_01_data_classification_matrix.py`

   - Classifies columns by practical sensitivity level.
   - Expected output: mapping from column name to security tier.

1. `ex_02_basic_masking_pseudonymization.py`

   - Demonstrates masking and deterministic pseudonymization.
   - Expected output: masked email and pseudonymized identifier.

1. `ex_03_access_control_matrix.py`

   - Checks role-based access decisions for critical resources.
   - Expected output: allow/deny decisions by role.

1. `ex_04_pipeline_security_checklist.py`

   - Scores a lightweight pipeline security checklist.
   - Expected output: passed checks count over total.

1. `ex_05_prompt_injection_risk_demo.py`

   - Classifies prompt injection risk from suspicious patterns.
   - Expected output: low/medium/high risk labels.

1. `ex_06_reproducible_privacy_audit_pipeline.py`

   - Runs a deterministic privacy audit summary with fixed seed.
   - Expected output: `same_result=True` across repeated runs.

## How to use these examples

```bash
python modules/11-data-privacy-security/examples/ex_01_data_classification_matrix.py
python modules/11-data-privacy-security/examples/ex_02_basic_masking_pseudonymization.py
python modules/11-data-privacy-security/examples/ex_03_access_control_matrix.py
python modules/11-data-privacy-security/examples/ex_04_pipeline_security_checklist.py
python modules/11-data-privacy-security/examples/ex_05_prompt_injection_risk_demo.py
python modules/11-data-privacy-security/examples/ex_06_reproducible_privacy_audit_pipeline.py
```

Recommended order: classification -> masking -> access control -> checklist -> prompt injection -> reproducibility.

## Next steps

1. Add a lightweight data-retention enforcement example.
1. Add a secrets-scanning example for config files.
1. Document operational limitations in `notes/README.md`.
