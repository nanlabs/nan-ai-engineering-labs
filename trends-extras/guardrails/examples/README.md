# Guardrails — Safety and Control for LLMs

## Available examples

- `ex_01_input_validation.py`

  - Blocks prompt-injection and SQL-injection style inputs.
  - Expected output: clear allow/deny decisions with reasons.

- `ex_02_output_filtering.py`

  - Applies output filtering for PII redaction, toxicity, and policy checks.
  - Expected output: safe output or blocked response with issue list.

- `ex_03_nemo_guardrails_demo.py`

  - Conceptual NeMo Guardrails flow with input/output rails.
  - Expected output: stage-aware blocking (`input` or `output`) when a rail is triggered.

- `ex_04_reproducible_guardrail_audit.py`

  - Deterministic guardrail audit over a small test set.
  - Expected output: repeated runs produce `same_result=True`.

## How to use these examples

```bash
python trends-extras/guardrails/examples/ex_01_input_validation.py
python trends-extras/guardrails/examples/ex_02_output_filtering.py
python trends-extras/guardrails/examples/ex_03_nemo_guardrails_demo.py
python trends-extras/guardrails/examples/ex_04_reproducible_guardrail_audit.py
```

Recommended order: validation (`01`) -> filtering (`02`) -> orchestration (`03`)
-> reproducible audit (`04`).

## Next steps

1. Add a threshold-tuning example for false-positive/false-negative trade-offs.
1. Add a policy versioning example with changelog-driven tests.
1. Convert one script into a guided practice in `practices/`.
