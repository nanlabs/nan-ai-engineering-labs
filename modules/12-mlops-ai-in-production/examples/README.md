# Examples — MLOps & AI in Production

## Example 1 — Reproducible pipeline

Version data, training runs, and artifacts.

## Example 2 — Baseline serving

Expose a model through an API or a batch process.

## Example 3 — Initial monitoring

Record latency and quality metrics.

## Example 4 — Drift simulation

Detect degradation and trigger an alert.

## Rules

- Each example must state an operational objective.
- Document infrastructure assumptions.

## Available examples

### Executable scripts (phase-2 continuation)

1. `ex_01_reproducible_pipeline_manifest.py`

   - Builds a lightweight manifest for reproducible training runs.
   - Expected output: deterministic version/asset metadata.

1. `ex_02_batch_vs_realtime_serving.py`

   - Chooses serving mode from latency and volume constraints.
   - Expected output: realtime, batch, or hybrid decision.

1. `ex_03_latency_monitoring_baseline.py`

   - Summarizes latency metrics and raises a simple alert.
   - Expected output: average/min/max latency and alert flag.

1. `ex_04_drift_detection_demo.py`

   - Detects simple feature drift from reference vs production means.
   - Expected output: drift score and alert status.

1. `ex_05_canary_rollout_decision.py`

   - Decides whether a canary model should be promoted.
   - Expected output: boolean rollout decision.

1. `ex_06_reproducible_mlops_runbook.py`

   - Produces deterministic runbook metadata using a fixed seed.
   - Expected output: repeated runs with `same_result=True`.

## How to use these examples

```bash
python modules/12-mlops-ai-in-production/examples/ex_01_reproducible_pipeline_manifest.py
python modules/12-mlops-ai-in-production/examples/ex_02_batch_vs_realtime_serving.py
python modules/12-mlops-ai-in-production/examples/ex_03_latency_monitoring_baseline.py
python modules/12-mlops-ai-in-production/examples/ex_04_drift_detection_demo.py
python modules/12-mlops-ai-in-production/examples/ex_05_canary_rollout_decision.py
python modules/12-mlops-ai-in-production/examples/ex_06_reproducible_mlops_runbook.py
```

Recommended order: manifest -> serving -> latency -> drift -> canary -> reproducibility.

## Next steps

1. Add a lightweight incident escalation example.
1. Add an offline/online metric comparison example.
1. Record rollout lessons in `notes/README.md`.
