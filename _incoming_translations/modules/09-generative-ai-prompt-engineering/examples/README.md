# Examples — Generative AI & Prompt Engineering

## Example 1 — Baseline prompt

Design an initial prompt for a summarization task.

## Example 2 — Structured prompt

Add an output format and explicit constraints.

## Example 3 — Guided iteration

Improve the prompt with feedback about errors.

## Example 4 — Basic guardrails

Include validations for safe and useful responses.

## Rules

- Record the prompt version and the obtained result.
- Compare quality across iterations.

## Available examples

### Executable scripts (phase-2 continuation)

1. `ex_01_prompt_baseline_quality.py`

   - Scores weak vs improved prompts with simple quality heuristics.
   - Expected output: improved prompt should score higher.

1. `ex_02_structured_prompt_constraints.py`

   - Builds a structured prompt with task, context, constraints, and output format.
   - Expected output: deterministic template-ready prompt text.

1. `ex_03_prompt_iteration_feedback_loop.py`

   - Runs iterative prompt refinement with feedback rounds.
   - Expected output: quality score increases across iterations.

1. `ex_04_guardrails_input_output.py`

   - Applies input/output guardrails for sensitive terms.
   - Expected output: safe prompt allowed, unsafe prompt blocked.

1. `ex_05_rag_prompt_pipeline_demo.py`

   - Demonstrates retrieval plus context-injected prompt building.
   - Expected output: selected context and preview of RAG prompt.

1. `ex_06_reproducible_prompt_evaluation.py`

   - Runs deterministic prompt evaluation using a fixed seed.
   - Expected output: repeated runs produce identical score values.

## How to use these examples

```bash
python modules/09-generative-ai-prompt-engineering/examples/ex_01_prompt_baseline_quality.py
python modules/09-generative-ai-prompt-engineering/examples/ex_02_structured_prompt_constraints.py
python modules/09-generative-ai-prompt-engineering/examples/ex_03_prompt_iteration_feedback_loop.py
python modules/09-generative-ai-prompt-engineering/examples/ex_04_guardrails_input_output.py
python modules/09-generative-ai-prompt-engineering/examples/ex_05_rag_prompt_pipeline_demo.py
python modules/09-generative-ai-prompt-engineering/examples/ex_06_reproducible_prompt_evaluation.py
```

Recommended order: baseline -> structure -> iteration -> guardrails -> RAG -> reproducibility.

## Next steps

1. Add a lightweight prompt regression test harness.
1. Add side-by-side prompt variants for cost/latency trade-off analysis.
1. Document practical prompt anti-patterns in `notes/README.md`.
