# AI Agents — Autonomous Intelligent Systems

## Available examples

- `ex_01_simple_agent_langchain.py`

  - LangChain agent with calculator and search tools.
  - Expected output: a visible reasoning/action loop with tool calls.
  - Note: requires external dependencies and API keys.

- `ex_02_react_agent.py`

  - From-scratch ReAct loop simulation with local tools.
  - Expected output: iterative `Thought -> Action -> Observation -> Answer` steps.

- `ex_03_multi_agent_system.py`

  - Multi-agent coordination demo with role specialization.
  - Expected output: coordinated messages and a synthesized final report.

- `ex_04_reproducible_agent_evaluation.py`

  - Deterministic mini evaluation pipeline for agent decisions.
  - Expected output: repeated runs produce `same_result=True`.

## How to use these examples

```bash
python trends-extras/agents/examples/ex_01_simple_agent_langchain.py
python trends-extras/agents/examples/ex_02_react_agent.py
python trends-extras/agents/examples/ex_03_multi_agent_system.py
python trends-extras/agents/examples/ex_04_reproducible_agent_evaluation.py
```

Recommended order: `02 -> 03 -> 04` first, then `01` after installing LangChain
and configuring provider credentials.

## Next steps

1. Add a production-style retries/timeouts example for tool failures.
1. Add a trace schema example for observability integration.
1. Convert one script into a guided practice in `practices/`.
