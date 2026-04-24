"""Run a deterministic mini evaluation pipeline for agent decisions.

Run:
    python trends-extras/agents/examples/ex_04_reproducible_agent_evaluation.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Task:
    """Simple task envelope with category and complexity."""

    name: str
    category: str
    complexity: int


def route_tool(task: Task) -> str:
    """Route task to a tool family using simple policy rules."""
    if task.category == "math":
        return "calculator"
    if task.category == "research":
        return "search"
    return "reasoner"


def evaluate(tasks: list[Task], seed: int) -> dict[str, object]:
    """Return deterministic routing metrics and a pseudo score."""
    random.seed(seed)

    routed = [route_tool(task) for task in tasks]
    tool_counts = {
        "calculator": routed.count("calculator"),
        "search": routed.count("search"),
        "reasoner": routed.count("reasoner"),
    }

    # Deterministic noise to emulate evaluation variability under fixed seed.
    noise = round(random.uniform(-0.02, 0.02), 4)

    base_score = sum(task.complexity for task in tasks) / (10 * len(tasks))
    final_score = round(max(0.0, min(1.0, base_score + noise)), 4)

    return {
        "tool_counts": tool_counts,
        "score": final_score,
        "decisions": routed,
    }


def main() -> None:
    """Evaluate reproducibility for agent routing decisions."""
    tasks = [
        Task("sum invoices", "math", 3),
        Task("find policy update", "research", 4),
        Task("draft response", "writing", 2),
        Task("check totals", "math", 3),
    ]

    run_a = evaluate(tasks, seed=42)
    run_b = evaluate(tasks, seed=42)
    run_c = evaluate(tasks, seed=43)

    print("Reproducible agent evaluation")
    print(f"run_a: {run_a}")
    print(f"run_b: {run_b}")
    print(f"run_c: {run_c}")
    print(f"same_result: {run_a == run_b}")
    print(f"different_seed_changes_output: {run_a != run_c}")


if __name__ == "__main__":
    main()
