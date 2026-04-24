"""Reproducible prompt evaluation run with fixed seed behavior.

Run:
    python modules/09-generative-ai-prompt-engineering/examples/ex_06_reproducible_prompt_evaluation.py
"""

from __future__ import annotations

import random


def evaluate_prompt(prompt: str, seed: int) -> float:
    """Return deterministic pseudo-score for demonstration."""
    random.seed(seed)
    base = 0.6 if "context" in prompt.lower() else 0.4
    noise = random.uniform(0.0, 0.2)
    return round(base + noise, 4)


def main() -> None:
    """Show deterministic output for repeated seeded runs."""
    prompt = "Use context and constraints to summarize with JSON output."

    score_a = evaluate_prompt(prompt, seed=11)
    score_b = evaluate_prompt(prompt, seed=11)

    print(f"score_a={score_a}")
    print(f"score_b={score_b}")
    print(f"same_result={score_a == score_b}")


if __name__ == "__main__":
    main()
