"""Prompt iteration loop with feedback-based refinement.

Run:
    python modules/09-generative-ai-prompt-engineering/examples/ex_03_prompt_iteration_feedback_loop.py
"""

from __future__ import annotations


def quality_score(prompt: str) -> int:
    """Compute quality score from explicit criteria presence."""
    checks = ["context", "constraints", "format", "verify"]
    lowered = prompt.lower()
    return sum(1 for check in checks if check in lowered)


def refine_prompt(prompt: str, feedback: str) -> str:
    """Append feedback-guided instruction to current prompt."""
    return f"{prompt}\nRefinement note: {feedback}"


def main() -> None:
    """Run a tiny iteration process for prompt refinement."""
    prompt = "Summarize the report."
    feedback_rounds = [
        "Add context about audience and scope.",
        "Add constraints on factuality and length.",
        "Add output format and verification instruction.",
    ]

    for idx, feedback in enumerate(feedback_rounds, start=1):
        prompt = refine_prompt(prompt, feedback)
        print(f"round={idx} score={quality_score(prompt)}/4")


if __name__ == "__main__":
    main()
