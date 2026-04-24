"""Structured prompt template with explicit constraints.

Run:
    python modules/09-generative-ai-prompt-engineering/examples/ex_02_structured_prompt_constraints.py
"""

from __future__ import annotations


def build_prompt(task: str, context: str, constraints: list[str]) -> str:
    """Build a deterministic structured prompt."""
    constraint_block = "\n".join(f"- {item}" for item in constraints)
    return (
        f"Task:\n{task}\n\n"
        f"Context:\n{context}\n\n"
        f"Constraints:\n{constraint_block}\n\n"
        "Output format:\nJSON with keys summary, risks, action_items"
    )


def main() -> None:
    """Print an example structured prompt."""
    prompt = build_prompt(
        task="Summarize onboarding feedback.",
        context="Feedback comes from 120 users over the last 2 weeks.",
        constraints=[
            "Use max 80 words",
            "Do not mention user names",
            "Include one action item",
        ],
    )
    print(prompt)


if __name__ == "__main__":
    main()
