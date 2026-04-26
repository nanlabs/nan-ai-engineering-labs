"""Evaluate prompt variants with a simple rubric.

Run:
    python modules/05-nlp-large-language-models/examples/ex_03_prompt_variants_quality.py
"""

from __future__ import annotations


def score_prompt(prompt: str) -> int:
    """Score prompt quality using simple heuristic checks."""
    score = 0
    if "context" in prompt.lower():
        score += 1
    if "formato" in prompt.lower() or "json" in prompt.lower():
        score += 1
    if "restricciones" in prompt.lower() or "limites" in prompt.lower():
        score += 1
    if len(prompt.split()) >= 18:
        score += 1
    return score


def main() -> None:
    """Compare weak vs strong prompts for the same task."""
    prompts = {
        "weak": "Resume este text.",
        "medium": "Con este context de production, resume el text en 3 bullets.",
        "strong": (
            "Con el context de onboarding de users, resume el text en formato JSON, "
            "incluye 3 bullets, una accion sugerida y restricciones de no inventar data."
        ),
    }

    for name, prompt in prompts.items():
        print(f"{name}: score={score_prompt(prompt)}/4")
        print(f"prompt={prompt}")
        print()


if __name__ == "__main__":
    main()
