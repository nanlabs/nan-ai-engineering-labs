"""Baseline prompt quality scoring demo.

Run:
    python modules/09-generative-ai-prompt-engineering/examples/ex_01_prompt_baseline_quality.py
"""

from __future__ import annotations


def score_prompt(prompt: str) -> int:
    """Score prompt clarity using lightweight heuristics."""
    score = 0
    lowered = prompt.lower()
    if "context" in lowered:
        score += 1
    if "output" in lowered or "format" in lowered:
        score += 1
    if "constraint" in lowered or "do not" in lowered:
        score += 1
    if len(prompt.split()) >= 15:
        score += 1
    return score


def main() -> None:
    """Compare a weak baseline prompt and a stronger prompt."""
    baseline = "Summarize this text quickly."
    improved = (
        "Given the product update context, summarize the text in 3 bullet points, "
        "include one risk, and do not invent missing facts."
    )

    print(f"baseline_score={score_prompt(baseline)}/4")
    print(f"improved_score={score_prompt(improved)}/4")


if __name__ == "__main__":
    main()
