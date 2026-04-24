"""Minimal text preprocessing baseline for NLP tasks.

Run:
    python modules/05-nlp-large-language-models/examples/ex_01_text_preprocessing_baseline.py
"""

from __future__ import annotations

import re

STOPWORDS = {"the", "a", "an", "in", "on", "is", "of", "and", "that"}


def normalize_text(text: str) -> str:
    """Lowercase text and remove punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Split text into tokens and remove stopwords."""
    tokens = normalize_text(text).split(" ")
    return [token for token in tokens if token and token not in STOPWORDS]


def main() -> None:
    """Show deterministic preprocessing output for three short texts."""
    corpus = [
        "User experience in AI is key.",
        "A robust model improves response quality.",
        "In production, monitoring drift prevents surprises.",
    ]

    for index, sentence in enumerate(corpus, start=1):
        print(f"Text {index}: {sentence}")
        print(f"Tokens: {tokenize(sentence)}")
        print()


if __name__ == "__main__":
    main()
