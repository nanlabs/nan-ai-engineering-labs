"""Minimal text preprocessing baseline for NLP tasks.

Run:
    python modules/05-nlp-large-language-models/examples/ex_01_text_preprocessing_baseline.py
"""

from __future__ import annotations

import re

STOPWORDS = {"el", "la", "de", "y", "en", "es", "un", "una", "que"}


def normalize_text(text: str) -> str:
    """Lowercase text and remove punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9áéíóúñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Split text into tokens and remove stopwords."""
    tokens = normalize_text(text).split(" ")
    return [token for token in tokens if token and token not in STOPWORDS]


def main() -> None:
    """Show deterministic preprocessing output for three short texts."""
    corpus = [
        "La experiencia de usuario en AI es clave.",
        "Un modelo robusto mejora la calidad de respuesta.",
        "En producción, monitorear deriva evita sorpresas.",
    ]

    for index, sentence in enumerate(corpus, start=1):
        print(f"Text {index}: {sentence}")
        print(f"Tokens: {tokenize(sentence)}")
        print()


if __name__ == "__main__":
    main()
