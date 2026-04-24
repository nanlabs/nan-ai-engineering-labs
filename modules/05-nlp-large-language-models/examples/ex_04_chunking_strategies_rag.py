"""Compare chunking strategies for a tiny RAG-like retrieval step.

Run:
    python modules/05-nlp-large-language-models/examples/ex_04_chunking_strategies_rag.py
"""

from __future__ import annotations


def fixed_chunks(text: str, size: int) -> list[str]:
    """Split text by fixed character window."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def sentence_chunks(text: str) -> list[str]:
    """Split text by sentences."""
    return [chunk.strip() for chunk in text.split(".") if chunk.strip()]


def retrieve(chunks: list[str], query_terms: set[str]) -> list[tuple[int, str]]:
    """Score chunks by overlap with query terms."""
    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        tokens = set(chunk.lower().replace(",", " ").split())
        overlap = len(tokens & query_terms)
        scored.append((overlap, chunk))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def main() -> None:
    """Run fixed vs sentence chunking and show top retrieval result."""
    text = (
        "RAG combina recuperacion y generacion para responder con contexto. "
        "Chunking define como se parte el contenido antes de indexarlo. "
        "Chunks muy grandes pierden precision y chunks muy pequenos pierden contexto."
    )
    query_terms = {"rag", "chunking", "contexto", "precision"}

    fixed = fixed_chunks(text, size=70)
    sentence = sentence_chunks(text)

    fixed_top = retrieve(fixed, query_terms)[0]
    sentence_top = retrieve(sentence, query_terms)[0]

    print(f"fixed_chunks={len(fixed)} | top_score={fixed_top[0]} | top_chunk={fixed_top[1]}")
    print(f"sentence_chunks={len(sentence)} | top_score={sentence_top[0]} | top_chunk={sentence_top[1]}")


if __name__ == "__main__":
    main()
