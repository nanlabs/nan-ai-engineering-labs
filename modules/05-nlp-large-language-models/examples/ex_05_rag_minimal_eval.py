"""Minimal retrieval + answer scoring loop for RAG-style evaluation.

Run:
    python modules/05-nlp-large-language-models/examples/ex_05_rag_minimal_eval.py
"""

from __future__ import annotations


def retrieve_context(query: str, docs: list[str]) -> str:
    """Retrieve best document by term overlap."""
    query_terms = set(query.lower().split())
    best_doc = ""
    best_score = -1
    for doc in docs:
        doc_terms = set(doc.lower().split())
        score = len(query_terms & doc_terms)
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc


def generate_answer(query: str, context: str) -> str:
    """Generate deterministic answer from selected context."""
    return f"Pregunta: {query} | Answer basada en: {context}"


def exact_match(expected: str, predicted: str) -> int:
    """Return 1 when expected phrase appears in predicted answer."""
    return int(expected.lower() in predicted.lower())


def main() -> None:
    """Run a tiny eval set with exact-match style scoring."""
    docs = [
        "RAG usa recuperacion de context para mejorar responses.",
        "Prompt engineering define formato, tono y restricciones.",
        "Embeddings permiten busqueda semantica de similarity.",
    ]
    eval_set = [
        ("Como mejora RAG las responses", "recuperacion de context"),
        ("Para que sirven los embeddings", "busqueda semantica"),
    ]

    scores: list[int] = []
    for query, expected_phrase in eval_set:
        context = retrieve_context(query, docs)
        answer = generate_answer(query, context)
        score = exact_match(expected_phrase, answer)
        scores.append(score)
        print(f"query={query}")
        print(f"score={score} | answer={answer}")
        print()

    final_score = sum(scores) / len(scores)
    print(f"exact_match_avg={final_score:.4f}")


if __name__ == "__main__":
    main()
