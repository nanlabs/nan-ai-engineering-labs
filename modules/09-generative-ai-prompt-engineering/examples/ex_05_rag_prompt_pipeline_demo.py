"""Minimal RAG-style prompt pipeline demo.

Run:
    python modules/09-generative-ai-prompt-engineering/examples/ex_05_rag_prompt_pipeline_demo.py
"""

from __future__ import annotations


def retrieve_context(query: str, docs: list[str]) -> str:
    """Retrieve best document by keyword overlap."""
    query_terms = set(query.lower().split())
    best_doc = ""
    best_score = -1
    for doc in docs:
        doc_terms = set(doc.lower().split())
        score = len(query_terms & doc_terms)
        if score > best_score:
            best_doc, best_score = doc, score
    return best_doc


def build_rag_prompt(query: str, context: str) -> str:
    """Create prompt that injects retrieved context."""
    return (
        "You are an assistant that answers only from provided context.\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        "Answer in 2 concise sentences."
    )


def main() -> None:
    """Run retrieval and prompt construction."""
    docs = [
        "RAG improves factuality by retrieving relevant context before generation.",
        "Prompt templates improve consistency and readability.",
        "Evaluation loops reduce hallucination risk.",
    ]
    query = "How does RAG improve factuality"

    context = retrieve_context(query, docs)
    prompt = build_rag_prompt(query, context)

    print(f"selected_context={context}")
    print("prompt_preview=")
    print(prompt)


if __name__ == "__main__":
    main()
