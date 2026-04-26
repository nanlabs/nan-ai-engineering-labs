"""Sentence similarity baseline with bag-of-words vectors.

Run:
    python modules/05-nlp-large-language-models/examples/ex_02_embeddings_similarity_baseline.py
"""

from __future__ import annotations

import math


def tokenize(text: str) -> list[str]:
    """Tokenize a lowercased sentence by spaces."""
    return text.lower().split()


def build_vocabulary(sentences: list[str]) -> list[str]:
    """Create a sorted vocabulary from all tokens."""
    vocab = {token for sentence in sentences for token in tokenize(sentence)}
    return sorted(vocab)


def vectorize(sentence: str, vocabulary: list[str]) -> list[float]:
    """Convert sentence into term-frequency vector."""
    tokens = tokenize(sentence)
    return [float(tokens.count(term)) for term in vocabulary]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    """Vectorize short sentences and print pairwise similarity scores."""
    sentences = [
        "ai engineering workflows for production",
        "mlops workflows and production monitoring",
        "cocina de recetas italians tradicionales",
    ]

    vocabulary = build_vocabulary(sentences)
    vectors = [vectorize(sentence, vocabulary) for sentence in sentences]

    sim_0_1 = cosine_similarity(vectors[0], vectors[1])
    sim_0_2 = cosine_similarity(vectors[0], vectors[2])

    print("Sentence A:", sentences[0])
    print("Sentence B:", sentences[1])
    print("Sentence C:", sentences[2])
    print()
    print(f"Similarity A-B: {sim_0_1:.4f}")
    print(f"Similarity A-C: {sim_0_2:.4f}")


if __name__ == "__main__":
    main()
