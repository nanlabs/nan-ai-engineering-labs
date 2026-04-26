# Example 01 — RAG (Retrieval-Augmented Generation) Basic

## Context

RAG combines document search with LLM generation to answer questions based on specific knowledge (not LLM training data).

## Objective

Build system Q&A about internal documentation using embeddings + LLM.

______________________________________________________________________

## 🚀 Setup

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Simulation de embeddings (en production: use OpenAI/HuggingFace)
def mock_embed(text):
    """Simula embedding vectorial"""
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(768)  # Dimension typical

def mock_llm_generate(prompt):
    """Simula response de LLM"""
    return f"[SIMULATED ANSWER based on: {prompt[:100]}...]"
```

______________________________________________________________________

## 📚 Create knowledge base

```python
# Documentos de example (empresa ficticia)
documents = [
    {
        "id": "doc1",
        "content": "Nuestra policy de vacaciones otorga 20 days skillful al año para empleados con más de 1 año de antigüedad.",
        "metadata": {"category": "HR", "last_updated": "2024-01-15"}
    },
    {
        "id": "doc2",
        "content": "Para solicitar reembolso de gastos, complete formulario F-301 y adjuntar recibos. Plazo maximum: 30 days.",
        "metadata": {"category": "Finance", "last_updated": "2023-12-01"}
    },
    {
        "id": "doc3",
        "content": "El code de vestimenta es business casual. Se permite vestimenta informal los viernes.",
        "metadata": {"category": "HR", "last_updated": "2023-11-20"}
    },
    {
        "id": "doc4",
        "content": "Horario de trabajo flexible: input entre 8AM-10AM, output 8 horas after. Core obligatorio: 10AM-3PM.",
        "metadata": {"category": "HR", "last_updated": "2024-02-01"}
    },
    {
        "id": "doc5",
        "content": "Acceso VPN: use Cisco AnyConnect con credenciales corporativas. Soporte: ext. 5555.",
        "metadata": {"category": "IT", "last_updated": "2024-01-20"}
    }
]

# Generate embeddings
for doc in documents:
    doc["embedding"] = mock_embed(doc["content"])

print(f"Knowledge base: {len(documents)} documents indexados")
```

______________________________________________________________________

## 🔍 Retrieval function

```python
def retrieve_relevant_docs(query, documents, top_k=3):
    """
    Encuentra documents más relevant using similarity cosine
    """
    # Embed query
    query_embedding = mock_embed(query)

    # Calculate similitudes
    similarities = []
    for doc in documents:
        sim = cosine_similarity([query_embedding], [doc["embedding"]])[0][0]
        similarities.append((doc, sim))

    # Ordenar por similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Retornar top K
    return [doc for doc, sim in similarities[:top_k]]

# Example: buscar documents about vacaciones
query = "¿How many days de vacaciones tengo?"
relevant_docs = retrieve_relevant_docs(query, documents, top_k=2)

print(f"\nQuery: {query}")
print(f"Documentos relevant:\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"{i}. {doc['content'][:80]}...")
```

**Output:**

```
Query: ¿How many days de vacaciones tengo?
Documentos relevant:

1. Nuestra policy de vacaciones otorga 20 days skillful al año para empleados con...
2. Horario de trabajo flexible: input entre 8AM-10AM, output 8 horas after...
```

______________________________________________________________________

## 🤖 RAG Pipeline complete

```python
def rag_answer(question, documents, top_k=3):
    """
    Pipeline RAG:
    1. Retrieve documents relevant
    2. Build prompt con context
    3. Generate response con LLM
    """
    # Paso 1: Retrieve
    relevant_docs = retrieve_relevant_docs(question, documents, top_k)

    # Paso 2: Build context
    context = "\n\n".join([f"I document {i+1}: {doc['content']}"
                           for i, doc in enumerate(relevant_docs)])

    # Paso 3: Prompt engineering
    prompt = f"""
Respond, Response, Responds, Responded, Responder la next pregunta SOLO basing you en el context proporcionado.
Si la information no this en el context, di "No tengo information suficiente".

Contexto:
{context}

Pregunta: {question}

Answer:
"""

    # Generate (simulated)
    answer = mock_llm_generate(prompt)

    # Metadata de sources
    sources = [{"id": doc["id"], "category": doc["metadata"]["category"]}
               for doc in relevant_docs]

    return {
        "answer": answer,
        "sources": sources,
        "relevant_docs": relevant_docs
    }

# Preguntas de testing
questions = [
    "¿How many days de vacaciones tengo?",
    "¿As solicito reembolso de gastos?",
    "¿Which es el horario de trabajo?",
    "¿Qué es la policy de pets in office?"  # Fuera del knowledge base
]

print("\n=== RAG Q&A System ===\n")

for q in questions:
    result = rag_answer(q, documents, top_k=2)

    print(f"Pregunta: {q}")
    print(f"Answer: {result['answer']}")
    print(f"Fuentes: {', '.join([s['id'] for s in result['sources']])}")
    print("-" * 80)
```

______________________________________________________________________

## 📊 Evaluation

### Retrieval Metrics

```python
# Ground truth: preguntas con documents relevant conocidos
eval_set = [
    {"question": "¿How many days de vacaciones?", "relevant_doc_ids": ["doc1"]},
    {"question": "¿As pedir reembolso?", "relevant_doc_ids": ["doc2"]},
    {"question": "¿Horario de input?", "relevant_doc_ids": ["doc4"]},
]

def evaluate_retrieval(eval_set, documents, top_k=3):
    """
    Calculate Precision@K y Recall@K
    """
    precisions = []
    recalls = []

    for item in eval_set:
        query = item["question"]
        relevant_ids = set(item["relevant_doc_ids"])

        # Retrieve
        retrieved_docs = retrieve_relevant_docs(query, documents, top_k)
        retrieved_ids = set([doc["id"] for doc in retrieved_docs])

        # Metrics
        tp = len(relevant_ids & retrieved_ids)
        precision = tp / len(retrieved_ids) if retrieved_ids else 0
        recall = tp / len(relevant_ids) if relevant_ids else 0

        precisions.append(precision)
        recalls.append(recall)

    return {
        "precision@k": np.mean(precisions),
        "recall@k": np.mean(recalls)
    }

metrics = evaluate_retrieval(eval_set, documents, top_k=2)
print(f"\n=== Metrics de Retrieval ===\n")
print(f"Precision@2: {metrics['precision@k']:.2f}")
print(f"Recall@2: {metrics['recall@k']:.2f}")
```

______________________________________________________________________

## 💡 Production improvements

### 1. Vector Database (Pinecone, Weaviate, ChromaDB)

```python
# Example conceptual con ChromaDB
# pip install chromadb

# import chromadb
#
# client = chromadb.Client()
# collection = client.create_collection("company_docs")
#
# # Indexar documents
# collection.add(
#     documents=[doc["content"] for doc in documents],
#     metadatas=[doc["metadata"] for doc in documents],
#     ids=[doc["id"] for doc in documents]
# )
#
# # Search
# results = collection.query(
#     query_texts=["¿How many days de vacaciones?"],
#     n_results=3
# )
```

### 2. Reranking

```python
def rerank_results(query, docs, llm_model):
    """
    Reordena results using LLM para mejorarcalidad
    """
    # Prompt para reranking
    docs_text = "\n".join([f"{i}. {doc['content']}" for i, doc in enumerate(docs)])

    prompt = f"""
Ordena estos documents de más a less relevante para la pregunta.

Pregunta: {query}

Documentos:
{docs_text}

Respond, Response, Responds, Responded, Responder solo con los numbers ordenados (ej: 2,0,1):
"""

    # LLM reranking (simulated)
    # ranking = llm_model.generate(prompt)
    return docs  # Placeholder

reranked = rerank_results("¿Vacaciones?", relevant_docs, None)
```

### 3. Hybrid Search (keyword + semantic)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# BM25 o TF-IDF para keyword search
tfidf = TfidfVectorizer()
doc_texts = [doc["content"] for doc in documents]
tfidf_matrix = tfidf.fit_transform(doc_texts)

def hybrid_retrieve(query, documents, alpha=0.5):
    """
    alpha: peso de semantic search (1-alpha = keyword search)
    """
    # Semantic similarity
    semantic_scores = [cosine_similarity([mock_embed(query)], [doc["embedding"]])[0][0]
                       for doc in documents]

    # Keyword similarity
    query_vec = tfidf.transform([query])
    keyword_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Combiner
    hybrid_scores = alpha * np.array(semantic_scores) + (1-alpha) * keyword_scores

    # Top docs
    top_indices = hybrid_scores.argsort()[-3:][::-1]
    return [documents[i] for i in top_indices]

hybrid_results = hybrid_retrieve("days de vacaciones", documents, alpha=0.7)
```

______________________________________________________________________

## 📝 Summary

### ✅ RAG Pipeline

```
User pregunta
  ↓
Embed query
  ↓
Vector search (top K documents)
  ↓
Build prompt con context
  ↓
LLM genera response
  ↓
Retornar response + fuentes
```

### 🎯 Advantages of RAG

1. **Updated knowledge:** Does not require retraining LLM
1. **Reduce hallucinations:** LLM responds based on real documents
1. **Transparency:** Cite sources
1. **Privacy:** Private documents do not go to training data

### 💡 Best Practices

- ✅ Smart Chunking (long documents → chunks of 512 tokens)
- ✅ Metadata filtering (filter by category, date before searching)
- ✅ Hybrid search (semantics + keywords)
- ✅ Reranking with cross-encoder
- ✅ Prompt engineering (instruct the LLM to cite sources)
- ✅ Evaluation continua (precision, recall, answer quality)

### 🚫 Errors common

- ❌ Very large chunks → context exceeds LLM token limit
- ❌ Do not consider metadata → Irrelevant results
- ❌ Only semantic search → queries exact match fails
- ❌ Do not validate quality of responses → hallucinations pass

### 📌 Checklist RAG

- ✅ Chunking strategy defined
- ✅ embedding model selected
- ✅ Scalable DB Vector
- ✅ Top-K tuned (3-5 typically)
- ✅ Prompt engineering to cite sources
- ✅ Retrieval evaluation (precision@k)
- ✅ Generation evaluation (RAGAS, human)
- ✅ Monitoring in production
