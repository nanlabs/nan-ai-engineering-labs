# Ejemplo 01 — RAG (Retrieval-Augmented Generation) Básico

## Contexto

RAG combina búsqueda de documentos con generación de LLMs para responder preguntas basándose en conocimiento específico (no en training data del LLM).

## Objective

Construir sistema Q&A sobre documentación interna usando embeddings + LLM.

______________________________________________________________________

## 🚀 Setup

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Simulación de embeddings (en producción: usar OpenAI/HuggingFace)
def mock_embed(text):
    """Simula embedding vectorial"""
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(768)  # Dimensión típica

def mock_llm_generate(prompt):
    """Simula respuesta de LLM"""
    return f"[SIMULATED ANSWER based on: {prompt[:100]}...]"
```

______________________________________________________________________

## 📚 Crear knowledge base

```python
# Documentos de ejemplo (empresa ficticia)
documents = [
    {
        "id": "doc1",
        "content": "Nuestra política de vacaciones otorga 20 días hábiles al año para empleados con más de 1 año de antigüedad.",
        "metadata": {"category": "HR", "last_updated": "2024-01-15"}
    },
    {
        "id": "doc2",
        "content": "Para solicitar reembolso de gastos, completar formulario F-301 y adjuntar recibos. Plazo máximo: 30 días.",
        "metadata": {"category": "Finance", "last_updated": "2023-12-01"}
    },
    {
        "id": "doc3",
        "content": "El código de vestimenta es business casual. Se permite vestimenta informal los viernes.",
        "metadata": {"category": "HR", "last_updated": "2023-11-20"}
    },
    {
        "id": "doc4",
        "content": "Horario de trabajo flexible: entrada entre 8AM-10AM, salida 8 horas después. Núcleo obligatorio: 10AM-3PM.",
        "metadata": {"category": "HR", "last_updated": "2024-02-01"}
    },
    {
        "id": "doc5",
        "content": "Acceso VPN: usar Cisco AnyConnect con credenciales corporativas. Soporte: ext. 5555.",
        "metadata": {"category": "IT", "last_updated": "2024-01-20"}
    }
]

# Generar embeddings
for doc in documents:
    doc["embedding"] = mock_embed(doc["content"])

print(f"Knowledge base: {len(documents)} documentos indexados")
```

______________________________________________________________________

## 🔍 Función de retrieval

```python
def retrieve_relevant_docs(query, documents, top_k=3):
    """
    Encuentra documentos más relevantes usando similitud coseno
    """
    # Embed query
    query_embedding = mock_embed(query)

    # Calcular similitudes
    similarities = []
    for doc in documents:
        sim = cosine_similarity([query_embedding], [doc["embedding"]])[0][0]
        similarities.append((doc, sim))

    # Ordenar por similitud
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Retornar top K
    return [doc for doc, sim in similarities[:top_k]]

# Ejemplo: buscar documentos sobre vacaciones
query = "¿Cuántos días de vacaciones tengo?"
relevant_docs = retrieve_relevant_docs(query, documents, top_k=2)

print(f"\nQuery: {query}")
print(f"Documentos relevantes:\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"{i}. {doc['content'][:80]}...")
```

**Salida:**

```
Query: ¿Cuántos días de vacaciones tengo?
Documentos relevantes:

1. Nuestra política de vacaciones otorga 20 días hábiles al año para empleados con...
2. Horario de trabajo flexible: entrada entre 8AM-10AM, salida 8 horas después...
```

______________________________________________________________________

## 🤖 RAG Pipeline completo

```python
def rag_answer(question, documents, top_k=3):
    """
    Pipeline RAG:
    1. Retrieve documentos relevantes
    2. Construir prompt con contexto
    3. Generate respuesta con LLM
    """
    # Paso 1: Retrieve
    relevant_docs = retrieve_relevant_docs(question, documents, top_k)

    # Paso 2: Construir contexto
    context = "\n\n".join([f"Documento {i+1}: {doc['content']}"
                           for i, doc in enumerate(relevant_docs)])

    # Paso 3: Prompt engineering
    prompt = f"""
Responde la siguiente pregunta SOLO basándote en el contexto proporcionado.
Si la información no está en el contexto, di "No tengo información suficiente".

Contexto:
{context}

Pregunta: {question}

Respuesta:
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

# Preguntas de prueba
questions = [
    "¿Cuántos días de vacaciones tengo?",
    "¿Cómo solicito reembolso de gastos?",
    "¿Cuál es el horario de trabajo?",
    "¿Qué es la política de pets in office?"  # Fuera del knowledge base
]

print("\n=== RAG Q&A System ===\n")

for q in questions:
    result = rag_answer(q, documents, top_k=2)

    print(f"Pregunta: {q}")
    print(f"Respuesta: {result['answer']}")
    print(f"Fuentes: {', '.join([s['id'] for s in result['sources']])}")
    print("-" * 80)
```

______________________________________________________________________

## 📊 Evaluación

### Métricas de retrieval

```python
# Ground truth: preguntas con documentos relevantes conocidos
eval_set = [
    {"question": "¿Cuántos días de vacaciones?", "relevant_doc_ids": ["doc1"]},
    {"question": "¿Cómo pedir reembolso?", "relevant_doc_ids": ["doc2"]},
    {"question": "¿Horario de entrada?", "relevant_doc_ids": ["doc4"]},
]

def evaluate_retrieval(eval_set, documents, top_k=3):
    """
    Calcular Precision@K y Recall@K
    """
    precisions = []
    recalls = []

    for item in eval_set:
        query = item["question"]
        relevant_ids = set(item["relevant_doc_ids"])

        # Retrieve
        retrieved_docs = retrieve_relevant_docs(query, documents, top_k)
        retrieved_ids = set([doc["id"] for doc in retrieved_docs])

        # Métricas
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
print(f"\n=== Métricas de Retrieval ===\n")
print(f"Precision@2: {metrics['precision@k']:.2f}")
print(f"Recall@2: {metrics['recall@k']:.2f}")
```

______________________________________________________________________

## 💡 Mejoras para producción

### 1. Vector Database (Pinecone, Weaviate, ChromaDB)

```python
# Ejemplo conceptual con ChromaDB
# pip install chromadb

# import chromadb
#
# client = chromadb.Client()
# collection = client.create_collection("company_docs")
#
# # Indexar documentos
# collection.add(
#     documents=[doc["content"] for doc in documents],
#     metadatas=[doc["metadata"] for doc in documents],
#     ids=[doc["id"] for doc in documents]
# )
#
# # Búsqueda
# results = collection.query(
#     query_texts=["¿Cuántos días de vacaciones?"],
#     n_results=3
# )
```

### 2. Reranking

```python
def rerank_results(query, docs, llm_model):
    """
    Reordena resultados usando LLM para mejorarcalidad
    """
    # Prompt para reranking
    docs_text = "\n".join([f"{i}. {doc['content']}" for i, doc in enumerate(docs)])

    prompt = f"""
Ordena estos documentos de más a menos relevante para la pregunta.

Pregunta: {query}

Documentos:
{docs_text}

Responde solo con los números ordenados (ej: 2,0,1):
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

    # Combinar
    hybrid_scores = alpha * np.array(semantic_scores) + (1-alpha) * keyword_scores

    # Top docs
    top_indices = hybrid_scores.argsort()[-3:][::-1]
    return [documents[i] for i in top_indices]

hybrid_results = hybrid_retrieve("días de vacaciones", documents, alpha=0.7)
```

______________________________________________________________________

## 📝 Resumen

### ✅ RAG Pipeline

```
Usuario pregunta
  ↓
Embed query
  ↓
Vector search (top K documentos)
  ↓
Construir prompt con contexto
  ↓
LLM genera respuesta
  ↓
Retornar respuesta + fuentes
```

### 🎯 Ventajas de RAG

1. **Conocimiento actualizado:** No requiere reentrenar LLM
1. **Reduce hallucinations:** LLM responde basándose en documentos reales
1. **Transparencia:** Cita fuentes
1. **Privacidad:** Documentos privados no van al training data

### 💡 Mejores prácticas

- ✅ Chunking inteligente (documentos largos → pedazos de 512 tokens)
- ✅ Metadata filtering (filtrar por categoría, fecha antes de buscar)
- ✅ Hybrid search (semántica + keywords)
- ✅ Reranking con cross-encoder
- ✅ Prompt engineering (instruir al LLM a citar fuentes)
- ✅ Evaluación continua (precision, recall, answer quality)

### 🚫 Errores comunes

- ❌ Chunks muy grandes → contexto excede límite tokens LLM
- ❌ No considerar metadata → resultados irrelevantes
- ❌ Solo semantic search → falla en queries exact match
- ❌ No validar calidad de respuestas → hallucinations pasan

### 📌 Checklist RAG

- ✅ Chunking strategy definida
- ✅ Embedding model seleccionado
- ✅ Vector DB escalable
- ✅ Top-K tuneado (3-5 típicamente)
- ✅ Prompt engineering para citar fuentes
- ✅ Evaluación de retrieval (precision@k)
- ✅ Evaluación de generación (RAGAS, humana)
- ✅ Monitoring en producción
