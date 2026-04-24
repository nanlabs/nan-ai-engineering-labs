# Práctica 02 — Fine-Tuning y Prompting con LLMs

## 🎯 Objetivos

- Usar transformers pre-entrenados
- Aplicar prompt engineering
- Fine-tuning de modelos pequeños
- Evaluar outputs de LLMs

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Hugging Face Transformers

```python
from transformers import pipeline

# Sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, not great."
]

for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']} ({result['score']:.4f})\\n")
```

**✅ Solución - Text Generation:**

```python
# Text generation
generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial intelligence will"
outputs = generator(prompt, max_length=50, num_return_sequences=3)

print(f"Prompt: {prompt}\\n")
for i, output in enumerate(outputs, 1):
    print(f"{i}. {output['generated_text']}\\n")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Few-Shot Prompting

**Enunciado:**
Crea prompts con 3 ejemplos para clasificación:

```
Classify: "Great product!" → Positive
Classify: "Worst ever" → Negative
Classify: "It's fine" → Neutral
Classify: "[NEW_TEXT]" → ?
```

### Ejercicio 2.2: Chain-of-Thought

**Enunciado:**
Implementa CoT prompting:

- Pide al modelo razonar paso a paso
- Compara con respuesta directa

### Ejercicio 2.3: Fine-Tuning con LoRA

**Enunciado:**
Fine-tune un modelo pequeño (DistilBERT) con LoRA:

1. Carga modelo base
1. Agrega adaptadores LoRA
1. Entrena en dataset custom
1. Evalúa mejora

### Ejercicio 2.4: RAGpipeline

**Enunciado:**
Implementa pipeline RAG simple:

1. Vectoriza documentos con embeddings
1. Retrieve top-k documentos relevantes
1. Construye prompt con contexto
1. Genera respuesta con LLM

### Ejercicio 2.5: Prompt Templating

**Enunciado:**
Crea sistema de templates reutilizables:

```python
templates = {
    'summarize': "Summarize the following: {text}",
    'translate': "Translate to {lang}: {text}",
    'classify': "Classify sentiment: {text}"
}
```

______________________________________________________________________

## ✅ Checklist

- [ ] Usar pipelines de Hugging Face
- [ ] Aplicar prompt engineering (zero-shot, few-shot, CoT)
- [ ] Fine-tuning de modelos pre-entrenados
- [ ] Implementar RAG básico
- [ ] Evaluar calidad de outputs (BLEU, ROUGE)

______________________________________________________________________

## 📚 Recursos

- [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1/1)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
