# Practice 02 — Fine-Tuning and Prompting with LLMs

## 🎯 Objectives

- Wear transformers pre-trained
- Apply prompt engineering
- Fine-tuning of small models
- Evaluate outputs of LLMs

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Hugging Face Transformers

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

**✅ Solution - Text Generation:**

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

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Few-Shot Prompting

**Statement:**
Create prompts with 3 Examples for Classification:

```
Classify: "Great product!" → Positive
Classify: "Worst ever" → Negative
Classify: "It's fine" → Neutral
Classify: "[NEW_TEXT]" → ?
```

### Exercise 2.2: Chain-of-Thought

**Statement:**
Implement CoT prompting:

- Pide al Model razonar paso a paso
- Compare with direct response

### Exercise 2.3: Fine-Tuning with LoRA

**Statement:**
Fine-tune a small Model (DistilBERT) with LoRA:

1. Carga Model base
1. Agrega adaptadores LoRA
1. Train in custom dataset
1. Evaluate improvement

### Exercise 2.4: RAGpipeline

**Statement:**
Implement pipeline RAG simple:

1. Vectorize documents with embeddings
1. Retrieve top-k documents relevant
1. Build prompt with context
1. Generate response with LLM

### Exercise 2.5: Prompt Templating

**Statement:**
Create reusable template system:

```python
templates = {
    'summarize': "Summarize the following: {text}",
    'translate': "Translate to {lang}: {text}",
    'classify': "Classify sentiment: {text}"
}
```

______________________________________________________________________

## ✅ Checklist

- [ ] Wear Hugging Face pipelines
- [ ] Apply prompt engineering (zero-shot, few-shot, CoT)
- [ ] Fine-tuning of pre-trained Models
- [ ] Implement RAG basic
- [ ] Evaluate quality of outputs (BLEU, ROUGE)

______________________________________________________________________

## 📚 Resources

- [Hugging Face Course](https://huggingface.co/learn/nlp-course/chapter1/1)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
