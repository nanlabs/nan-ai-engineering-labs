# Theory — NLP & Large Language Models

## Why this module matters

El 80% de los Data empresariales está en formato texto. NLP te permite extraer valor de documentos, emails, chats, reviews y más. Los Large Language Models (LLMs) han democratizado capacidades antes imposibles: generación, traducción, resumen y razonamiento sobre texto.

______________________________________________________________________

## 1. ¿Qué es NLP?

**Natural Language Processing (NLP)** permite que sistemas computacionales **entiendan, procesen y generen** lenguaje humano.

### Desafíos del lenguaje humano

- **Ambigüedad:** "banco" puede ser institución financiera o asiento.
- **Contexto:** el significado depende de lo anterior/posterior.
- **Sarcasmo e ironía:** difíciles de detectar.
- **Variaciones lingüísticas:** dialectos, slang, Errors ortográficos.

### Aplicaciones reales

- Chatbots y asistentes virtuales.
- Analysis de Sentiment en redes sociales.
- Traducción automática.
- Resumen de documentos.
- Búsqueda semántica.
- Extracción de información de contratos/facturas.

📹 **Videos recomendados:**

1. [NLP Introduction - Stanford](https://www.youtube.com/watch?v=8rXD5-xhemo) - 20 min
1. [NLP Zero to Hero - TensorFlow](https://www.youtube.com/watch?v=fNxaJsNG3-s&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S) - serie completa

______________________________________________________________________

## 2. Pipeline clásico de NLP

### Paso 1: Recolección de texto

- Web scraping, APIs (Twitter, Reddit), datasets públicos.

### Paso 2: Cleaning

**Operaciones comunes:**

- Convertir a lowercase.
- Remover puntuación (según caso).
- Remover stopwords (palabras muy frecuentes: "el", "la", "de").
- Normalization de caracteres (acentos, unicode).

**Cuidado:** Siempre validar si la Cleaning ayuda a tu tarea específica.

### Paso 3: Tokenization

Dividir texto en unidades (tokens): palabras, subpalabras o caracteres.

**Example:**

```
"Machine Learning es genial."
→ ["Machine", "Learning", "es", "genial", "."]
```

### Paso 4: Representación numérica

Los Models necesitan números, no texto.

**Opciones:**

- **Bag of Words (BoW):** contar frecuencia de cada palabra (ignora orden).
- **TF-IDF:** penaliza palabras muy frecuentes, premia palabras discriminativas.
- **embeddings:** vectores densos que capturan semántica (Word2Vec, GloVe, FastText).

### Paso 5: Modelado

- Clasificadores clásicos (Naive Bayes, Logistic Regression, Random Forest).
- neural networks (RNN, LSTM, Transformers).

### Paso 6: Evaluation

- Metrics según tarea (accuracy, f1, BLEU, ROUGE).

📹 **Videos recomendados:**

1. [Text Preprocessing - Krish Naik](https://www.youtube.com/watch?v=nxhCyeRR75Q) - 30 min
1. [Tokenization Explained - Hugging Face](https://www.youtube.com/watch?v=VFp38yj8h3A) - 15 min

📚 **Resources escritos:**

- [NLTK Book (Free)](https://www.nltk.org/book/)
- [spaCy Documentation](https://spacy.io/usage/spacy-101)

______________________________________________________________________

## 3. Representaciones de texto

### Bag of Words (BoW)

**Concept:** Representar documento como vector de frecuencias de palabras.

**Ventaja:** Simple, interpretable.
**Desventaja:** Ignora orden y contexto.

### TF-IDF (Term Frequency - Inverse Document Frequency)

**Fórmula:** `TF-IDF = (frecuencia en documento) × log(total docs / docs con término)`

**Ventaja:** Reduce peso de palabras muy comunes.
**Usage:** Búsqueda de información, Classification clásica.

### Word embeddings

**Concept:** Representar palabras como vectores densos en espacio continuo donde palabras similares están cerca.

**Models populares:**

- **Word2Vec:** aprende embeddings prediciendo palabras vecinas.
- **GloVe:** basado en co-ocurrencias globales.
- **FastText:** similar a Word2Vec pero maneja subpalabras (bueno para typos).

**Ventaja:** Captura relaciones semánticas.

- Example: `king - man + woman ≈ queen`

### embeddings contextuales (Transformers)

Palabras tienen representaciones diferentes según contexto.

- **BERT, RoBERTa, GPT:** Models pre-entrenados que generan embeddings contextuales.

📹 **Videos recomendados:**

1. [Word2Vec Explained - StatQuest](https://www.youtube.com/watch?v=viZrOnJclY0) - 20 min
1. [Word Embeddings - Corey Schafer](https://www.youtube.com/watch?v=aHLslaWO-AQ) - 15 min

______________________________________________________________________

## 4. Tareas frecuentes en NLP

### Classification de texto

- Detectar spam, Sentiment (positivo/negativo), intención del usuario.
- **Metrics:** accuracy, Precision, recall, f1.

### Named Entity Recognition (NER)

Extraer entidades: personas, lugares, organizaciones, fechas.

**Example:**

```
"Elon Musk fundó Tesla en 2003."
→ [Elon Musk: PERSONA], [Tesla: ORGANIZACIÓN], [2003: FECHA]
```

### Búsqueda semántica

Encontrar documentos relevantes basándose en significado, no solo palabras exactas.

### Resumen automático

- **Extractivo:** seleccionar oraciones clave del original.
- **Abstractivo:** generar resumen con lenguaje propio (requiere LLM).

### Generación de texto

Producir texto coherente: chatbots, auto-completado, creación de Content.

📹 **Videos recomendados:**

1. [Text Classification - Krish Naik](https://www.youtube.com/watch?v=VtRLrQ3Ev-U) - 45 min
1. [Named Entity Recognition - freeCodeCamp](https://www.youtube.com/watch?v=q-LE7w55ZGA) - 30 min

______________________________________________________________________

## 5. Arquitecturas de neural networks para NLP

### RNN (Recurrent Neural Networks)

Procesan secuencias de izquierda a derecha, manteniendo "memoria" del contexto anterior.

**Limitación:** Dificultad con secuencias largas (vanishing gradients).

### LSTM (Long Short-Term Memory)

Variante de RNN que mitiga vanishing gradients con compuertas (gates).

### Transformers

**Revolución en NLP (2017).** Basados en mecanismo de **attention**.

**Ventajas:**

- Procesan secuencias en paralelo (no secuencialmente como RNN).
- Capturan dependencias a largo plazo.
- Escalables a Models enormes.

**Arquitecturas clave:**

- **BERT:** encoder (bueno para Classification, NER).
- **GPT:** decoder (bueno para generación).
- **T5, BART:** encoder-decoder (traducción, resumen).

📹 **Videos recomendados (FUNDAMENTALES):**

1. [Attention is All You Need - Illustrated](https://www.youtube.com/watch?v=4Bdc55j80l8) - 15 min
1. [BERT Explained - CodeEmporium](https://www.youtube.com/watch?v=xI0HHN5XKDo) - 20 min
1. [Transformers from Scratch - Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) - 2 horas

📚 **Resources escritos:**

- [The Illustrated Transformer (Blog)](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Course (Free)](https://huggingface.co/course/chapter1)

______________________________________________________________________

## 6. Large Language Models (LLMs)

### ¿Qué son?

Models de lenguaje **masivos** (millones/billones de parámetros) entrenados sobre corpus textuales enormes (internet, libros, código).

**Examples:**

- GPT-4 (OpenAI)
- Claude (Anthropic)
- Llama (Meta)
- Gemini (Google)

### Capacidades emergentes

A medida que los Models crecen, emergen habilidades no explícitamente entrenadas:

- Razonamiento multi-paso.
- Traducción entre idiomas que el Model nunca vio juntos.
- Generación de código.
- Resolución de Problems matemáticos.

### Limitaciones críticas

- **Alucinaciones:** generar información falsa con confianza.
- **Sesgo:** reflejan sesgos presentes en Data de Training.
- **No razonan:** memorizan y generan patrones estadísticos.
- **Interpretabilidad:** difícil entender "por qué" generan cierta salida.

📹 **Videos recomendados:**

1. [Large Language Models Explained - Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g) - 1 hora
1. [GPT-3 Paper Explained](https://www.youtube.com/watch?v=SY5PvZrJhLE) - 30 min

______________________________________________________________________

## 7. Prompting y Prompt Engineering

### ¿Qué es un prompt?

La entrada de texto que le das al LLM para generar una salida.

### Anatomía de un buen prompt

1. **Instrucción clara:** qué quieres que haga.
1. **Contexto:** información relevante.
1. **Formato esperado:** cómo quieres la respuesta.

**Example:**

```
Tarea: Clasificar el sentimiento de la siguiente reseña.
Reseña: "El producto llegó roto y el servicio al cliente nunca respondió."
Formato: Responde solo con "Positivo", "Neutral" o "Negativo".
```

### Técnicas avanzadas

- **Zero-shot:** sin Examples.
- **Few-shot:** incluir 2-5 Examples de entrada-salida.
- **Chain-of-Thought (CoT):** pedir razonamiento paso a paso.
- **Self-consistency:** generar múltiples respuestas y votar.

📹 **Videos recomendados:**

1. [Prompt Engineering Guide - OpenAI](https://www.youtube.com/watch?v=T9aRN5JkmL8) - 25 min
1. [Advanced Prompting - DeepLearning.AI](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) - curso corto gratuito

📚 **Resources escritos:**

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering Guide (GitHub)](https://github.com/dair-ai/Prompt-Engineering-Guide)

______________________________________________________________________

## 8. Evaluation de LLMs

### Metrics automáticas

- **BLEU, ROUGE:** comparar texto generado con referencia (traducción, resumen).
- **Perplexity:** qué tan "sorprendido" está el Model (menor = mejor).

**Limitación:** No capturan calidad real, especialmente en tareas abiertas.

### Evaluation humana

- Correctitud, relevancia, coherencia, consistencia, seguridad.
- Costosa pero esencial para producción.

### Evaluation con LLM

Usar un LLM más potente para evaluar salidas de otro Model ("LLM-as-a-judge").

📹 **Videos recomendados:**

1. [Evaluating LLMs - Stanford CS324](https://www.youtube.com/watch?v=HJUVRyIHpCQ) - 40 min

______________________________________________________________________

## 9. Fine-tuning y RAG

### Fine-tuning

Reentrenar un LLM pre-entrenado en Data específicos de tu dominio.

**Ventaja:** Model especializado.
**Desventaja:** Costoso, requiere Data de calidad.

### RAG (Retrieval-Augmented Generation)

Combinar LLM con búsqueda en base de documentos.

**Pipeline:**

1. Usuario hace pregunta.
1. Sistema busca documentos relevantes.
1. LLM genera respuesta usando esos documentos como contexto.

**Ventaja:** Siempre actualizado, menos alucinaciones, más barato que fine-tuning.

📹 **Videos recomendados:**

1. [Fine-tuning LLMs - Hugging Face](https://www.youtube.com/watch?v=eC6Hd1hFvos) - 30 min
1. [RAG Explained - LangChain](https://www.youtube.com/watch?v=sVcwVQRHIc8) - 20 min

📚 **Resources escritos:**

- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)

______________________________________________________________________

## 10. Buenas Practices

- ✅ Empezar con baseline clásico (TF-IDF + Logistic Regression) para establecer punto de referencia.
- ✅ Probar LLMs solo si el Problem lo justifica (no usar cañón para matar mosquito).
- ✅ Guardar prompts y Results para comparar iteraciones.
- ✅ Medir con Metrics Y analizar Examples de error.
- ✅ Validar que el Model no memorice Data sensibles.
- ✅ Implementar guardrails (Filters de seguridad, Validation de salidas).
- ✅ Monitorear costos en APIs de LLMs.

📚 **Resources generales:**

- [Speech and Language Processing (Book - Free)](https://web.stanford.edu/~jurafsky/slp3/)
- [Hugging Face Transformers Course](https://huggingface.co/course/chapter0)
- [Fast.ai NLP Course](https://www.fast.ai/posts/2019-07-08-fastai-nlp.html)

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente Module, deberías poder:

- ✅ Explicar diferencia entre BoW, TF-IDF y embeddings contextuales.
- ✅ Decidir cuándo usar Model clásico vs Transformer vs LLM.
- ✅ Tokenizar y preprocesar texto para Classification.
- ✅ Diseñar prompts efectivos con instrucción, contexto y formato.
- ✅ Implementar Classification de texto con scikit-learn y Transformers.
- ✅ Evaluar Models NLP con Metrics apropiadas.
- ✅ Identificar alucinaciones y sesgos en salidas de LLMs.
- ✅ Explicar trade-offs entre fine-tuning y RAG.

Si respondiste "sí" a todas, estás listo para aplicaciones avanzadas de NLP.
