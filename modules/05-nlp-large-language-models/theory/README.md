# Theory — NLP & Large Language Models

## Why this module matters

80% of business data is in text format. NLP allows you to extract value from documents, emails, chats, reviews and more. Large Language Models (LLMs) have democratized previously impossible capabilities: generation, translation, summary and reasoning about text.

______________________________________________________________________

## 1. What is NLP?

**Natural Language Processing (NLP)** allows computer systems to **understand, process and generate** human language.

### Challenges of human language

- **Ambiguity:** "bank" can be a financial institution or seat.
- **Context:** the meaning depends on what comes before/after.
- **Sarcasm and irony:** difficult to detect.
- **Linguistic variations:** dialects, slang, spelling errors.

### Real applications

- Chatbots and virtual assistants.
- Analysis of Sentiment in social networks.
- Automatic translation.
- Summary of documents.
- Semantic search.
- Extraction of information from contracts/invoices.

📹 **Videos recommended:**

1. [NLP Introduction - Stanford](https://www.youtube.com/watch?v=8rXD5-xhemo) - 20 min
1. [NLP Zero to Hero - TensorFlow](https://www.youtube.com/watch?v=fNxaJsNG3-s&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S) - series complete

______________________________________________________________________

## 2. Classic NLP Pipeline

### Step 1: Text collection

- Web scraping, APIs (Twitter, Reddit), public datasets.

### Paso 2: Cleaning

**Common operations:**

- Convert to lowercase.
- Remove punctuation (according to case).
- Remove stopwords (very frequent words: "el", "la", "de").
- Character normalization (accents, unicode).

**Caution:** Always validate if Cleaning helps your specific task.

### Paso 3: Tokenization

Divide text into units (tokens): words, subwords or characters.

**Example:**

```
"Machine Learning es genial."
→ ["Machine", "Learning", "es", "genial", "."]
```

### Step 4: Numerical representation

Models need numbers, not text.

**Options:**

- **Bag of Words (BoW):** count frequency of each word (ignore order).
- **TF-IDF:** penalizes very frequent words, rewards discriminative words.
- **embeddings:** dense vectors that capture semantics (Word2Vec, GloVe, FastText).

### Step 5: Modeling

- Classic classifiers (Naive Bayes, Logistic Regression, Random Forest).
- neural networks (RNN, LSTM, Transformers).

### Paso 6: Evaluation

- Metrics according to task (accuracy, f1, BLEU, ROUGE).

📹 **Videos recommended:**

1. [Text Preprocessing - Krish Naik](https://www.youtube.com/watch?v=nxhCyeRR75Q) - 30 min
1. [Tokenization Explained - Hugging Face](https://www.youtube.com/watch?v=VFp38yj8h3A) - 15 min

📚 **Resources written:**

- [NLTK Book (Free)](https://www.nltk.org/book/)
- [spaCy Documentation](https://spacy.io/usage/spacy-101)

______________________________________________________________________

## 3. Text representations

### Bag of Words (BoW)

**Concept:** Represent document as a vector of frequencies of words.

**Advantage:** Simple, interpretable.
**Disadvantage:** Ignore order and context.

### TF-IDF (Term Frequency - Inverse Document Frequency)

**Formula:** `TF-IDF = (frequency in document) × log(total docs / docs with term)`

**Advantage:** Reduces weight of very common words.
**Usage:** Information search, classic Classification.

### Word embeddings

**Concept:** Represent words as dense vectors in continuous space where similar words are close.

**Popular models:**

- **Word2Vec:** learns embeddings by predicting neighboring words.
- **GloVe:** based on global co-occurrences.
- **FastText:** similar to Word2Vec but handles subwords (good for typos).

**Advantage:** Captures semantic relationships.

- Example: `king - man + woman ≈ queen`

### contextual embeddings (Transformers)

Words have different representations according to context.

- **BERT, RoBERTa, GPT:** Pre-trained models that generate contextual embeddings.

📹 **Videos recommended:**

1. [Word2Vec Explained - StatQuest](https://www.youtube.com/watch?v=viZrOnJclY0) - 20 min
1. [Word Embeddings - Corey Schafer](https://www.youtube.com/watch?v=aHLslaWO-AQ) - 15 min

______________________________________________________________________

## 4. Frequent tasks in NLP

### Text classification

- Detect spam, Sentiment (positive/negative), user intention.
- **Metrics:** accuracy, Precision, recall, f1.

### Named Entity Recognition (NER)

Extract entities: people, places, organizations, dates.

**Example:**

```
"Elon Musk founded Tesla en 2003."
→ [Elon Musk: PERSONA], [Tesla: ORGANIZATION], [2003: DATE]
```

### Semantic search

Find relevant documents based on meaning, not just exact words.

### Automatic summary

- **Extraction:** select key sentences from the original.
- **Abstractive:** generate summary with your own language (requires LLM).

### Text generation

Produce coherent text: chatbots, auto-completion, content creation.

📹 **Videos recommended:**

1. [Text Classification - Krish Naik](https://www.youtube.com/watch?v=VtRLrQ3Ev-U) - 45 min
1. [Named Entity Recognition - freeCodeCamp](https://www.youtube.com/watch?v=q-LE7w55ZGA) - 30 min

______________________________________________________________________

## 5. Neural network architectures for NLP

### RNN (Recurrent Neural Networks)

They process sequences from left to right, maintaining "memory" of the previous context.

**Limitation:** Difficulty with long sequences (vanishing gradients).

### LSTM (Long Short-Term Memory)

RNN variant that mitigates vanishing gradients with gates.

### Transformers

**Revolution in NLP (2017).** Based on **attention** mechanism.

**Advantages:**

- They process sequences in parallel (not sequentially like RNN).
- They capture long-term dependencies.
- Scalable to huge Models.

**Key architectures:**

- **BERT:** encoder (good for Classification, NER).
- **GPT:** decoder (good for generation).
- **T5, BART:** encoder-decoder (translation, summary).

📹 **Videos recommended (FUNDAMENTAL):**

1. [Attention is All You Need - Illustrated](https://www.youtube.com/watch?v=4Bdc55j80l8) - 15 min
1. [BERT Explained - CodeEmporium](https://www.youtube.com/watch?v=xI0HHN5XKDo) - 20 min
1. [Transformers from Scratch - Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) - 2 hours

📚 **Resources written:**

- [The Illustrated Transformer (Blog)](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Course (Free)](https://huggingface.co/course/chapter1)

______________________________________________________________________

## 6. Large Language Models (LLMs)

### What are they?

**Massive** language models (millions/billions of parameters) trained on huge textual corpora (internet, books, code).

**Examples:**

- GPT-4 (OpenAI)
- Claude (Anthropic)
- Llama (Meta)
- Gemini (Google)

### Emerging capabilities

As Models grow, skills not explicitly trained emerge:

- Multi-step reasoning.
- Translation between languages ​​that the Model never saw together.
- Code generation.
- Resolution of mathematical problems.

### Critical limitations

- **Hallucinations:** generating false information with confidence.
- **Bias:** reflect biases present in Training Data.
- **They do not reason:** they memorize and generate statistical patterns.
- **Interpretability:** difficult to understand "why" they generate certain output.

📹 **Videos recommended:**

1. [Large Language Models Explained - Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g) - 1 hour
1. [GPT-3 Paper Explained](https://www.youtube.com/watch?v=SY5PvZrJhLE) - 30 min

______________________________________________________________________

## 7. Prompting and Prompt Engineering

### What is a prompt?

The text input that you give to the LLM to generate an output.

### Anatomy of a good prompt

1. **Clear instruction:** what do you want it to do.
1. **Context:** relevant information.
1. **Expected format:** how you want the response.

**Example:**

```
Task: Classify the sentiment of the following review.
Review: "The product arrived broken and customer service never responded."
Format: Respond only with "Positive", "Neutral", or "Negative".
```

### Advanced techniques

- **Zero-shot:** sin Examples.
- **Few-shot:** include 2-5 input-output examples.
- **Chain-of-Thought (CoT):** ask for step-by-step reasoning.
- **Self-consistency:** generate multiple responses and vote.

📹 **Videos recommended:**

1. [Prompt Engineering Guide - OpenAI](https://www.youtube.com/watch?v=T9aRN5JkmL8) - 25 min
1. [Advanced Prompting - DeepLearning.AI](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) - free short cursor

📚 **Resources written:**

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering Guide (GitHub)](https://github.com/dair-ai/Prompt-Engineering-Guide)

______________________________________________________________________

## 8. LLM Evaluation

### Automatic Metrics

- **BLEU, ROUGE:** compare generated text with reference (translation, summary).
- **Perplexity:** how "surprised" the Model is (less = better).

**Limitation:** They do not capture real quality, especially in open tasks.

### Evaluation humana

- Correctness, relevance, coherence, consistency, security.
- Expensive but essential for production.

### Evaluation with LLM

Use a more powerful LLM to evaluate outputs from another Model ("LLM-as-a-judge").

📹 **Videos recommended:**

1. [Evaluating LLMs - Stanford CS324](https://www.youtube.com/watch?v=HJUVRyIHpCQ) - 40 min

______________________________________________________________________

## 9. Fine-tuning and RAG

### Fine-tuning

Retrain a pre-trained LLM on data specific to your domain.

**Advantage:** Specialized model.
**Disadvantage:** Expensive, requires quality data.

### RAG (Retrieval-Augmented Generation)

Combine LLM with document base search.

**Pipeline:**

1. User has question.
1. System searches for relevant documents.
1. LLM generates response using those documents as context.

**Advantage:** Always updated, less hallucinations, cheaper than fine-tuning.

📹 **Videos recommended:**

1. [Fine-tuning LLMs - Hugging Face](https://www.youtube.com/watch?v=eC6Hd1hFvos) - 30 min
1. [RAG Explained - LangChain](https://www.youtube.com/watch?v=sVcwVQRHIc8) - 20 min

📚 **Resources written:**

- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)

______________________________________________________________________

## 10. Buenas Practices

- ✅ Start with classic baseline (TF-IDF + Logistic Regression) to establish reference point.
- ✅ Try LLMs only if the Problem justifies it (do not use a cannon to kill mosquitoes).
- ✅ Save prompts and Results to compare iterations.
- ✅ Measure with Metrics and analyze error examples.
- ✅ Validate that the Model does not memorize sensitive Data.
- ✅ Implement guardrails (security filters, output validation).
- ✅ Monitor costs in LLM APIs.

📚 **General resources:**

- [Speech and Language Processing (Book - Free)](https://web.stanford.edu/~jurafsky/slp3/)
- [Hugging Face Transformers Course](https://huggingface.co/course/chapter0)
- [Fast.ai NLP Course](https://www.fast.ai/posts/2019-07-08-fastai-nlp.html)

______________________________________________________________________

## Final comprehension checklist

Before moving to the next Module, you should be able to:

- ✅ Explain difference between BoW, TF-IDF and contextual embeddings.
- ✅ Decide when to use Classic Model vs Transformer vs LLM.
- ✅ Tokenize and preprocess text for Classification.
- ✅ Design effective prompts with instruction, context and format.
- ✅ Implement text classification with scikit-learn and Transformers.
- ✅ Evaluate NLP Models with appropriate Metrics.
- ✅ Identify hallucinations and biases in LLM outputs.
- ✅ Explain trade-offs between fine-tuning and RAG.

If you answered "yes" to all, you are ready for advanced NLP applications.
