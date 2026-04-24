# Práctica 01 — Procesamiento de Texto y Embeddings

## 🎯 Objetivos

- Preprocesar texto (tokenización, limpieza)
- Vectorizar texto (BoW, TF-IDF, embeddings)
- Entrenar modelos de clasificación de texto
- Usar word embeddings pre-entrenados

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Preprocesamiento de texto

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

# Descargar recursos
nltk.download('stopwords')
nltk.download('wordnet')

# Texto de ejemplo
texts = [
    "Machine Learning is amazing!!! I love ML.",
    "NLP is a subfield of AI. Natural Language Processing rocks.",
    "Deep learning models require lots of data for training.",
]

# Pipeline de limpieza
def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remover puntuación y números
    text = re.sub(r'[^a-zA-Z\\s]', '', text)

    # Tokenizar
    tokens = text.split()

    # Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Aplicar
cleaned_texts = [preprocess_text(text) for text in texts]

for original, cleaned in zip(texts, cleaned_texts):
    print(f"Original: {original}")
    print(f"Cleaned:  {cleaned}\\n")
```

**✅ Solución - Vectorización:**

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bag of Words
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(cleaned_texts)

print("=== Bag of Words ===")
print(f"Vocabulary: {bow_vectorizer.get_feature_names_out()}")
print(f"\\nBoW Matrix:\\n{bow_matrix.toarray()}")

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_texts)

print("\\n=== TF-IDF ===")
print(f"TF-IDF Matrix:\\n{tfidf_matrix.toarray()}")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Clasificación de Sentimientos

**Enunciado:**

1. Carga dataset de reviews (IMDB o custom)
1. Preprocesa textos
1. Vectoriza con TF-IDF
1. Entrena Logistic Regression y SVM
1. Compara accuracy

### Ejercicio 2.2: N-grams

**Enunciado:**
Usa `CountVectorizer(ngram_range=(1,2))` para capturar bigramas.
Compara performance de unigrams vs unigrams+bigrams.

### Ejercicio 2.3: Word2Vec con Gensim

**Enunciado:**

1. Entrena Word2Vec con `gensim`
1. Encuentra palabras similares: `model.most_similar('machine')`
1. Visualiza embeddings con t-SNE

### Ejercicio 2.4: Embeddings Pre-entrenados

**Enunciado:**
Descarga GloVe embeddings:

1. Carga vectores pre-entrenados
1. Representa documentos como promedio de embeddings
1. Clasifica con Random Forest

### Ejercicio 2.5: Topic Modeling con LDA

**Enunciado:**
Usa `sklearn.decomposition.LatentDirichletAllocation`:

1. Extrae 5 topics de corpus
1. Visualiza top words por topic
1. Asigna documentos a topics

______________________________________________________________________

## ✅ Checklist

- [ ] Limpiar y tokenizar texto
- [ ] Remover stopwords y aplicar stemming/lemmatization
- [ ] Vectorizar con BoW y TF-IDF
- [ ] Entrenar clasificadores de texto
- [ ] Usar word embeddings (Word2Vec, GloVe)
- [ ] Visualizar embeddings
- [ ] Aplicar topic modeling

______________________________________________________________________

## 📚 Recursos

- [NLTK Book](https://www.nltk.org/book/)
- [Gensim Tutorials](https://radimrehurek.com/gensim/auto_examples/index.html)
