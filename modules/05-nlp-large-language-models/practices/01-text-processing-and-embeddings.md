# Practice 01 — Text processing and embeddings

## 🎯 Objectives

- Preprocesar text (Tokenization, Cleaning)
- Vectorizar text (BoW, TF-IDF, embeddings)
- Train Models of Text Classification
- Wear word embeddings pre-trained

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Text preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

# Descargar resources
nltk.download('stopwords')
nltk.download('wordnet')

# Texto de example
texts = [
    "Machine Learning is amazing!!! I love ML.",
    "NLP is a subfield of AI. Natural Language Processing rocks.",
    "Deep learning models require lots of data for training.",
]

# Pipeline de cleaning
def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remover punctuation y numbers
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

# Apply
cleaned_texts = [preprocess_text(text) for text in texts]

for original, cleaned in zip(texts, cleaned_texts):
    print(f"Original: {original}")
    print(f"Cleaned:  {cleaned}\\n")
```

**✅ Solution - Vectorization:**

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

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Classification of Feelings

**Statement:**

1. Load reviews dataset (IMDB or custom)
1. Preprocesa textos
1. Vectorize with TF-IDF
1. Train Logistic Regression and SVM
1. Compare accuracy

### Exercise 2.2: N-grams

**Statement:**
Use `CountVectorizer(ngram_range=(1,2))` to capture bigrams.
Compare performance of unigrams vs unigrams+bigrams.

### Exercise 2.3: Word2Vec with Gensim

**Statement:**

1. Train Word2Vec with `gensim`
1. Encuentra words similar: `model.most_similar('machine')`
1. Visualize embeddings with t-SNE

### Exercise 2.4: embeddings Pre-trained

**Statement:**
Descarga GloVe embeddings:

1. Carga vectors pre-trained
1. Represents documents as average of embeddings
1. Classify with Random Forest

### Exercise 2.5: Topic Modeling with LDA

**Statement:**
Usa `sklearn.decomposition.LatentDirichletAllocation`:

1. Extract 5 topics from corpus
1. View top words by topic
1. Asigna documents a topics

______________________________________________________________________

## ✅ Checklist

- [ ] Clear and tokenize text
- [ ] Remove stopwords and apply stemming/lemmatization
- [ ] Vectorize with BoW and TF-IDF
- [ ] Train text classifiers
- [ ] Wear word embeddings (Word2Vec, GloVe)
- [ ] Visualize embeddings
- [ ] Apply topic modeling

______________________________________________________________________

## 📚 Resources

- [NLTK Book](https://www.nltk.org/book/)
- [Gensim Tutorials](https://radimrehurek.com/gensim/auto_examples/index.html)
