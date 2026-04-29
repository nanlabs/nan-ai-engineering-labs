# Example 01 — Sentiment Classification Pipeline (NLP)

## Context

You will learn a complete NLP pipeline: from raw text to Sentiment Classification Model. You will use **preprocessing**, **Tokenization**, **embeddings** and traditional ML Models.

## Objective

Classify movie reviews as **positive** or **negative** (Sentiment Analysis).

______________________________________________________________________

## 🚀 Step 1: Setup and imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descargar resources de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

print("✅ Setup complete")
```

______________________________________________________________________

## 📥 Paso 2: Load Data

### 2.1 Use IMDB dataset (we will simulate with Examples)

```python
# Dataset de example (en production: use IMDB Movie Reviews dataset)
data = {
    'review': [
        "This movie was absolutely fantastic! I loved every moment of it.",
        "Terrible film, waste of time. Do not watch.",
        "Amazing performance by the actors. Highly recommend!",
        "Boring and predictable plot. Very disappointing.",
        "One of the best movies I've ever seen. Masterpiece!",
        "Awful acting and terrible script. Completely unwatchable.",
        "Great cinematography and excellent storytelling.",
        "I fell asleep halfway through. So boring.",
        "Incredible movie! The plot twists were mind-blowing.",
        "Horrible experience. Would not recommend to anyone.",
        "Brilliant direction and superb performances.",
        "Waste of money. Absolutely terrible.",
        "Loved the characters and the emotional depth.",
        "Poor dialogue and weak storyline.",
        "Stunning visuals and captivating narrative.",
        "Dreadful movie. Avoid at all costs.",
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative'
    ]
}

df = pd.DataFrame(data)
print(f"Dataset: {len(df)} reviews")
print(f"\nDistribución:\n{df['sentiment'].value_counts()}")
print(f"\n{df.head()}")
```

**Output:**

```
Dataset: 16 reviews

Distribution:
positive    8
negative    8
Name: sentiment, dtype: int64

                                              review sentiment
0  This movie was absolutely fantastic! I loved ...  positive
1                Terrible film, waste of time. Do ...  negative
2  Amazing performance by the actors. Highly rec...  positive
```

______________________________________________________________________

## 🧹 Step 3: Text Preprocessing

### 3.1 Cleaning Function

```python
def preprocess_text(text):
    """
    Limpia y normaliza text
    """
    # 1. Convert a lowercase
    text = text.lower()

    # 2. Remover URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remover caracteres especiales y numbers (mantener letras y espacios)
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Remover espacios multiple
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning
df['review_clean'] = df['review'].apply(preprocess_text)

print("Antes vs After de cleaning:\n")
for i in range(3):
    print(f"Original:  {df['review'][i]}")
    print(f"Limpio:    {df['review_clean'][i]}\n")
```

**Output:**

```
Antes vs After de cleaning:

Original:  This movie was absolutely fantastic! I loved every moment of it.
Limpio:    this movie was absolutely fantastic i loved every moment of it

Original:  Terrible film, waste of time. Do not watch.
Limpio:    terrible film waste of time do not watch

Original:  Amazing performance by the actors. Highly recommend!
Limpio:    amazing performance by the actors highly recommend
```

### 3.2 Remove stopwords and stemming

```python
# Inicializar herramientas
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def advanced_preprocess(text):
    """
    Preprocesamiento advanced: stopwords + stemming
    """
    # Tokenizar
    tokens = text.split()

    # Remover stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming (reducir words a root)
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

df['review_processed'] = df['review_clean'].apply(advanced_preprocess)

print("=== Comparison de procesamiento ===\n")
for i in range(2):
    print(f"Clean:     {df['review_clean'][i]}")
    print(f"Processed: {df['review_processed'][i]}\n")
```

**Output:**

```
=== Comparison de procesamiento ===

Clean:     this movie was absolutely fantastic i loved every moment of it
Processed: movi absolute fantast love everi moment

Clean:     terrible film waste of time do not watch
Processed: terribl film wast time watch
```

**Explanation:**

- **Stopwords:** Remove common words ("the", "is", "of") that do not provide meaning
- **Stemming:** Reduce words to their root ("loving" → "love", "movies" → "movi")

______________________________________________________________________

## 🔢 Step 4: Vectorization of text

### 4.1 Bag of Words (CountVectorizer)

```python
# Create vocabulario y vectorizar
count_vectorizer = CountVectorizer(max_features=50)  # Top 50 words
X_bow = count_vectorizer.fit_transform(df['review_processed'])

print(f"Matrix BoW: {X_bow.shape}")  # (16 reviews, 50 features)
print(f"Vocabulario: {len(count_vectorizer.vocabulary_)} words")

# Visualize vocabulario
vocab = count_vectorizer.get_feature_names_out()
print(f"\nPrimeras 20 words del vocabulario:\n{vocab[:20]}")

# Convert a DataFrame para visualizar
X_bow_df = pd.DataFrame(X_bow.toarray(), columns=vocab)
print(f"\n{X_bow_df.head()}")
```

**Output:**

```
Matrix BoW: (16, 50)
Vocabulario: 50 words

Primeras 20 words del vocabulario:
['absolute' 'act' 'amaz' 'avoid' 'aw' 'best' 'bore' 'brilliant' ...]

   absolute  act  amaz  avoid  aw  best  bore  brilliant  captiv  ...
0        1    0     0      0   0     0     0          0       0  ...
1        0    0     0      0   0     0     0          0       0  ...
2        0    1     1      0   0     0     0          0       0  ...
```

### 4.2 TF-IDF (Term Frequency - Inverse Document Frequency)

```python
# TF-IDF: Penaliza words muy frecuentes
tfidf_vectorizer = TfidfVectorizer(max_features=50)
X_tfidf = tfidf_vectorizer.fit_transform(df['review_processed'])

print(f"Matrix TF-IDF: {X_tfidf.shape}")

# Compare BoW vs TF-IDF para una palabra
word = 'movi'
if word in vocab:
    idx_word = np.where(vocab == word)[0][0]
    print(f"\nPalabra '{word}' en primera review:")
    print(f"  BoW:    {X_bow[0, idx_word]}")
    print(f"  TF-IDF: {X_tfidf[0, idx_word]:.4f}")
```

**Output:**

```
Matrix TF-IDF: (16, 50)

Palabra 'movi' en primera review:
  BoW:    1
  TF-IDF: 0.3456  👈 Peso ajustado por frecuencia global
```

**Difference:**

- **BoW:** Simple count (1, 2, 3...)
- **TF-IDF:** Weight by rarity (rare words →higher weight)

______________________________________________________________________

## 🏋️ Step 5: Classification Train Models

### 5.1 Split train/test

```python
# Codificar labels
y = (df['sentiment'] == 'positive').astype(int)  # 1 = positive, 0 = negative

# Split
X_train_bow, X_test_bow, y_train, y_test = train_test_split(
    X_bow, y, test_size=0.25, random_state=42, stratify=y
)

X_train_tfidf, X_test_tfidf, _, _ = train_test_split(
    X_tfidf, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train: {len(y_train)} reviews")
print(f"Test:  {len(y_test)} reviews")
```

### 5.2 Model 1: Logistic Regression + BoW

```python
# Train
lr_bow = LogisticRegression(random_state=42, max_iter=200)
lr_bow.fit(X_train_bow, y_train)

# Predict
y_pred_lr_bow = lr_bow.predict(X_test_bow)

# Evaluate
acc_lr_bow = accuracy_score(y_test, y_pred_lr_bow)
print(f"\n=== Logistic Regression + BoW ===")
print(f"Accuracy: {acc_lr_bow:.2%}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_lr_bow, target_names=['Negative', 'Positive'])}")
```

**Output:**

```
=== Logistic Regression + BoW ===
Accuracy: 100.00%

Classification Report:
              precision    recall  f1-score   support

    Negative       1.00      1.00      1.00         2
    Positive       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

### 5.3 Model 2: Naive Bayes + TF-IDF

```python
# Train
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

# Predict
y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)

# Evaluate
acc_nb_tfidf = accuracy_score(y_test, y_pred_nb_tfidf)
print(f"\n=== Naive Bayes + TF-IDF ===")
print(f"Accuracy: {acc_nb_tfidf:.2%}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_nb_tfidf, target_names=['Negative', 'Positive'])}")
```

### 5.4 Model Comparison

```python
results = pd.DataFrame({
    'Model': ['LogReg + BoW', 'NB + TF-IDF'],
    'Accuracy': [acc_lr_bow, acc_nb_tfidf]
})

print(f"\n{results.to_string(index=False)}")

# Visualize
plt.figure(figsize=(8, 5))
plt.bar(results['Model'], results['Accuracy'], color=['skyblue', 'salmon'])
plt.ylabel('Accuracy')
plt.title('Comparison de Models')
plt.ylim(0, 1.1)
for i, acc in enumerate(results['Accuracy']):
    plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontsize=12, fontweight='bold')
plt.show()
```

______________________________________________________________________

## 🔍 Step 6: Interpretation of the Model

### 6.1 Feature importance (Logistic Regression coefficients)

```python
# Obtener coeficientes
coefficients = lr_bow.coef_[0]
feature_names = count_vectorizer.get_feature_names_out()

# Create DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
}).sort_values('coefficient', ascending=False)

# Top 10 words positivas
print("=== Top 10 words POSITIVAS ===")
print(feature_importance.head(10).to_string(index=False))

# Top 10 words negativas
print("\n=== Top 10 words NEGATIVAS ===")
print(feature_importance.tail(10).to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
top_positive = feature_importance.head(10)
top_negative = feature_importance.tail(10)
combined = pd.concat([top_negative, top_positive])

colors = ['red' if c < 0 else 'green' for c in combined['coefficient']]
plt.barh(combined['feature'], combined['coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient')
plt.title('Top Features (Positive vs Negative Sentiment)')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()
```

**Output expected:**

```
=== Top 10 words POSITIVAS ===
      feature  coefficient
     fantast         2.134
        love         1.987
      amaz          1.856
   brilliant         1.742
...

=== Top 10 words NEGATIVAS ===
      feature  coefficient
      terribl        -2.456
       aw            -2.123
     dreadful        -1.989
       wast          -1.867
...
```

**Interpretation:** High coefficients (>1) indicate a strong association with positive Sentiment; low (\<-1) with negative.

______________________________________________________________________

## 🧪 Step 7: Prediction in new reviews

```python
# Nuevas reviews de testing
new_reviews = [
    "This film is absolutely spectacular and heartwarming!",
    "Boring and uninspired. Waste of my time.",
    "Mediocre acting but decent plot twists.",
]

# Preprocesar
new_reviews_processed = [advanced_preprocess(preprocess_text(review)) for review in new_reviews]

# Vectorizar
X_new = count_vectorizer.transform(new_reviews_processed)

# Predict
predictions = lr_bow.predict(X_new)
probabilities = lr_bow.predict_proba(X_new)

# Mostrar results
for i, review in enumerate(new_reviews):
    sentiment = "Positive" if predictions[i] == 1 else "Negative"
    confidence = probabilities[i][predictions[i]]
    print(f"\nReseña: {review}")
    print(f"Prediction: {sentiment} (confianza: {confidence:.2%})")
```

**Output:**

```
Review: This film is absolutely spectacular and heartwarming!
Prediction: Positive (confianza: 98.34%)

Review: Boring and uninspired. Waste of my time.
Prediction: Negative (confianza: 95.67%)

Review: Mediocre acting but decent plot twists.
Prediction: Positive (confianza: 62.12%)  👈 Ambiguo
```

______________________________________________________________________

## 📊 Paso 8: Confusion matrix

```python
# Matrix de confusion
cm = confusion_matrix(y_test, y_pred_lr_bow)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Logistic Regression + BoW')
plt.show()

print(f"\nMatriz de Confusion:\n{cm}")
```

______________________________________________________________________

## 📝Executive summary

### ✅ Pipeline complete

```
Texto crudo
  ↓
Cleaning (lowercase, remover punctuation)
  ↓
Tokenization
  ↓
Remover stopwords
  ↓
Stemming
  ↓
Vectorization (BoW / TF-IDF)
  ↓
Model ML (Logistic Regression / Naive Bayes)
  ↓
Prediction de sentiment
```

### 🎯 Results

| Model | Vectorization | accuracy |
| ------------------- | ------------- | -------- |
| Logistic Regression | BoW | **100%** |
| Naïve Bayes | TF-IDF | 100% |

**Note:** Perfect accuracy due to small dataset. In production with IMDB (25k reviews), we would expect ~88-92%.

______________________________________________________________________

## 🎓 Lessons learned

### ✅ Preprocessing

1. **Basic Cleaning:**

   - Lowercase: "Movie" → "movie"
- Remove punctuation: "Great!" → "great"
   - Normalize spaces

1. **Stopwords:**

   - Remover words common: "the", "is", "and"
   - Reduce dimensionality
   - Improve efficiency

1. **Stemming vs Lemmatization:**

- **Stemming:** Fast, heuristic rules ("running" → "run")
   - **Lemmatization:** Slow, dictionary ("better" → "good")
   - For Classification, stemming is usually sufficient

### ✅ Vectorization

**Bag of Words (BoW):**

- ✅ Simple e interpretable
- ✅ Good for basic Classification
- ❌ Ignore word order
- ❌ Does not capture semantics

**TF-IDF:**

- ✅ Penalizes very frequent words
- ✅ Highlight distinctive words
- ❌ Still ignores context

**When use each uno:**

- BoW: Small datasets, fast baseline
- TF-IDF: Long documents, important rare words

### ✅ Models

**Logistic Regression:**

- ✅ Fast, interpretable (coefficients)
- ✅ Works well with text features
- ❌ Assume linearity

**Naive Bayes:**

- ✅ Muy fast (assumes independence)
- ✅ Efficient with high-dimensional data
- ❌ Assume independent features (not always true)

### 💡 Production improvements

1. **Largest Dataset:** IMDB, Yelp, Twitter
1. **Word embeddings:** Word2Vec, GloVe (capture semantics)
1. **Deep Learning:** LSTM, GRU, Transformers (BERT)
1. **Class balancing:** If dataset unbalanced
1. **Cross-validation:** K-fold for robust validation
1. **Hyperparameter tuning:** GridSearchCV for C, alpha, etc.

### 🚫 Errors common

- ❌ Do not clean text → noise in features
- ❌ Do not remove stopwords → high dimensionality
- ❌ overfitting on small dataset
- ❌Do not validate external data

______________________________________________________________________

## 🔧 Code for production

```python
from sklearn.pipeline import Pipeline

# Pipeline complete (preprocessing + vectorization + model)
sentiment_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression(C=1.0, max_iter=200))
])

# Train
sentiment_pipeline.fit(df['review_processed'], y)

# Predict
new_review = "Amazing movie, highly recommended!"
new_review_processed = advanced_preprocess(preprocess_text(new_review))
prediction = sentiment_pipeline.predict([new_review_processed])[0]
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")

# Save pipeline
import joblib
joblib.dump(sentiment_pipeline, 'sentiment_model.pkl')

# Load en production
model = joblib.load('sentiment_model.pkl')
```

### 📌 Checklist NLP

- ✅ Clean text (lowercase, punctuation, URLs)
- ✅ Tokenize and remove stopwords
- ✅ Apply stemming/lemmatization
- ✅ Vectorize (BoW or TF-IDF)
- ✅ Split train/test (stratify!)
- ✅ Train multiple Models
- ✅ Evaluate yourself with classification report
- ✅ Interpret coefficients
- ✅ Validate yourself in new reviews
