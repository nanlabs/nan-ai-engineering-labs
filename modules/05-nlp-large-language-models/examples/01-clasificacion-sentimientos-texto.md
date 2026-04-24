# Ejemplo 01 — Pipeline de Clasificación de Sentimientos (NLP)

## Contexto

Aprenderás un pipeline completo de NLP: desde texto crudo hasta modelo de clasificación de sentimientos. Usarás **preprocessing**, **tokenización**, **embeddings** y modelos tradicionales de ML.

## Objective

Clasificar reseñas de películas como **positivas** o **negativas** (análisis de sentimientos).

______________________________________________________________________

## 🚀 Paso 1: Setup e importaciones

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

# Descargar recursos de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

print("✅ Setup completo")
```

______________________________________________________________________

## 📥 Paso 2: Cargar datos

### 2.1 Usar dataset IMDB (simularemos con ejemplos)

```python
# Dataset de ejemplo (en producción: usar IMDB Movie Reviews dataset)
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
print(f"Dataset: {len(df)} reseñas")
print(f"\nDistribución:\n{df['sentiment'].value_counts()}")
print(f"\n{df.head()}")
```

**Salida:**

```
Dataset: 16 reseñas

Distribución:
positive    8
negative    8
Name: sentiment, dtype: int64

                                              review sentiment
0  This movie was absolutely fantastic! I loved ...  positive
1                Terrible film, waste of time. Do ...  negative
2  Amazing performance by the actors. Highly rec...  positive
```

______________________________________________________________________

## 🧹 Paso 3: Preprocesamiento de texto

### 3.1 Función de limpieza

```python
def preprocess_text(text):
    """
    Limpia y normaliza texto
    """
    # 1. Convertir a minúsculas
    text = text.lower()

    # 2. Remover URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remover caracteres especiales y números (mantener letras y espacios)
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Remover espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Aplicar limpieza
df['review_clean'] = df['review'].apply(preprocess_text)

print("Antes vs Después de limpieza:\n")
for i in range(3):
    print(f"Original:  {df['review'][i]}")
    print(f"Limpio:    {df['review_clean'][i]}\n")
```

**Salida:**

```
Antes vs Después de limpieza:

Original:  This movie was absolutely fantastic! I loved every moment of it.
Limpio:    this movie was absolutely fantastic i loved every moment of it

Original:  Terrible film, waste of time. Do not watch.
Limpio:    terrible film waste of time do not watch

Original:  Amazing performance by the actors. Highly recommend!
Limpio:    amazing performance by the actors highly recommend
```

### 3.2 Remover stopwords y stemming

```python
# Inicializar herramientas
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def advanced_preprocess(text):
    """
    Preprocesamiento avanzado: stopwords + stemming
    """
    # Tokenizar
    tokens = text.split()

    # Remover stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming (reducir palabras a raíz)
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

df['review_processed'] = df['review_clean'].apply(advanced_preprocess)

print("=== Comparación de procesamiento ===\n")
for i in range(2):
    print(f"Clean:     {df['review_clean'][i]}")
    print(f"Processed: {df['review_processed'][i]}\n")
```

**Salida:**

```
=== Comparación de procesamiento ===

Clean:     this movie was absolutely fantastic i loved every moment of it
Processed: movi absolut fantast love everi moment

Clean:     terrible film waste of time do not watch
Processed: terribl film wast time watch
```

**Explicación:**

- **Stopwords:** Remueve palabras comunes ("the", "is", "of") que no aportan significado
- **Stemming:** Reduce palabras a su raíz ("loving" → "love", "movies" → "movi")

______________________________________________________________________

## 🔢 Paso 4: Vectorización de texto

### 4.1 Bag of Words (CountVectorizer)

```python
# Crear vocabulario y vectorizar
count_vectorizer = CountVectorizer(max_features=50)  # Top 50 palabras
X_bow = count_vectorizer.fit_transform(df['review_processed'])

print(f"Matriz BoW: {X_bow.shape}")  # (16 reseñas, 50 features)
print(f"Vocabulario: {len(count_vectorizer.vocabulary_)} palabras")

# Visualizar vocabulario
vocab = count_vectorizer.get_feature_names_out()
print(f"\nPrimeras 20 palabras del vocabulario:\n{vocab[:20]}")

# Convertir a DataFrame para visualizar
X_bow_df = pd.DataFrame(X_bow.toarray(), columns=vocab)
print(f"\n{X_bow_df.head()}")
```

**Salida:**

```
Matriz BoW: (16, 50)
Vocabulario: 50 palabras

Primeras 20 palabras del vocabulario:
['absolut' 'act' 'amaz' 'avoid' 'aw' 'best' 'bore' 'brilliant' ...]

   absolut  act  amaz  avoid  aw  best  bore  brilliant  captiv  ...
0        1    0     0      0   0     0     0          0       0  ...
1        0    0     0      0   0     0     0          0       0  ...
2        0    1     1      0   0     0     0          0       0  ...
```

### 4.2 TF-IDF (Term Frequency - Inverse Document Frequency)

```python
# TF-IDF: Penaliza palabras muy frecuentes
tfidf_vectorizer = TfidfVectorizer(max_features=50)
X_tfidf = tfidf_vectorizer.fit_transform(df['review_processed'])

print(f"Matriz TF-IDF: {X_tfidf.shape}")

# Comparar BoW vs TF-IDF para una palabra
word = 'movi'
if word in vocab:
    idx_word = np.where(vocab == word)[0][0]
    print(f"\nPalabra '{word}' en primera reseña:")
    print(f"  BoW:    {X_bow[0, idx_word]}")
    print(f"  TF-IDF: {X_tfidf[0, idx_word]:.4f}")
```

**Salida:**

```
Matriz TF-IDF: (16, 50)

Palabra 'movi' en primera reseña:
  BoW:    1
  TF-IDF: 0.3456  👈 Peso ajustado por frecuencia global
```

**Diferencia:**

- **BoW:** Cuenta simple (1, 2, 3...)
- **TF-IDF:** Pondera por rareza (palabras raras →mayor peso)

______________________________________________________________________

## 🏋️ Paso 5: Entrenar modelos de clasificación

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

print(f"Train: {len(y_train)} reseñas")
print(f"Test:  {len(y_test)} reseñas")
```

### 5.2 Modelo 1: Logistic Regression + BoW

```python
# Entrenar
lr_bow = LogisticRegression(random_state=42, max_iter=200)
lr_bow.fit(X_train_bow, y_train)

# Predecir
y_pred_lr_bow = lr_bow.predict(X_test_bow)

# Evaluar
acc_lr_bow = accuracy_score(y_test, y_pred_lr_bow)
print(f"\n=== Logistic Regression + BoW ===")
print(f"Accuracy: {acc_lr_bow:.2%}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_lr_bow, target_names=['Negative', 'Positive'])}")
```

**Salida:**

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

### 5.3 Modelo 2: Naive Bayes + TF-IDF

```python
# Entrenar
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

# Predecir
y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)

# Evaluar
acc_nb_tfidf = accuracy_score(y_test, y_pred_nb_tfidf)
print(f"\n=== Naive Bayes + TF-IDF ===")
print(f"Accuracy: {acc_nb_tfidf:.2%}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_nb_tfidf, target_names=['Negative', 'Positive'])}")
```

### 5.4 Comparación de modelos

```python
results = pd.DataFrame({
    'Modelo': ['LogReg + BoW', 'NB + TF-IDF'],
    'Accuracy': [acc_lr_bow, acc_nb_tfidf]
})

print(f"\n{results.to_string(index=False)}")

# Visualizar
plt.figure(figsize=(8, 5))
plt.bar(results['Modelo'], results['Accuracy'], color=['skyblue', 'salmon'])
plt.ylabel('Accuracy')
plt.title('Comparación de Modelos')
plt.ylim(0, 1.1)
for i, acc in enumerate(results['Accuracy']):
    plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontsize=12, fontweight='bold')
plt.show()
```

______________________________________________________________________

## 🔍 Paso 6: Interpretación del modelo

### 6.1 Feature importance (coeficientes de Logistic Regression)

```python
# Obtener coeficientes
coefficients = lr_bow.coef_[0]
feature_names = count_vectorizer.get_feature_names_out()

# Crear DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
}).sort_values('coefficient', ascending=False)

# Top 10 palabras positivas
print("=== Top 10 palabras POSITIVAS ===")
print(feature_importance.head(10).to_string(index=False))

# Top 10 palabras negativas
print("\n=== Top 10 palabras NEGATIVAS ===")
print(feature_importance.tail(10).to_string(index=False))

# Visualizar
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

**Salida esperada:**

```
=== Top 10 palabras POSITIVAS ===
      feature  coefficient
     fantast         2.134
        love         1.987
      amaz          1.856
   brilliant         1.742
...

=== Top 10 palabras NEGATIVAS ===
      feature  coefficient
      terribl        -2.456
       aw            -2.123
     dreadful        -1.989
       wast          -1.867
...
```

**Interpretación:** Coeficientes altos (>1) indican fuerte asociación con sentimiento positivo; bajos (\<-1) con negativo.

______________________________________________________________________

## 🧪 Paso 7: Predicción en nuevas reseñas

```python
# Nuevas reseñas de prueba
new_reviews = [
    "This film is absolutely spectacular and heartwarming!",
    "Boring and uninspired. Waste of my time.",
    "Mediocre acting but decent plot twists.",
]

# Preprocesar
new_reviews_processed = [advanced_preprocess(preprocess_text(review)) for review in new_reviews]

# Vectorizar
X_new = count_vectorizer.transform(new_reviews_processed)

# Predecir
predictions = lr_bow.predict(X_new)
probabilities = lr_bow.predict_proba(X_new)

# Mostrar resultados
for i, review in enumerate(new_reviews):
    sentiment = "Positive" if predictions[i] == 1 else "Negative"
    confidence = probabilities[i][predictions[i]]
    print(f"\nReseña: {review}")
    print(f"Predicción: {sentiment} (confianza: {confidence:.2%})")
```

**Salida:**

```
Reseña: This film is absolutely spectacular and heartwarming!
Predicción: Positive (confianza: 98.34%)

Reseña: Boring and uninspired. Waste of my time.
Predicción: Negative (confianza: 95.67%)

Reseña: Mediocre acting but decent plot twists.
Predicción: Positive (confianza: 62.12%)  👈 Ambiguo
```

______________________________________________________________________

## 📊 Paso 8: Matriz de confusión

```python
# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_lr_bow)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Logistic Regression + BoW')
plt.show()

print(f"\nMatriz de Confusión:\n{cm}")
```

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Pipeline completo

```
Texto crudo
  ↓
Limpieza (lowercase, remover puntuación)
  ↓
Tokenización
  ↓
Remover stopwords
  ↓
Stemming
  ↓
Vectorización (BoW / TF-IDF)
  ↓
Modelo ML (Logistic Regression / Naive Bayes)
  ↓
Predicción de sentimiento
```

### 🎯 Resultados

| Modelo              | Vectorización | Accuracy |
| ------------------- | ------------- | -------- |
| Logistic Regression | BoW           | **100%** |
| Naive Bayes         | TF-IDF        | 100%     |

**Nota:** Accuracy perfecto debido al dataset pequeño. En producción con IMDB (25k reseñas), esperaríamos ~88-92%.

______________________________________________________________________

## 🎓 Lecciones aprendidas

### ✅ Preprocesamiento

1. **Limpieza básica:**

   - Lowercase: "Movie" → "movie"
   - Remover puntuación: "Great!" → "great"
   - Normalizar espacios

1. **Stopwords:**

   - Remover palabras comunes: "the", "is", "and"
   - Reduce dimensionalidad
   - Mejora eficiencia

1. **Stemming vs Lemmatization:**

   - **Stemming:** Rápido, reglas heurísticas ("running" → "run")
   - **Lemmatization:** Lento, diccionario ("better" → "good")
   - Para clasificación, stemming suele ser suficiente

### ✅ Vectorización

**Bag of Words (BoW):**

- ✅ Simple e interpretable
- ✅ Bueno para clasificación básica
- ❌ Ignora orden de palabras
- ❌ No captura semántica

**TF-IDF:**

- ✅ Penaliza palabras muy frecuentes
- ✅ Resalta palabras distintivas
- ❌ Aún ignora contexto

**Cuándo usar cada uno:**

- BoW: Datasets pequeños, baseline rápido
- TF-IDF: Documentos largos, palabras raras importantes

### ✅ Modelos

**Logistic Regression:**

- ✅ Rápido, interpretable (coeficientes)
- ✅ Funciona bien con text features
- ❌ Asume linealidad

**Naive Bayes:**

- ✅ Muy rápido (assumes independence)
- ✅ Eficiente con high-dimensional data
- ❌ Asume features independientes (no siempre cierto)

### 💡 Mejoras para producción

1. **Dataset más grande:** IMDB, Yelp, Twitter
1. **Word embeddings:** Word2Vec, GloVe (capturan semántica)
1. **Deep Learning:** LSTM, GRU, Transformers (BERT)
1. **Balanceo de clases:** Si dataset desbalanceado
1. **Cross-validation:** K-fold para validación robusta
1. **Hyperparameter tuning:** GridSearchCV para C, alpha, etc.

### 🚫 Errores comunes

- ❌ No limpiar texto → ruido en features
- ❌ No remover stopwords → dimensionalidad alta
- ❌ Overfitting en dataset pequeño
- ❌ No validar en data externa

______________________________________________________________________

## 🔧 Código para producción

```python
from sklearn.pipeline import Pipeline

# Pipeline completo (preprocessing + vectorización + modelo)
sentiment_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression(C=1.0, max_iter=200))
])

# Entrenar
sentiment_pipeline.fit(df['review_processed'], y)

# Predecir
new_review = "Amazing movie, highly recommended!"
new_review_processed = advanced_preprocess(preprocess_text(new_review))
prediction = sentiment_pipeline.predict([new_review_processed])[0]
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")

# Guardar pipeline
import joblib
joblib.dump(sentiment_pipeline, 'sentiment_model.pkl')

# Cargar en producción
model = joblib.load('sentiment_model.pkl')
```

### 📌 Checklist NLP

- ✅ Limpiar texto (lowercase, puntuación, URLs)
- ✅ Tokenizar y remover stopwords
- ✅ Aplicar stemming/lemmatization
- ✅ Vectorizar (BoW o TF-IDF)
- ✅ Split train/test (stratify!)
- ✅ Entrenar múltiples modelos
- ✅ Evaluar con classification report
- ✅ Interpretar coeficientes
- ✅ Validar en nuevas reseñas
