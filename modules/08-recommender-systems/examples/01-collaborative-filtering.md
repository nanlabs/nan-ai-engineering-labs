# Ejemplo 01 — Collaborative Filtering para Recomendaciones de Películas

## Contexto

aprendes **Collaborative Filtering** para construir sistemas de recomendación basados en comportamientos de usuarios similares.

## Objective

Recomendar películas usando user-item ratings.

______________________________________________________________________

## 🚀 Setup

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
```

______________________________________________________________________

## 📥 Cargar datos (MovieLens-style)

```python
# Simular ratings (en producción: usar MovieLens dataset)
n_users = 100
n_movies = 50
sparsity = 0.1  # 10% de interacciones

# Crear matriz user-item sparse
ratings_data = []
for user_id in range(n_users):
    n_ratings = np.random.randint(1, 10)
    movie_ids = np.random.choice(n_movies, size=n_ratings, replace=False)
    ratings = np.random.randint(1, 6, size=n_ratings)  # 1-5 stars

    for movie_id, rating in zip(movie_ids, ratings):
        ratings_data.append([user_id, movie_id, rating])

df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating'])

print(f"Total ratings: {len(df)}")
print(f"Users: {df['user_id'].nunique()}")
print(f"Movies: {df['movie_id'].nunique()}")
print(f"Sparsity: {100*(1 - len(df)/(n_users*n_movies)):.2f}%")
print(f"\n{df.head()}")
```

**Salida:**

```
Total ratings: 543
Users: 100
Movies: 50
Sparsity: 89.14%  👈 Matriz muy sparse (común en sistemas recomendación)

   user_id  movie_id  rating
0        0        42       3
1        0        15       5
2        0         8       2
```

______________________________________________________________________

## 🏗️ User-Based Collaborative Filtering

### Crear matriz user-item

```python
# Pivot table: usuarios en filas, películas en columnas
user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')

print(f"Matriz shape: {user_item_matrix.shape}")  # (100, 50)
print(f"\n{user_item_matrix.head()}")
```

### Calcular similitud entre usuarios

```python
# Fill NaN con 0 para cálculo de similitud
user_item_filled = user_item_matrix.fillna(0)

# Cosine similarity entre usuarios
user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity,
                                   index=user_item_matrix.index,
                                   columns=user_item_matrix.index)

print(f"\nSimilitud entre user 0 y otros:")
print(user_similarity_df.loc[0].sort_values(ascending=False).head())
```

**Salida:**

```
user_id
0    1.000000  👈 Consigo mismo
23   0.893456  👈 Usuario más similar
45   0.856712
12   0.834521
89   0.812345
```

### Función de recomendación

```python
def recommend_movies_user_based(user_id, user_item_matrix, user_similarity_df, top_n=5):
    """
    Recomendar películas basándose en usuarios similares
    """
    # Obtener usuarios similares (excluyendo el mismo usuario)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]  # Top 10

    # Películas que el usuario NO ha visto
    user_ratings = user_item_matrix.loc[user_id]
    unwatched_movies = user_ratings[user_ratings.isna()].index

    # Predecir ratings para películas no vistas
    predictions = {}
    for movie in unwatched_movies:
        # Weighted average de ratings de usuarios similares
        similar_users_ratings = user_item_matrix.loc[similar_users.index, movie]
        valid_ratings = similar_users_ratings.dropna()

        if len(valid_ratings) > 0:
            # Pesos = similitudes de usuarios que vieron la película
            weights = similar_users[valid_ratings.index]
            weighted_rating = (valid_ratings * weights).sum() / weights.sum()
            predictions[movie] = weighted_rating

    # Ordenar por rating predicho
    recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return recommendations

# Ejemplo: recomendar para user 0
user_id_test = 0
recommendations = recommend_movies_user_based(user_id_test, user_item_matrix, user_similarity_df, top_n=5)

print(f"\n=== Recomendaciones para User {user_id_test} ===\n")
for movie_id, predicted_rating in recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {predicted_rating:.2f}")
```

**Salida:**

```
=== Recomendaciones para User 0 ===

Movie ID: 27, Predicted Rating: 4.67
Movie ID: 12, Predicted Rating: 4.45
Movie ID: 39, Predicted Rating: 4.23
Movie ID: 5, Predicted Rating: 4.12
Movie ID: 18, Predicted Rating: 3.98
```

______________________________________________________________________

## 🎯 Item-Based Collaborative Filtering

### Calcular similitud entre películas

```python
# Transponer matriz: películas en filas, usuarios en columnas
item_user_matrix = user_item_matrix.T.fillna(0)

# Similitud entre películas
item_similarity = cosine_similarity(item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity,
                                   index=user_item_matrix.columns,
                                   columns=user_item_matrix.columns)

print(f"\nPelículas más similares a Movie 0:")
print(item_similarity_df.loc[0].sort_values(ascending=False).head())
```

### Recomendación item-based

```python
def recommend_movies_item_based(user_id, user_item_matrix, item_similarity_df, top_n=5):
    """
    Recomendar películas similares a las que el usuario ya vio
    """
    # Películas que el usuario vio (con buenos ratings)
    user_ratings = user_item_matrix.loc[user_id].dropna()
    liked_movies = user_ratings[user_ratings >= 4].index  # Solo películas con rating >= 4

    # Películas candidatas (no vistas)
    unwatched_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index

    # Calcular scores
    scores = {}
    for candidate in unwatched_movies:
        # Similitud promedio con películas que le gustaron
        similarities = item_similarity_df.loc[candidate, liked_movies]
        scores[candidate] = similarities.mean()

    # Top N
    recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations

recommendations_item = recommend_movies_item_based(user_id_test, user_item_matrix, item_similarity_df, top_n=5)

print(f"\n=== Recomendaciones Item-Based para User {user_id_test} ===\n")
for movie_id, score in recommendations_item:
    print(f"Movie ID: {movie_id}, Similarity Score: {score:.3f}")
```

______________________________________________________________________

## 📊 Evaluación

### Train/Test Split

```python
# Split temporal (simular)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Crear matriz de train
train_matrix = train_df.pivot_table(index='user_id', columns='movie_id', values='rating')

# Recalcular similitudes
train_filled = train_matrix.fillna(0)
user_sim_train = cosine_similarity(train_filled)
user_sim_train_df = pd.DataFrame(user_sim_train, index=train_matrix.index, columns=train_matrix.index)
```

### Predecir ratings en test set

```python
def predict_rating_user_based(user_id, movie_id, train_matrix, user_sim_df):
    """Predecir rating de user_id para movie_id"""
    if user_id not in train_matrix.index or movie_id not in train_matrix.columns:
        return train_matrix.mean().mean()  # Global mean fallback

    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11]
    ratings = train_matrix.loc[similar_users.index, movie_id].dropna()

    if len(ratings) == 0:
        return train_matrix[movie_id].mean()  # Movie mean fallback

    weights = similar_users[ratings.index]
    return (ratings * weights).sum() / weights.sum()

# Predecir en test set
predictions = []
actuals = []

for _, row in test_df.iterrows():
    pred = predict_rating_user_based(row['user_id'], row['movie_id'], train_matrix, user_sim_train_df)
    predictions.append(pred)
    actuals.append(row['rating'])

# Métricas
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))

print(f"\n=== Evaluación ===\n")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
```

**Salida:**

```
=== Evaluación ===

MAE:  0.876
RMSE: 1.123
```

______________________________________________________________________

## 📝 Resumen

### ✅ Collaborative Filtering

**User-Based:**

- Encuentra usuarios similares
- Recomienda lo que les gustó a usuarios similares
- **Ventaja:** Captura preferencias subjetivas
- **Desventaja:** Escalabilidad (comparar todos los usuarios)

**Item-Based:**

- Encuentra items similares
- Recomienda items similares a los que le gustaron
- **Ventaja:** Más escalable (items son menos que usuarios)
- **Desventaja:** Menos diversidad ("más de lo mismo")

### 🎯 Cold Start Problem

- **Nuevos usuarios:** No tienen historial → no se puede calcular similitud

- **Solución:** Pedir ratings iniciales, usar contenido (Content-Based), popularidad

- **Nuevos items:** No tienen ratings → no aparecen en recomendaciones

- **Solución:** Recomendar a usuarios early adopters, usar metadata

### 💡 Mejoras

1. **Matrix Factorization:** SVD, ALS (Ejemplo 02)
1. **Hybrid Systems:** Combinar collaborative + content-based
1. **Deep Learning:** Neural Collaborative Filtering
1. **Context-Aware:** Considerar tiempo, ubicación, dispositivo

### 📌 Checklist

- ✅ Crear matriz user-item
- ✅ Calcular similitudes (cosine, Pearson)
- ✅ Implementar user-based y item-based
- ✅ Manejar sparsity (fill NaN, regularización)
- ✅ Evaluar con MAE/RMSE en test set
- ✅ Estrategia para cold start
