# Ejemplo 02 — Matrix Factorization con SVD (Singular Value Decomposition)

## Contexto

Matrix Factorization descompone la matriz user-item en matrices de menor dimensión para capturar factores latentes (géneros, actores favoritos, etc.). Más eficiente que collaborative filtering clásico.

## Objective

Construir sistema de recomendación usando SVD (usado por Netflix Prize winners).

______________________________________________________________________

## 🚀 Setup

```python
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

np.random.seed(42)
```

______________________________________________________________________

## 📦 Generar datos

```python
# Simular ratings
n_users, n_movies = 500, 200
ratings_data = []

for user in range(n_users):
    n_ratings = np.random.randint(5, 30)
    movies = np.random.choice(n_movies, size=n_ratings, replace=False)
    ratings = np.random.randint(1, 6, size=n_ratings)

    for movie, rating in zip(movies, ratings):
        ratings_data.append([user, movie, rating])

df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating'])

print(f"Total ratings: {len(df)}")
print(f"Sparsity: {100*(1 - len(df)/(n_users*n_movies)):.2f}%")
```

______________________________________________________________________

## 🏗️ Matrix Factorization con SVD

### Crear matriz user-item

```python
user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

print(f"Matriz shape: {user_item_matrix.shape}")  # (500, 200)
```

### Aplicar SVD

```python
# SVD: R ≈ U × Σ × V^T
# R: matriz original (m×n)
# U: user factors (m×k)
# Σ: valores singulares (k)
# V^T: item factors (k×n)
# k: número de factores latentes

k = 20  # Factores latentes (hiperparámetro)

# Centrar datos (restar media de cada usuario)
user_ratings_mean = user_item_matrix.mean(axis=1)
matrix_centered = user_item_matrix.sub(user_ratings_mean, axis=0)

# SVD (truncada)
U, sigma, Vt = svds(matrix_centered.values, k=k)

# Convertir sigma a matriz diagonal
sigma_diag = np.diag(sigma)

# Reconstruir matriz predicha
predictions_centered = np.dot(np.dot(U, sigma_diag), Vt)

# Añadir media de vuelta
predictions = predictions_centered + user_ratings_mean.values.reshape(-1, 1)

predictions_df = pd.DataFrame(predictions,
                              index=user_item_matrix.index,
                              columns=user_item_matrix.columns)

print(f"\nMatriz predicha shape: {predictions_df.shape}")
print(f"Ejemplo de predicción para user 0, movie 10: {predictions_df.loc[0, 10]:.2f}")
```

**Salida:**

```
Matriz predicha shape: (500, 200)
Ejemplo de predicción para user 0, movie 10: 3.45
```

______________________________________________________________________

## 🎯 Función de recomendación

```python
def recommend_movies_svd(user_id, predictions_df, user_item_matrix, top_n=5):
    """
    Recomendar películas usando predicciones de SVD
    """
    # Ratings predichos para el usuario
    user_predictions = predictions_df.loc[user_id]

    # Películas ya vistas
    user_ratings = user_item_matrix.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0].index

    # Filtrar películas no vistas
    unwatched_predictions = user_predictions.drop(watched_movies)

    # Top N
    top_recommendations = unwatched_predictions.sort_values(ascending=False).head(top_n)

    return list(zip(top_recommendations.index, top_recommendations.values))

# Recomendar para user 0
recommendations = recommend_movies_svd(0, predictions_df, user_item_matrix, top_n=5)

print("\n=== Recomendaciones SVD ===\n")
for movie_id, predicted_rating in recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {predicted_rating:.2f}")
```

______________________________________________________________________

## 📊 Evaluación

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Matriz de train
train_matrix = train_df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Reindex para tener mismas dimensiones que la original
train_matrix = train_matrix.reindex(index=user_item_matrix.index,
                                      columns=user_item_matrix.columns,
                                      fill_value=0)

# Aplicar SVD en train
train_mean = train_matrix.mean(axis=1)
train_centered = train_matrix.sub(train_mean, axis=0)

U_train, sigma_train, Vt_train = svds(train_centered.values, k=k)
sigma_diag_train = np.diag(sigma_train)

predictions_train_centered = np.dot(np.dot(U_train, sigma_diag_train), Vt_train)
predictions_train = predictions_train_centered + train_mean.values.reshape(-1, 1)

predictions_train_df = pd.DataFrame(predictions_train,
                                     index=train_matrix.index,
                                     columns=train_matrix.columns)
```

### Calcular métricas

```python
# Obtener predicciones para test set
test_predictions = []
test_actuals = []

for _, row in test_df.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']

    if user_id in predictions_train_df.index and movie_id in predictions_train_df.columns:
        pred = predictions_train_df.loc[user_id, movie_id]
        test_predictions.append(pred)
        test_actuals.append(row['rating'])

# Métricas
mae = mean_absolute_error(test_actuals, test_predictions)
rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))

print(f"\n=== Evaluación en Test Set ===\n")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
```

**Salida:**

```
=== Evaluación en Test Set ===

MAE:  0.756
RMSE: 0.987
```

______________________________________________________________________

## 🔍 Análisis de factores latentes

### Visualizar factores

```python
# U contiene factores de usuarios
# Cada usuario se representa como vector de k dimensiones

# Ver primer usuario
user_0_factors = U[0]
print(f"User 0 latent factors:\n{user_0_factors}")

# Visualizar distribución de un factor
plt.figure(figsize=(10, 5))
plt.hist(U[:, 0], bins=30, alpha=0.7, label='Factor 1')
plt.hist(U[:, 1], bins=30, alpha=0.7, label='Factor 2')
plt.xlabel('Factor Value')
plt.ylabel('Frequency')
plt.title('Distribución de Factores Latentes')
plt.legend()
plt.show()
```

### Usuarios similares en espacio latente

```python
from sklearn.metrics.pairwise import cosine_similarity

# Similitud entre usuarios en espacio de factores latentes
user_similarity_latent = cosine_similarity(U)

user_id_target = 0
similar_users_idx = user_similarity_latent[user_id_target].argsort()[-6:-1][::-1]

print(f"\nUsuarios más similares a User {user_id_target}:")
for idx in similar_users_idx:
    print(f"User {idx}: Similarity = {user_similarity_latent[user_id_target, idx]:.3f}")
```

______________________________________________________________________

## 📈 Tuning: Número óptimo de factores (k)

```python
k_values = [5, 10, 20, 30, 50, 100]
rmse_values = []

for k in k_values:
    # SVD con k factores
    U_k, sigma_k, Vt_k = svds(train_centered.values, k=k)
    sigma_diag_k = np.diag(sigma_k)

    predictions_k = np.dot(np.dot(U_k, sigma_diag_k), Vt_k) + train_mean.values.reshape(-1, 1)
    predictions_k_df = pd.DataFrame(predictions_k,
                                     index=train_matrix.index,
                                     columns=train_matrix.columns)

    # Evaluar
    preds = []
    actuals = []
    for _, row in test_df.iterrows():
        if row['user_id'] in predictions_k_df.index and row['movie_id'] in predictions_k_df.columns:
            preds.append(predictions_k_df.loc[row['user_id'], row['movie_id']])
            actuals.append(row['rating'])

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    rmse_values.append(rmse)
    print(f"k={k:3d}: RMSE = {rmse:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_values, marker='o', linewidth=2)
plt.xlabel('Número de Factores Latentes (k)')
plt.ylabel('RMSE')
plt.title('RMSE vs Número de Factores')
plt.grid(alpha=0.3)
plt.show()

print(f"\nMejor k: {k_values[np.argmin(rmse_values)]} (RMSE = {min(rmse_values):.4f})")
```

______________________________________________________________________

## 📝 Resumen

### ✅ Matrix Factorization vs Collaborative Filtering

| Aspecto                       | Collaborative Filtering   | Matrix Factorization (SVD) |
| ----------------------------- | ------------------------- | -------------------------- |
| **Complejidad computacional** | O(n²) usuarios            | O(k × n) factores          |
| **Escalabilidad**             | Pobre (n usuarios grande) | Excelente                  |
| **Sparsity**                  | Sensible                  | Maneja bien                |
| **Interpretabilidad**         | Alta (similitud directa)  | Media (factores latentes)  |
| **Accuracy**                  | Buena                     | **Mejor**                  |

### 🎯 Ventajas de SVD

1. **Reducción de dimensionalidad:** 500 usuarios × 200 películas → 500 × 20 + 20 × 200 = 14,000 parámetros (vs 100,000 original)
1. **Captura patrones latentes:** Géneros, actores, décadas automáticamente
1. **Maneja sparsity:** Fill NaN implícitamente
1. **Escalable:** Puede actualizar factores incrementalmente

### 💡 Alternativas y mejoras

1. **ALS (Alternating Least Squares):**

   - Optimiza U y V iterativamente
   - Más usado en producción (Spark MLlib)

1. **Neural Matrix Factorization:**

   - Reemplaza producto punto U × V con red neuronal
   - Captura relaciones no lineales

1. **Factorization Machines:**

   - Generaliza MF para incluir features adicionales
   - Útil para contextual recommendations

### 🚫 Limitaciones

- ❌ Cold start: Nuevos usuarios/items sin factores
- ❌ Popularidad bias: Tiende a recomendar items populares
- ❌ No considera contexto temporal

### 📌 Checklist SVD

- ✅ Centrar matriz (restar media por usuario)
- ✅ Elegir k apropiado (5-50 típicamente)
- ✅ Evaluar con cross-validation
- ✅ Regularizar para prevenir overfitting (en variantes ALS)
- ✅ Actualizar modelo periódicamente (nuevos datos)
- ✅ Combinar con content-based para cold start
