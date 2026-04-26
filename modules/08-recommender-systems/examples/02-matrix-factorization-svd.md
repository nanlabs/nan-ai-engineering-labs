# Example 02 — Matrix Factorization with SVD (Singular Value Decomposition)

## Context

Matrix Factorization decomposes the user-item matrix into lower-dimensional matrices to capture latent factors (genres, favorite actors, etc.). More efficient than classic collaborative filtering.

## Objective

Build recommendation system using SVD (used by Netflix Prize winners).

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

## 📦 Generate Data

```python
# Similar ratings
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

## 🏗️ Matrix Factorization with SVD

### Create matrix user-item

```python
user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

print(f"Matrix shape: {user_item_matrix.shape}")  # (500, 200)
```

### Apply SVD

```python
# SVD: R ≈ U × Σ × V^T
# R: matrix original (m×n)
# U: user factors (m×k)
# Σ: values singulares (k)
# V^T: item factors (k×n)
# k: number de factors latentes

k = 20  # Factors latentes (hyperparameter)

# Centrar data (restar media de each user)
user_ratings_mean = user_item_matrix.mean(axis=1)
matrix_centered = user_item_matrix.sub(user_ratings_mean, axis=0)

# SVD (truncada)
U, sigma, Vt = svds(matrix_centered.values, k=k)

# Convert sigma a matrix diagonal
sigma_diag = np.diag(sigma)

# Reconstruir matrix predicha
predictions_centered = np.dot(np.dot(U, sigma_diag), Vt)

# Add media de vuelta
predictions = predictions_centered + user_ratings_mean.values.reshape(-1, 1)

predictions_df = pd.DataFrame(predictions,
                              index=user_item_matrix.index,
                              columns=user_item_matrix.columns)

print(f"\nMatriz predicha shape: {predictions_df.shape}")
print(f"Example de prediction para user 0, movie 10: {predictions_df.loc[0, 10]:.2f}")
```

**Output:**

```
Matrix predicha shape: (500, 200)
Example de prediction para user 0, movie 10: 3.45
```

______________________________________________________________________

## 🎯 Recommendation function

```python
def recommend_movies_svd(user_id, predictions_df, user_item_matrix, top_n=5):
    """
    Recommend movies using predictions de SVD
    """
    # Ratings predichos para el user
    user_predictions = predictions_df.loc[user_id]

    # Movies ya vistas
    user_ratings = user_item_matrix.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0].index

    # Filtrar movies no vistas
    unwatched_predictions = user_predictions.drop(watched_movies)

    # Top N
    top_recommendations = unwatched_predictions.sort_values(ascending=False).head(top_n)

    return list(zip(top_recommendations.index, top_recommendations.values))

# Recommend para user 0
recommendations = recommend_movies_svd(0, predictions_df, user_item_matrix, top_n=5)

print("\n=== Recommendations SVD ===\n")
for movie_id, predicted_rating in recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {predicted_rating:.2f}")
```

______________________________________________________________________

## 📊 Evaluation

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Matrix de train
train_matrix = train_df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Reindex para tener mismas dimensions que la original
train_matrix = train_matrix.reindex(index=user_item_matrix.index,
                                      columns=user_item_matrix.columns,
                                      fill_value=0)

# Apply SVD en train
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

### Calculate Metrics

```python
# Obtener predictions para test set
test_predictions = []
test_actuals = []

for _, row in test_df.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']

    if user_id in predictions_train_df.index and movie_id in predictions_train_df.columns:
        pred = predictions_train_df.loc[user_id, movie_id]
        test_predictions.append(pred)
        test_actuals.append(row['rating'])

# Metrics
mae = mean_absolute_error(test_actuals, test_predictions)
rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))

print(f"\n=== Evaluation en Test Set ===\n")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
```

**Output:**

```
=== Evaluation en Test Set ===

MAE:  0.756
RMSE: 0.987
```

______________________________________________________________________

## 🔍 Analysis of latent factors

### Visualize factors

```python
# U contiene factors de users
# Cada user se represents como vector de k dimensions

# Ver primer user
user_0_factors = U[0]
print(f"User 0 latent factors:\n{user_0_factors}")

# Visualize distribution de un factor
plt.figure(figsize=(10, 5))
plt.hist(U[:, 0], bins=30, alpha=0.7, label='Factor 1')
plt.hist(U[:, 1], bins=30, alpha=0.7, label='Factor 2')
plt.xlabel('Factor Value')
plt.ylabel('Frequency')
plt.title('Distribution de Factors Latentes')
plt.legend()
plt.show()
```

### Similar users in latent space

```python
from sklearn.metrics.pairwise import cosine_similarity

# Similarity entre users en espacio de factors latentes
user_similarity_latent = cosine_similarity(U)

user_id_target = 0
similar_users_idx = user_similarity_latent[user_id_target].argsort()[-6:-1][::-1]

print(f"\nUsuarios más similar a User {user_id_target}:")
for idx in similar_users_idx:
    print(f"User {idx}: Similarity = {user_similarity_latent[user_id_target, idx]:.3f}")
```

______________________________________________________________________

## 📈 Tuning: Optimal number of factors (k)

```python
k_values = [5, 10, 20, 30, 50, 100]
rmse_values = []

for k in k_values:
    # SVD con k factors
    U_k, sigma_k, Vt_k = svds(train_centered.values, k=k)
    sigma_diag_k = np.diag(sigma_k)

    predictions_k = np.dot(np.dot(U_k, sigma_diag_k), Vt_k) + train_mean.values.reshape(-1, 1)
    predictions_k_df = pd.DataFrame(predictions_k,
                                     index=train_matrix.index,
                                     columns=train_matrix.columns)

    # Evaluate
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
plt.xlabel('Number de Factors Latentes (k)')
plt.ylabel('RMSE')
plt.title('RMSE vs Number de Factors')
plt.grid(alpha=0.3)
plt.show()

print(f"\nMejor k: {k_values[np.argmin(rmse_values)]} (RMSE = {min(rmse_values):.4f})")
```

______________________________________________________________________

## 📝 Summary

### ✅ Matrix Factorization vs Collaborative Filtering

| Appearance | Collaborative Filtering | Matrix Factorization (SVD) |
| -------------------------- | ------------------------- | -------------------------- |
| **Computational complexity** | O(n²) users | O(k × n) factors |
| **Scalability** | Poor (large n users) | Excellent |
| **Sparsity** | Sensitive | Drive well |
| **Interpretability** | High (direct similarity) | Average (latent factors) |
| **accuracy** | Good | **Best** |

### 🎯 Advantages of SVD

1. **Dimensionality reduction:** 500 users × 200 movies → 500 × 20 + 20 × 200 = 14,000 parameters (vs 100,000 original)
1. **Captures latent patterns:** Genres, actors, decades automatically
1. **Handles sparsity:** Fill NaN implicitly
1. **Escalable:** Puede actualizar factors incrementalmente

### 💡 Alternatives and improvements

1. **ALS (Alternating Least Squares):**

   - Optimize U and V iteratively
- Most used in production (Spark MLlib)

1. **Neural Matrix Factorization:**

   - Replace product point U × V with Neural network
   - Captura relaciones no lineales

1. **Factorization Machines:**

   - Generalize MF to include additional features
- Useful for contextual recommendations

### 🚫 Limitations

- ❌ Cold start: Nuevos users/items sin factors
- ❌ Popularidad bias: Tiende a recomendar items populares
- ❌ No considera context temporal

### 📌 Checklist SVD

- ✅ Center matrix (subtract average per user)
- ✅ Choose appropriate k (5-50 typically)
- ✅ Evaluate yourself with cross-validation
- ✅ Regularize to prevent overfitting (in ALS variants)
- ✅ Update Model periodically (new Data)
- ✅ Combiner with content-based for cold start
