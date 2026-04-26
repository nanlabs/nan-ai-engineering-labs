# Example 01 — Collaborative Filtering for Movie Recommendations

## Context

You learn **Collaborative Filtering** to build recommendation systems based on similar user behaviors.

## Objective

Recommend movies using user-item ratings.

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

## 📥 Load Data (MovieLens-style)

```python
# Similar ratings (en production: use MovieLens dataset)
n_users = 100
n_movies = 50
sparsity = 0.1  # 10% de interacciones

# Create matrix user-item sparse
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

**Output:**

```
Total ratings: 543
Users: 100
Movies: 50
Sparsity: 89.14%  👈 Matrix muy sparse (common en systems recommendation)

   user_id  movie_id  rating
0        0        42       3
1        0        15       5
2        0         8       2
```

______________________________________________________________________

## 🏗️ User-Based Collaborative Filtering

### Create matrix user-item

```python
# Pivot table: users en rows, movies en columns
user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')

print(f"Matrix shape: {user_item_matrix.shape}")  # (100, 50)
print(f"\n{user_item_matrix.head()}")
```

### Calculate similarity entre users

```python
# Fill NaN con 0 para calculation de similarity
user_item_filled = user_item_matrix.fillna(0)

# Cosine similarity entre users
user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity,
                                   index=user_item_matrix.index,
                                   columns=user_item_matrix.index)

print(f"\nSimilitud entre user 0 y otros:")
print(user_similarity_df.loc[0].sort_values(ascending=False).head())
```

**Output:**

```
user_id
0    1.000000  👈 Consigo same
23   0.893456  👈 Usuario más similar
45   0.856712
12   0.834521
89   0.812345
```

### Recommendation function

```python
def recommend_movies_user_based(user_id, user_item_matrix, user_similarity_df, top_n=5):
    """
    Recommend movies based en users similar
    """
    # Obtener users similar (excluyendo el same user)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]  # Top 10

    # Movies que el user NO ha visto
    user_ratings = user_item_matrix.loc[user_id]
    unwatched_movies = user_ratings[user_ratings.isna()].index

    # Predict ratings para movies no vistas
    predictions = {}
    for movie in unwatched_movies:
        # Weighted average de ratings de users similar
        similar_users_ratings = user_item_matrix.loc[similar_users.index, movie]
        valid_ratings = similar_users_ratings.dropna()

        if len(valid_ratings) > 0:
            # Pesos = similitudes de users que vieron la movie
            weights = similar_users[valid_ratings.index]
            weighted_rating = (valid_ratings * weights).sum() / weights.sum()
            predictions[movie] = weighted_rating

    # Ordenar por rating predicho
    recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return recommendations

# Example: recomendar para user 0
user_id_test = 0
recommendations = recommend_movies_user_based(user_id_test, user_item_matrix, user_similarity_df, top_n=5)

print(f"\n=== Recomendaciones para User {user_id_test} ===\n")
for movie_id, predicted_rating in recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {predicted_rating:.2f}")
```

**Output:**

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

### Calculate similarity between movies

```python
# Transponer matrix: movies en rows, users en columns
item_user_matrix = user_item_matrix.T.fillna(0)

# Similarity entre movies
item_similarity = cosine_similarity(item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity,
                                   index=user_item_matrix.columns,
                                   columns=user_item_matrix.columns)

print(f"\nPelículas más similar a Movie 0:")
print(item_similarity_df.loc[0].sort_values(ascending=False).head())
```

### Item-based recommendation

```python
def recommend_movies_item_based(user_id, user_item_matrix, item_similarity_df, top_n=5):
    """
    Recommend movies similar a las que el user ya vio
    """
    # Movies que el user vio (con buenos ratings)
    user_ratings = user_item_matrix.loc[user_id].dropna()
    liked_movies = user_ratings[user_ratings >= 4].index  # Solo movies con rating >= 4

    # Movies candidates (no vistas)
    unwatched_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index

    # Calculate scores
    scores = {}
    for candidate in unwatched_movies:
        # Similarity average con movies que le gustaron
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

## 📊 Evaluation

### Train/Test Split

```python
# Split temporal (similar)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create matrix de train
train_matrix = train_df.pivot_table(index='user_id', columns='movie_id', values='rating')

# Recalcular similitudes
train_filled = train_matrix.fillna(0)
user_sim_train = cosine_similarity(train_filled)
user_sim_train_df = pd.DataFrame(user_sim_train, index=train_matrix.index, columns=train_matrix.index)
```

### Predict ratings in test set

```python
def predict_rating_user_based(user_id, movie_id, train_matrix, user_sim_df):
    """Predict rating de user_id para movie_id"""
    if user_id not in train_matrix.index or movie_id not in train_matrix.columns:
        return train_matrix.mean().mean()  # Global mean fallback

    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:11]
    ratings = train_matrix.loc[similar_users.index, movie_id].dropna()

    if len(ratings) == 0:
        return train_matrix[movie_id].mean()  # Movie mean fallback

    weights = similar_users[ratings.index]
    return (ratings * weights).sum() / weights.sum()

# Predict en test set
predictions = []
actuals = []

for _, row in test_df.iterrows():
    pred = predict_rating_user_based(row['user_id'], row['movie_id'], train_matrix, user_sim_train_df)
    predictions.append(pred)
    actuals.append(row['rating'])

# Metrics
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))

print(f"\n=== Evaluation ===\n")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
```

**Output:**

```
=== Evaluation ===

MAE:  0.876
RMSE: 1.123
```

______________________________________________________________________

## 📝 Summary

### ✅ Collaborative Filtering

**User-Based:**

- Find similar users
- Recommend what similar users liked
- **Advantage:** Captures subjective preferences
- **Disadvantage:** Scalability (compare all users)

**Item-Based:**

- Find similar items
- Recommends items similar to those you liked
- **Advantage:** More scalable (items are less than users)
- **Disadvantage:** Less diversity ("more of the same")

### 🎯 Cold Start Problem

- **New users:** They do not have history → cannot calculate similarity

- **Solution:** Ask for initial ratings, use Content (Content-Based), popularity

- **New items:** They have no ratings → they do not appear in recommendations

- **Solution:** Recommend a users early adopters, use metadata

### 💡 Improvements

1. **Matrix Factorization:** SVD, ALS (Example 02)
1. **Hybrid Systems:** Combine collaborative + content-based
1. **Deep Learning:** Neural Collaborative Filtering
1. **Context-Aware:** Consider time, location, device

### 📌 Checklist

- ✅ Create matrix user-item
- ✅ Calculate similarities (cosine, Pearson)
- ✅ Implement user-based and item-based
- ✅ Handle sparsity (fill NaN, Regularization)
- ✅ Evaluate yourself with MAE/RMSE in test set
- ✅ Strategy for cold start
