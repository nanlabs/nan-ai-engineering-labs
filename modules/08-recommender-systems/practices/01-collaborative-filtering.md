# Practice 01 — Collaborative Filtering

## 🎯 Objectives

- Implement user-based CF
- Implement item-based CF
- Matrix factorization with SVD
- Evaluate yourself with RMSE and MAE

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: User-Based CF

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Matrix user-item (ratings)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

users = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
items = ['Movie1', 'Movie2', 'Movie3', 'Movie4']

df = pd.DataFrame(ratings, index=users, columns=items)

# Similarity entre users
user_similarity = cosine_similarity(df.fillna(0))
user_sim_df = pd.DataFrame(user_similarity, index=users, columns=users)

print("User Similarity Matrix:")
print(user_sim_df)

# Predict rating
def predict_rating(user_idx, item_idx, k=2):
    # Top k users similar que rated item
    item_ratings = df.iloc[:, item_idx]
    similar_users = user_sim_df.iloc[user_idx].drop(users[user_idx])

    # Usuarios que rated este item
    rated_mask = item_ratings > 0
    similar_users = similar_users[rated_mask]

    top_k = similar_users.nlargest(k)

    if len(top_k) == 0:
        return df.mean()[item_idx]

    weighted_sum = sum(top_k * item_ratings[top_k.index])
    similarity_sum = top_k.sum()

    return weighted_sum / similarity_sum if similarity_sum > 0 else df.mean()[item_idx]

# Predict Alice's rating para Movie3
pred = predict_rating(0, 2, k=2)
print(f"\\nPredicted rating: {pred:.2f}")
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Item-Based CF

**Statement:**
Implement item-based collaborative filtering:

- Calculate similarity entre items
- Predict based on items similar ratings

### Exercise 2.2: SVD Matrix Factorization

**Statement:**
Use SVD for matrix factorization:

```python
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(ratings_matrix, k=2)
```

### Exercise 2.3: ALS Optimization

**Statement:**
Implement Alternating Least Squares:

- Optimiza user factors
- Optimiza item factors
- Itera hasta convergencia

### Exercise 2.4: Cold Start Handling

**Statement:**
Maneja nuevos users:

- Popularity-based recommendations
- Feature-based (demographics)
- Hybrid approach

### Exercise 2.5: Diversity Metrics

**Statement:**
Evaluate diversity of recommendations:

- Intra-list diversity (ILD)
- Coverage (% items recommended)
- Novelty (popularity average inversa)

______________________________________________________________________

## ✅ Checklist

- [ ] User-based CF
- [ ] Item-based CF
- [ ] Matrix factorization
- [ ] Cold start solutions
- [ ] RMSE and MAE evaluation

______________________________________________________________________

## 📚 Resources

- [Surprise Library](http://surpriselib.com/)
- [RecSys Papers](https://github.com/hongleizhang/RSPapers)
