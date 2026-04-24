# Práctica 01 — Collaborative Filtering

## 🎯 Objetivos

- Implementar user-based CF
- Implementar item-based CF
- Matrix factorization con SVD
- Evaluar con RMSE y MAE

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: User-Based CF

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Matriz user-item (ratings)
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

# Similitud entre usuarios
user_similarity = cosine_similarity(df.fillna(0))
user_sim_df = pd.DataFrame(user_similarity, index=users, columns=users)

print("User Similarity Matrix:")
print(user_sim_df)

# Predecir rating
def predict_rating(user_idx, item_idx, k=2):
    # Top k usuarios similares que rated item
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

# Predecir Alice's rating para Movie3
pred = predict_rating(0, 2, k=2)
print(f"\\nPredicted rating: {pred:.2f}")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Item-Based CF

**Enunciado:**
Implementa item-based collaborative filtering:

- Calcula similitud entre items
- Predice basado en items similares ratings

### Ejercicio 2.2: SVD Matrix Factorization

**Enunciado:**
Usa SVD para matrix factorization:

```python
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(ratings_matrix, k=2)
```

### Ejercicio 2.3: ALS Optimization

**Enunciado:**
Implementa Alternating Least Squares:

- Optimiza user factors
- Optimiza item factors
- Itera hasta convergencia

### Ejercicio 2.4: Cold Start Handling

**Enunciado:**
Maneja nuevos usuarios:

- Popularity-based recommendations
- Feature-based (demographics)
- Hybrid approach

### Ejercicio 2.5: Diversity Metrics

**Enunciado:**
Evalúa diversidad de recommendations:

- Intra-list diversity (ILD)
- Coverage (% items recomendados)
- Novelty (popularidad promedio inversa)

______________________________________________________________________

## ✅ Checklist

- [ ] User-based CF
- [ ] Item-based CF
- [ ] Matrix factorization
- [ ] Cold start solutions
- [ ] RMSE y MAE evaluation

______________________________________________________________________

## 📚 Recursos

- [Surprise Library](http://surpriselib.com/)
- [RecSys Papers](https://github.com/hongleizhang/RSPapers)
