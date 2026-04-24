# Práctica 02 — Content-Based y Hybrid Systems

## 🎯 Objetivos

- Implementar content-based filtering
- Usar TF-IDF para features
- Combinar collaborative + content-based
- Evaluar trade-offs

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Content-Based con TF-IDF

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Item profiles
movies = pd.DataFrame({
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'genres': ['action adventure', 'romance drama', 'action thriller', 'drama romance']
})

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Similitud entre items
item_similarity = cosine_similarity(tfidf_matrix)

print("Item Similarity (Content-Based):")
print(pd.DataFrame(item_similarity, index=movies['title'], columns=movies['title']))

# Recomendar basado en perfil usuario
def recommend_content_based(user_liked_items, top_n=3):
    # Promedio de features de items que le gustaron
    user_profile = tfidf_matrix[user_liked_items].mean(axis=0)

    # Similitud con todos los items
    scores = cosine_similarity(user_profile, tfidf_matrix).flatten()

    # Top N excluyendo items ya vistos
    scores[user_liked_items] = -1
    top_indices = scores.argsort()[-top_n:][::-1]

    return movies.iloc[top_indices]['title'].values

recs = recommend_content_based([0, 2])  # Usuario likes Movie A y C
print(f"\\nRecommendations: {recs}")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Hybrid Weighted

**Enunciado:**
Combina CF y content-based:

```
score = alpha * CF_score + (1-alpha) * CB_score
```

Encuentra alpha óptimo con grid search.

### Ejercicio 2.2: Feature Engineering

**Enunciado:**
Extrae features de metadata:

- Director, actors, year
- Combine con embeddings
- Compara performance

### Ejercicio 2.3: Neural Collaborative Filtering

**Enunciado:**
Implementa NCF con PyTorch:

- User/item embeddings
- MLP layers
- Binary cross-entropy loss

### Ejercicio 2.4: A/B Testing

**Enunciado:**
Simula A/B test:

- Control: popularity-based
- Treatment: hybrid recommender
- Mide CTR y engagement

### Ejercicio 2.5: Serendipity

**Enunciado:**
Balancea accuracy y serendipity:

- Añade items no obvios
- Mide surprise + relevance
- Encuentra trade-off óptimo

______________________________________________________________________

## ✅ Checklist

- [ ] Content-based filtering con TF-IDF
- [ ] Hybrid recommenders
- [ ] Neural CF
- [ ] A/B testing
- [ ] Balance accuracy/diversity/serendipity

______________________________________________________________________

## 📚 Recursos

- [Deep Learning for RecSys](https://deeplearning.neuromatch.io/)
- [Microsoft Recommenders](https://github.com/microsoft/recommenders)
