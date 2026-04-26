# Practice 02 — Content-Based and Hybrid Systems

## 🎯 Objectives

- Implement content-based filtering
- Wear TF-IDF for features
- Combiner collaborative + content-based
- Evaluate trade-offs

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Content-Based with TF-IDF

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

# Similarity entre items
item_similarity = cosine_similarity(tfidf_matrix)

print("Item Similarity (Content-Based):")
print(pd.DataFrame(item_similarity, index=movies['title'], columns=movies['title']))

# Recommend basado en perfil user
def recommend_content_based(user_liked_items, top_n=3):
    # Promedio de features de items que le gustaron
    user_profile = tfidf_matrix[user_liked_items].mean(axis=0)

    # Similarity con todos los items
    scores = cosine_similarity(user_profile, tfidf_matrix).flatten()

    # Top N excluyendo items ya vistos
    scores[user_liked_items] = -1
    top_indices = scores.argsort()[-top_n:][::-1]

    return movies.iloc[top_indices]['title'].values

recs = recommend_content_based([0, 2])  # User likes Movie A y C
print(f"\\nRecommendations: {recs}")
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Propuestos

### Exercise 2.1: Hybrid Weighted

**Statement:**
Combine CF and content-based:

```
score = alpha * CF_score + (1-alpha) * CB_score
```

Find optimal alpha with grid search.

### Exercise 2.2: Feature Engineering

**Statement:**
Extract features from metadata:

- Director, actors, year
- Combine with embeddings
- Compare performance

### Exercise 2.3: Neural Collaborative Filtering

**Statement:**
Implement NCF with PyTorch:

- User/item embeddings
- MLP layers
- Binary cross-entropy loss

### Exercise 2.4: A/B Testing

**Statement:**
Simula A/B test:

- Control: popularity-based
- Treatment: hybrid recommender
- Measure CTR and engagement

### Exercise 2.5: Serendipity

**Statement:**
Balance accuracy and serendipity:

- Add non-obvious items
- Measure surprise + relevance
- Find optimal trade-off

______________________________________________________________________

## ✅ Checklist

- [ ] Content-based filtering with TF-IDF
- [ ] Hybrid recommenders
- [ ] Neural CF
- [ ] A/B testing
- [ ] Balance accuracy/diversity/serendipity

______________________________________________________________________

## 📚 Resources

- [Deep Learning for RecSys](https://deeplearning.neuromatch.io/)
- [Microsoft Recommenders](https://github.com/microsoft/recommenders)
