# Theory — Recommender Systems

## Why this module matters

Recommendation systems drive billions of dollars in e-commerce (Amazon, Netflix, Spotify). Mastering these systems allows you to create personalized experiences that increase engagement, sales and user satisfaction.

______________________________________________________________________

## 1. What is a recommendation system?

**Recommendation System (RecSys):** Suggests **relevant items** (products, movies, songs, Content) to each user based on:

- Historical user behavior.
- User profile.
- Similar user behavior.
- Features of the items.

### Applications

- E-commerce: "Customers who bought this also bought..." (Amazon).
- Streaming: "Recommended for you" (Netflix, Spotify).
- Social networks: personalized feed (Facebook, Instagram).
- Search: Custom results (Google).
- Dating apps: suggested profiles (Tinder).

📹 **Videos recommended:**

1. [Recommender Systems Explained - IBM](https://www.youtube.com/watch?v=giIXNoiqO_U) - 10 min
1. [How Netflix Recommends - Vox](https://www.youtube.com/watch?v=wQABXV0_M5Y) - 8 min

______________________________________________________________________

## 2. Main approaches

### Popularity (Baseline)

Recommend most popular items globally.

**Advantage:** Simple, it always works.
**Disadvantage:** Not personalized, favors already popular items.

### Content-Based

Recommend items **similar** to those the user liked in the past.

**How ​​it works:**

1. Represent items with features (genre, author, keywords).
1. Calculate similarity between items.
1. Recommend items similar to those consumed by the user.

**Advantage:**

- It does not need Data from other users.
- Explain why you recommend something.

**Disadvantage:**

- Does not discover items outside the user's profile ("Filter bubble").
- Require quality features.

**Example:**
User watched horror movies → recommend more horror.

### Collaborative Filtering

Recommend based on **collective behavior patterns**.

**Types:**

#### User-Based Collaborative Filtering

- Find users similar to user Objective.
- Recommend items that those similar users consumed.

**Logic:** "Users similar to you also enjoyed...".

#### Item-Based Collaborative Filtering

- Find items similar to those the user consumed.
- Similarity based on collective consumption patterns.

**Logic:** "Users who liked X also liked Y."

**Advantage:**

- Discover non-obvious patterns.
- Does not need item features.

**Disadvantage:**

- Cold start: does not work with new users/items.
- Sparsity: very empty user-item matrix.

📹 **Videos recommended:**

1. [Collaborative Filtering - StatQuest](https://www.youtube.com/watch?v=h9gpufJFF-0) - 15 min
1. [Content-Based vs Collaborative - Krish Naik](https://www.youtube.com/watch?v=Eeg1DEeWUjA) - 20 min

______________________________________________________________________

## 3. User-item matrix

### Representation

```
           Ítem1  Ítem2  Ítem3  Ítem4
Usuario1     5      ?      3      ?
Usuario2     ?      4      ?      5
Usuario3     4      3      ?      ?
```

- **Rows:** users.
- **Columns:** items.
- **Values:** rating (1-5), purchase (0/1), clicks, viewing time.
- **?** = missing (user did not interact with item).

### Objective

Predict missing values ​​to recommend items with higher Prediction.

______________________________________________________________________

## 4. Matrix Factorization

**Idea:** Decompose user-item matrix into two low-dimensional matrices.

```
R (users × items) ≈ U (users × k) × V (k × items)
```

Where `k` is the number of latent factors (hidden features).

**Techniques:**

- **SVD (Singular Value Decomposition):** mathematical decomposition.
- **ALS (Alternating Least Squares):** iterative optimization (used in Apache Spark).
- **Deep Learning:** embeddings learned with neural networks.

📹 **Videos recommended:**

1. [Matrix Factorization - StatQuest](https://www.youtube.com/watch?v=ZspR5PZemcs) - 20 min
1. [SVD for Recommendations - 3Blue1Brown](https://www.youtube.com/watch?v=P5mlg91as1c) - 15 min

📚 **Resources written:**

- [Matrix Factorization Techniques - Netflix Prize](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)

______________________________________________________________________

## 5. Hybrid systems

Combine multiple approaches to take advantage of each one's strengths.

**Strategies:**

- **Weighted:** combine scores with weights.
- **Switching:** use one approach or another according to context.
- **Cascade:** filter with one approach, refine with another.
- **Feature Combinetion:** use Content + collaborative features in same Model.

**Example:** Netflix combines collaborative filtering + Content + popularity + temporal context.

______________________________________________________________________

## 6. Typical problems

### Cold Start

**Problem:** New user or item without history.

**Solutions:**

- **New user:** recommend popular items, ask initial preferences.
- **New item:** use content-based, initial promotion (exploration).

### Sparsity

**Problem:** User-item matrix very empty (most users do not interact with most items).

**Solutions:**

- Matrix factorization (reduces dimensionality).
- Use implicit data (clicks, views) in addition to explicit ratings.

### Popularity Bias

**Problem:** System always recommends popular items, ignores niche items.

**Solutions:**

- Intentional diversification.
- Re-ranking to balance popularity and relevance.

### Filter Bubble

**Problem:** User only sees Content similar to what they already consumed (echo chamber).

**Solutions:**

- Exploration vs Exploitation (bandits).
- Inject serendipity (recommend unexpected but potentially interesting items).

📹 **Videos recommended:**

1. [Cold Start Problem - Stanford CS 329S](https://www.youtube.com/watch?v=ZFCvvzvbVV4) - 20 min

______________________________________________________________________

## 7. Evaluation Metrics

### Offline metrics

#### Precision@K

Of the K items recommended, how many are relevant?

```
Precision@K = (# relevant en top-K) / K
```

#### recall@K

Of all the relevant items, how many did I capture in top-K?

```
Recall@K = (# relevant en top-K) / (total relevant)
```

#### MAP@K (Mean Average Precision)

Average precision at different values ​​of K.

#### NDCG (Normalized Discounted Cumulative Gain)

Consider order of recommendations: relevant items above are worth more.

#### Coverage

What % of the catalog is recommended at least once?

### Online metrics (A/B testing)

- Click-Through Rate (CTR).
- Conversion rate.
- Time spent.
- Revenue per user.

**Important:** Offline metrics do not always correlate with business impact. Validate in production.

📹 **Videos recommended:**

1. [Recommendation Metrics - Google ML](https://www.youtube.com/watch?v=eZp4oQLtlPM) - 15 min

📚 **Resources written:**

- [Evaluation Metrics for Recommender Systems](https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093)

______________________________________________________________________

## 8. Deep Learning for recommendations

### Neural Collaborative Filtering (NCF)

Replace linear matrix factorization with neural networks.

**Advantage:** Captures non-linear interactions.

### Two-Tower Models

Dos neural networks:

- Tower 1: user embeddings.
- Tower 2: item embeddings.
- Similarity: product point o cosine similarity.

**Usage:** efficient search in large catalogs.

### Sequence Models (RNN, Transformers)

Model user history as a temporal sequence.

**Example:** Predict next product based on purchasing sequence.

**Models:** GRU4Rec, BERT4Rec, SASRec.

📹 **Videos recommended:**

1. [Deep Learning for RecSys - NVIDIA](https://www.youtube.com/watch?v=Kw5cU7lyYgs) - 40 min
1. [Neural Collaborative Filtering - Paper Explained](https://www.youtube.com/watch?v=rQTK3NmMPtE) - 20 min

📚 **Resources written:**

- [Neural Collaborative Filtering Paper](https://arxiv.org/abs/1708.05031)
- [Transformers4Rec (NVIDIA)](https://github.com/NVIDIA-Merlin/Transformers4Rec)

______________________________________________________________________

## 9. Context and bandits

### Contextual Recommendations

Incorporate context:

- Time of day, day of week.
- Device (mobile vs desktop).
- Geographic location.
- Current session (what the user did 5 minutes ago).

### Multi-Armed Bandits

Balance exploration (test new recommendations) vs exploitation (recommend what we know works).

**Algorithms:**

- ε-greedy.
- UCB (Upper Confidence Bound).
- Thompson Sampling.
- Contextual Bandits.

📹 **Videos recommended:**

1. [Multi-Armed Bandits - StatQuest](https://www.youtube.com/watch?v=e3L4VocZnnQ) - 12 min

______________________________________________________________________

## 10. Buenas Practices

- ✅ Start with popularity baseline.
- ✅ Try item-based collaborative filtering (simple and effective).
- ✅ Separate Data temporarily (not randomly) if time matters.
- ✅ Measure diversity and coverage, not just precision.
- ✅ Audit exposure biases (do not confuse clicking with preference).
- ✅ Implement A/B testing in production to validate real impact.
- ✅ Consider cold start from the initial design.
- ✅ Monitor distribution of recommendations (avoid filter bubble).
- ✅ Document trade-offs (relevance vs diversity vs novelty).

📚 **General resources:**

- [Recommender Systems Textbook (Free)](https://www.springer.com/gp/book/9783319296579)
- [Google RecSys Course](https://developers.google.com/machine-learning/recommendation)
- [Surprise Library (Python)](http://surpriselib.com/) - scikit-learn for RecSys

______________________________________________________________________

## Final comprehension checklist

Before moving to the next Module, you should be able to:

- ✅ Explain the difference between content-based and collaborative filtering.
- ✅ Describe what cold start is and propose solutions.
- ✅ Build user-item matrix and calculate similarities.
- ✅ Implement collaborative item-based filtering with cosine similarity.
- ✅ Justify choice of Metric @K according to business objectives.
- ✅ Identify popularity bias and propose diversification strategies.
- ✅ Explain trade-offs exploration vs exploitation.
- ✅ Design A/B experiment to validate recommendation system.

If you answered "yes" to all, you are ready for advanced recommendation systems and in production.
