# Example 1 — Vectors and product point

## Objective

Represent vectors using NumPy and calculate basic similarity between them using the dot product.

## Concepts previous

- A **vector** is an ordered list of numbers.
- The **dot product** of two vectors measures their directional similarity.
- Formula: `a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ`
- If the dot product is high, the vectors point in similar directions.

## Step by step implementation

### 1. Import NumPy

```python
import numpy as np
```

### 2. Create dos vectors

```python
# Vector que represents features de un product A
producto_a = np.array([4.5, 3.2, 5.0])  # [price, quality, popularity]

# Vector que represents features de un product B
producto_b = np.array([4.7, 3.0, 4.8])
```

### 3. Calculate the dot product

```python
similarity = np.dot(producto_a, producto_b)
print(f"Similarity entre products: {similarity:.2f}")
```

**Output expected:**

```
Similarity entre products: 54.71
```

### 4. Calculate the norm (length) of each vector

```python
norma_a = np.linalg.norm(producto_a)
norma_b = np.linalg.norm(producto_b)

print(f"Norma de product A: {norma_a:.2f}")
print(f"Norma de product B: {norma_b:.2f}")
```

**Output expected:**

```
Norma de product A: 7.35
Norma de product B: 7.24
```

### 5. Calculate similarity normalized (cosine)

```python
similitud_coseno = similarity / (norma_a * norma_b)
print(f"Similarity cosine: {similitud_coseno:.4f}")
```

**Output expected:**

```
Similarity cosine: 1.0281
```

> ⚠️ **Note:** The cosine similarity should be between -1 and 1. If it is greater than 1, there is a rounding error or the vectors are almost identical.

### 6. Interpretation

- A close value of 1 indicates that both products are very similar in their features.
- We use this in ML to measure similarity between text embeddings, Images, etc.

## Complete executable code

```python
import numpy as np

# Define vectors
producto_a = np.array([4.5, 3.2, 5.0])
producto_b = np.array([4.7, 3.0, 4.8])

# Production point
similarity = np.dot(producto_a, producto_b)
print(f"Similarity (product point): {similarity:.2f}")

# Norms
norma_a = np.linalg.norm(producto_a)
norma_b = np.linalg.norm(producto_b)
print(f"Norma A: {norma_a:.2f}, Norma B: {norma_b:.2f}")

# Similarity cosine
similitud_coseno = similarity / (norma_a * norma_b)
print(f"Similarity cosine: {similitud_coseno:.4f}")
```

## Errors common

- ❌ Forget that vectors must have the same dimension.
- ❌ Confuse dot product with element-by-element multiplication (`*`).
- ❌ Do not normalize before calculating cosine (use norms).

## Exercise proposed

Create two vectors that represent users based on their movie genre preferences:

```python
usuario_1 = np.array([5, 3, 0, 4])  # [action, comedy, drama, sci-fi]
usuario_2 = np.array([4, 2, 1, 5])
```

Calculate the cosine similarity between both users to see how similar their tastes are.
