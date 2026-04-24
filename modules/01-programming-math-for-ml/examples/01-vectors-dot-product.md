# Example 1 — Vectores y producto punto

## Objective

Represent vectors using NumPy and calculate basic similarity between them using the dot product.

## Concepts previos

- A **vector** is an ordered list of numbers.
- The **dot product** of two vectors measures their directional similarity.
- Formula: `a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ`
- If the dot product is high, the vectors point in similar directions.

## Step by step implementation

### 1. Importar NumPy

```python
import numpy as np
```

### 2. Crear dos vectores

```python
# Vector que representa características de un producto A
producto_a = np.array([4.5, 3.2, 5.0])  # [precio, calidad, popularidad]

# Vector que representa características de un producto B
producto_b = np.array([4.7, 3.0, 4.8])
```

### 3. Calculate the dot product

```python
similitud = np.dot(producto_a, producto_b)
print(f"Similitud entre productos: {similitud:.2f}")
```

**Salida esperada:**

```
Similitud entre productos: 54.71
```

### 4. Calculate the norm (length) of each vector

```python
norma_a = np.linalg.norm(producto_a)
norma_b = np.linalg.norm(producto_b)

print(f"Norma de producto A: {norma_a:.2f}")
print(f"Norma de producto B: {norma_b:.2f}")
```

**Salida esperada:**

```
Norma de producto A: 7.35
Norma de producto B: 7.24
```

### 5. Calcular similitud normalizada (coseno)

```python
similitud_coseno = similitud / (norma_a * norma_b)
print(f"Similitud coseno: {similitud_coseno:.4f}")
```

**Salida esperada:**

```
Similitud coseno: 1.0281
```

> ⚠️ **Note:** The cosine similarity should be between -1 and 1. If it is greater than 1, there is a rounding error or the vectors are almost identical.

### 6. Interpretation

- Un valor cercano a 1 indica que ambos productos son muy similares en sus Features.
- We use this in ML to measure similarity between text embeddings, Images, etc.

## Complete executable code

```python
import numpy as np

# Definir vectores
producto_a = np.array([4.5, 3.2, 5.0])
producto_b = np.array([4.7, 3.0, 4.8])

# Producto punto
similitud = np.dot(producto_a, producto_b)
print(f"Similitud (producto punto): {similitud:.2f}")

# Normas
norma_a = np.linalg.norm(producto_a)
norma_b = np.linalg.norm(producto_b)
print(f"Norma A: {norma_a:.2f}, Norma B: {norma_b:.2f}")

# Similitud coseno
similitud_coseno = similitud / (norma_a * norma_b)
print(f"Similitud coseno: {similitud_coseno:.4f}")
```

## Errors comunes

- ❌ Forget that vectors must have the same dimension.
- ❌ Confuse dot product with element-by-element multiplication (`*`).
- ❌ Do not normalize before calculating cosine (use norms).

## Exercise propuesto

Create two vectors that represent users based on their movie genre preferences:

```python
usuario_1 = np.array([5, 3, 0, 4])  # [acción, comedia, drama, sci-fi]
usuario_2 = np.array([4, 2, 1, 5])
```

Calculate the cosine similarity between both users to see how similar their tastes are.
