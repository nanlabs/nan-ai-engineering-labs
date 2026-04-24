# Practice 02 — Applied linear algebra

## Objective

Apply linear algebra operations using NumPy to solve typical ML problems: data representation, transformations and similarity.

## Difficulty level

⭐⭐ Intermedio (L2)

## Estimated duration

40-50 minutos

______________________________________________________________________

## Part 1: Solved (step by step guide)

### Ejercicio 1.1: Representar un dataset como matriz

**Setpoint:** Represents a dataset of 4 samples with 3 features each as a matrix. Calculate the mean of each characteristic (column).

**ML Context:** In ML, each row is a sample and each column is a feature.

**Solution:**

```python
import numpy as np

# Dataset: 4 muestras, 3 características [altura_cm, peso_kg, edad]
X = np.array([
    [170, 65, 25],
    [160, 55, 30],
    [180, 75, 28],
    [175, 70, 35]
])

print("Dataset (matriz X):")
print(X)
print(f"\nDimensiones: {X.shape}")  # (4, 3) = 4 muestras, 3 features

# Media por columna (característica)
media_columnas = np.mean(X, axis=0)
print(f"\nMedia por característica: {media_columnas}")
```

**Salida esperada:**

```
Dataset (matriz X):
[[170  65  25]
 [160  55  30]
 [180  75  28]
 [175  70  35]]

Dimensiones: (4, 3)

Media por característica: [171.25  66.25  29.5 ]
```

**Explanation:**

- `axis=0` calculates the vertical mean (by columns).
- This is useful for normalizing datasets (subtract mean, divide by standard deviation).

______________________________________________________________________

### Exercise 1.2: Matrix-vector multiplication (linear prediction)

**Setpoint:** Given a weight vector `w` and an input vector `x`, calculate the prediction using the dot product (simple linear model).

**ML Context:** `y = w · x + b` is the formula for a linear model.

**Solution:**

```python
import numpy as np

# Vector de pesos (parámetros del modelo)
w = np.array([0.5, 1.2, -0.3])

# Vector de entrada (una muestra)
x = np.array([170, 65, 25])

# Bias (término independiente)
b = 10

# Predicción
y_pred = np.dot(w, x) + b

print(f"Pesos: {w}")
print(f"Entrada: {x}")
print(f"Predicción: {y_pred:.2f}")
```

**Salida esperada:**

```
Pesos: [ 0.5  1.2 -0.3]
Entrada: [170  65  25]
Predicción: 170.50
```

**Explanation:**

- This simulates how a neuron in a neural network calculates its output before applying the activation function.
- In real ML, `w` and `b` are learned during training.

______________________________________________________________________

## Part 2: To solve (proposal)

### Exercise 2.1: Data Normalization

**Consigna:**
Given the following dataset:

```python
X = np.array([
    [100, 200],
    [150, 250],
    [200, 300],
    [250, 350]
])
```

Normalize each column using the formula:
**X_norm = (X - mean) / standard_deviation**

**Success Criteria:**

- Each column must have mean ≈ 0 and standard deviation ≈ 1.
- Usa `np.mean()` y `np.std()` con `axis=0`.

**Pistas:**

```python
media = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - media) / std
```

______________________________________________________________________

### Ejercicio 2.2: Similitud coseno entre muestras

**Consigna:**
You have two users represented as preference vectors:

```python
usuario_1 = np.array([5, 3, 0, 4, 2])
usuario_2 = np.array([4, 2, 1, 5, 3])
```

Calculate the **cosine similarity** between the two.

**Formula:**

```
similitud_coseno = (a · b) / (||a|| * ||b||)
```

**Success Criteria:**

- Resultado entre -1 y 1.
- If the result is close to 1, both users have similar preferences.

**Pistas:**

```python
dot_product = np.dot(usuario_1, usuario_2)
norma_1 = np.linalg.norm(usuario_1)
norma_2 = np.linalg.norm(usuario_2)
similitud = dot_product / (norma_1 * norma_2)
```

______________________________________________________________________

### Exercise 2.3: Matrix multiplication

**Consigna:**
You have a data matrix `X` (3 samples, 2 features) and a weight matrix `W` (2 features, 4 neurons).

Calculate `Y = X @ W` (predictions for the 3 samples in 4 outputs).

```python
X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0]
])

W = np.array([
    [0.5, 1.0, -0.5, 0.2],
    [1.5, 0.5,  0.3, -0.1]
])
```

**Success Criteria:**

- Result must be a 3x4 matrix.
- Each row represents the outputs of a sample for 4 neurons.

**Pistas:**

- Use `@` or `np.matmul()` for matrix multiplication.
- Dimensiones: (3, 2) @ (2, 4) = (3, 4).

______________________________________________________________________

## Entregable

- A notebook or script `.py` with the solutions from Part 2.
- Comments explaining each step and demonstrating understanding.
- Validation of dimensions and results.

## Help resources

- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Broadcasting en NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Matrix Multiplication - Khan Academy](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/v/matrix-multiplication-intro)
