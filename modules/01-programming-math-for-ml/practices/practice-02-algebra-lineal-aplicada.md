# Práctica 02 — Álgebra lineal aplicada

## Objective

Aplicar operaciones de álgebra lineal usando NumPy para resolver problemas típicos de ML: representación de datos, transformaciones y similitud.

## Nivel de dificultad

⭐⭐ Intermedio (L2)

## Duración estimada

40-50 minutos

______________________________________________________________________

## Parte 1: Resuelta (guía paso a paso)

### Ejercicio 1.1: Representar un dataset como matriz

**Consigna:** Representa un dataset de 4 muestras con 3 características cada una como una matriz. Calcula la media de cada característica (columna).

**Contexto ML:** En ML, cada fila es una muestra (ejemplo) y cada columna es una feature (característica).

**Solución:**

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

**Explicación:**

- `axis=0` calcula la media vertical (por columnas).
- Esto es útil para normalizar datasets (restar la media, dividir por desv. estándar).

______________________________________________________________________

### Ejercicio 1.2: Multiplicación matriz-vector (predicción lineal)

**Consigna:** Dado un vector de pesos `w` y un vector de entrada `x`, calcula la predicción usando el producto punto (modelo lineal simple).

**Contexto ML:** `y = w · x + b` es la fórmula de un modelo lineal.

**Solución:**

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

**Explicación:**

- Esto simula cómo una neurona de una red neuronal calcula su salida antes de aplicar la función de activación.
- En ML real, `w` y `b` se aprenden durante el entrenamiento.

______________________________________________________________________

## Parte 2: Para resolver (propuesta)

### Ejercicio 2.1: Normalización de datos

**Consigna:**
Dado el siguiente dataset:

```python
X = np.array([
    [100, 200],
    [150, 250],
    [200, 300],
    [250, 350]
])
```

Normaliza cada columna usando la fórmula:
**X_norm = (X - media) / desviación_estándar**

**Criterio de éxito:**

- Cada columna debe tener media ≈ 0 y desviación estándar ≈ 1.
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
Tienes dos usuarios representados como vectores de preferencias:

```python
usuario_1 = np.array([5, 3, 0, 4, 2])
usuario_2 = np.array([4, 2, 1, 5, 3])
```

Calcula la **similitud coseno** entre ambos.

**Fórmula:**

```
similitud_coseno = (a · b) / (||a|| * ||b||)
```

**Criterio de éxito:**

- Resultado entre -1 y 1.
- Si el resultado es cercano a 1, ambos usuarios tienen preferencias similares.

**Pistas:**

```python
dot_product = np.dot(usuario_1, usuario_2)
norma_1 = np.linalg.norm(usuario_1)
norma_2 = np.linalg.norm(usuario_2)
similitud = dot_product / (norma_1 * norma_2)
```

______________________________________________________________________

### Ejercicio 2.3: Multiplicación de matrices

**Consigna:**
Tienes una matriz de datos `X` (3 muestras, 2 características) y una matriz de pesos `W` (2 características, 4 neuronas).

Calcula `Y = X @ W` (predicciones para las 3 muestras en 4 salidas).

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

**Criterio de éxito:**

- Resultado debe ser una matriz de 3x4.
- Cada fila representa las salidas de una muestra para 4 neuronas.

**Pistas:**

- Usa `@` o `np.matmul()` para multiplicación de matrices.
- Dimensiones: (3, 2) @ (2, 4) = (3, 4).

______________________________________________________________________

## Entregable

- Un notebook o script `.py` con las soluciones de la Parte 2.
- Comentarios explicando cada paso y demostrando comprensión.
- Validación de dimensiones y resultados.

## Recursos de ayuda

- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Broadcasting en NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Matrix Multiplication - Khan Academy](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/v/matrix-multiplication-intro)
