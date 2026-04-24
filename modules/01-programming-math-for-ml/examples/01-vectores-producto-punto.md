# Ejemplo 1 — Vectores y producto punto

## Objective

Representar vectores usando NumPy y calcular similitud básica entre ellos mediante el producto punto (dot product).

## Conceptos previos

- Un **vector** es una lista ordenada de números.
- El **producto punto** de dos vectores mide su similitud direccional.
- Fórmula: `a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ`
- Si el producto punto es alto, los vectores apuntan en direcciones similares.

## Implementación paso a paso

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

### 3. Calcular el producto punto

```python
similitud = np.dot(producto_a, producto_b)
print(f"Similitud entre productos: {similitud:.2f}")
```

**Salida esperada:**

```
Similitud entre productos: 54.71
```

### 4. Calcular la norma (longitud) de cada vector

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

> ⚠️ **Nota:** La similitud coseno debería estar entre -1 y 1. Si sale mayor a 1, hay un error de redondeo o los vectores son casi idénticos.

### 6. Interpretación

- Un valor cercano a 1 indica que ambos productos son muy similares en sus características.
- Esto lo usamos en ML para medir similitud entre embeddings de textos, imágenes, etc.

## Código completo ejecutable

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

## Errores comunes

- ❌ Olvidar que los vectores deben tener la misma dimensión.
- ❌ Confundir producto punto con multiplicación elemento a elemento (`*`).
- ❌ No normalizar antes de calcular coseno (usar normas).

## Ejercicio propuesto

Crea dos vectores que representen usuarios según sus preferencias de géneros de películas:

```python
usuario_1 = np.array([5, 3, 0, 4])  # [acción, comedia, drama, sci-fi]
usuario_2 = np.array([4, 2, 1, 5])
```

Calcula la similitud coseno entre ambos usuarios para ver qué tan parecidos son sus gustos.
