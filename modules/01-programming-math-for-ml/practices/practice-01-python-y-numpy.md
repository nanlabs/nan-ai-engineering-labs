# Práctica 01 — Python y NumPy

## Objective

Familiarizarte con Python básico y operaciones fundamentales con NumPy para manipular arrays y vectores.

## Nivel de dificultad

⭐ Básico (L1)

## Duración estimada

30-40 minutos

______________________________________________________________________

## Parte 1: Resuelta (guía paso a paso)

### Ejercicio 1.1: Crear arrays y realizar operaciones básicas

**Consigna:** Crea dos arrays de NumPy con 5 números cada uno y realiza suma, resta, multiplicación elemento a elemento.

**Solución:**

```python
import numpy as np

# Crear arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Operaciones elemento a elemento
suma = a + b
resta = b - a
multiplicacion = a * b

print("Array a:", a)
print("Array b:", b)
print("Suma:", suma)
print("Resta (b-a):", resta)
print("Multiplicación:", multiplicacion)
```

**Salida esperada:**

```
Array a: [1 2 3 4 5]
Array b: [10 20 30 40 50]
Suma: [11 22 33 44 55]
Resta (b-a): [ 9 18 27 36 45]
Multiplicación: [ 10  40  90 160 250]
```

**Explicación:**

- Las operaciones aritméticas en NumPy se aplican elemento por elemento (broadcasting).
- Esto es mucho más eficiente que usar loops en Python puro.

______________________________________________________________________

### Ejercicio 1.2: Calcular estadísticas básicas

**Consigna:** Dado un array de 10 números, calcula la media, el máximo, el mínimo y la suma total.

**Solución:**

```python
import numpy as np

datos = np.array([23, 45, 12, 67, 34, 89, 23, 56, 78, 90])

media = np.mean(datos)
maximo = np.max(datos)
minimo = np.min(datos)
suma_total = np.sum(datos)

print(f"Datos: {datos}")
print(f"Media: {media:.2f}")
print(f"Máximo: {maximo}")
print(f"Mínimo: {minimo}")
print(f"Suma total: {suma_total}")
```

**Salida esperada:**

```
Datos: [23 45 12 67 34 89 23 56 78 90]
Media: 51.70
Máximo: 90
Mínimo: 12
Suma total: 517
```

**Explicación:**

- `np.mean()`, `np.max()`, `np.min()`, `np.sum()` son funciones vectorizadas muy rápidas.
- Son la base para cálculos estadísticos en datasets grandes.

______________________________________________________________________

## Parte 2: Para resolver (propuesta)

### Ejercicio 2.1: Filtrado condicional

**Consigna:**
Crea un array con 15 números aleatorios entre 1 y 100. Filtra y muestra solo los números mayores a 50.

**Pistas:**

- Usa `np.random.randint(1, 101, size=15)` para generar números aleatorios.
- Usa indexación booleana: `array[array > 50]`.

**Criterio de éxito:**

- El array original debe tener 15 elementos.
- El array filtrado debe contener solo valores > 50.
- Debe ser reproducible (fija una semilla con `np.random.seed(42)`).

______________________________________________________________________

### Ejercicio 2.2: Reshape y transposición

**Consigna:**
Crea un array de 12 números consecutivos (del 1 al 12). Reshape a una matriz de 3x4. Luego transpone la matriz y muestra ambas.

**Pistas:**

- Usa `np.arange(1, 13)` para crear números consecutivos.
- Usa `.reshape(3, 4)` para convertir a matriz.
- Usa `.T` o `np.transpose()` para transponer.

**Criterio de éxito:**

- Matriz original: 3 filas x 4 columnas.
- Matriz transpuesta: 4 filas x 3 columnas.
- Los valores deben coincidir correctamente en posiciones traspuestas.

______________________________________________________________________

### Ejercicio 2.3: Producto punto entre vectores

**Consigna:**
Crea dos vectores de tamaño 5 con valores de tu elección. Calcula:

1. Producto punto.
1. Norma (magnitud) de cada vector.
1. Ángulo entre ambos vectores (usa la fórmula del coseno).

**Pistas:**

- Producto punto: `np.dot(a, b)`.
- Norma: `np.linalg.norm(a)`.
- Coseno del ángulo: `cos(θ) = (a·b) / (||a|| * ||b||)`.
- Ángulo en grados: `np.arccos(coseno) * 180 / np.pi`.

**Criterio de éxito:**

- Producto punto calculado correctamente.
- Normas positivas.
- Ángulo entre 0° y 180°.

______________________________________________________________________

## Entregable

- Un notebook o script `.py` con las soluciones de la Parte 2.
- Comentarios explicando cada paso.
- Salidas visibles (usa `print()` para mostrar resultados).

## Recursos de ayuda

- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Array Creation](https://numpy.org/doc/stable/user/basics.creation.html)
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
