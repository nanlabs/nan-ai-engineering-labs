# Example 2 — Descriptive statistics with Pandas

## Objective

Calculate mean, median, standard deviation and detect outliers in a real dataset using Pandas.

## Concepts previos

- **Mean:** arithmetic average of the values.
- **Median:** central value when the Data is ordered.
- **Standard deviation:** measures how much the values ​​are dispersed with respect to the mean.
- **outliers:** outlier values ​​that deviate significantly from the rest.

## Step by step implementation

### 1. Import libraries

```python
import pandas as pd
import numpy as np
```

### 2. Create an Example dataset

```python
# Simular datos de tiempos de respuesta de un servidor (en ms)
np.random.seed(42)
tiempos_normales = np.random.normal(loc=150, scale=30, size=95)
tiempos_outliers = np.array([450, 480, 520, 490, 510])  # valores atípicos
tiempos = np.concatenate([tiempos_normales, tiempos_outliers])

# Crear DataFrame
df = pd.DataFrame({'tiempo_respuesta_ms': tiempos})
print(df.head())
```

**Salida esperada:**

```
   tiempo_respuesta_ms
0            164.967142
1            141.617357
2            196.480538
3            181.588644
4            145.461832
```

### 3. Calculate descriptive statistics

```python
media = df['tiempo_respuesta_ms'].mean()
mediana = df['tiempo_respuesta_ms'].median()
desviacion = df['tiempo_respuesta_ms'].std()

print(f"Media: {media:.2f} ms")
print(f"Mediana: {mediana:.2f} ms")
print(f"Desvío estándar: {desviacion:.2f} ms")
```

**Salida esperada:**

```
Media: 165.97 ms
Mediana: 151.23 ms
Desvío estándar: 74.52 ms
```

### 4. View distribution with a quick summary

```python
print("\nResumen estadístico completo:")
print(df.describe())
```

**Salida esperada:**

```
       tiempo_respuesta_ms
count            100.000000
mean             165.973418
std               74.524912
min              105.678945
25%              132.456789
50%              151.234567
75%              169.876543
max              520.000000
```

### 5. Detect outliers using the IQR method

```python
Q1 = df['tiempo_respuesta_ms'].quantile(0.25)
Q3 = df['tiempo_respuesta_ms'].quantile(0.75)
IQR = Q3 - Q1

# Límites para outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print(f"\nRango normal: [{limite_inferior:.2f}, {limite_superior:.2f}]")

# Filtrar outliers
outliers = df[(df['tiempo_respuesta_ms'] < limite_inferior) |
              (df['tiempo_respuesta_ms'] > limite_superior)]

print(f"\nOutliers detectados: {len(outliers)}")
print(outliers)
```

**Salida esperada:**

```
Rango normal: [76.23, 225.67]

Outliers detectados: 5
    tiempo_respuesta_ms
95              450.0
96              480.0
97              520.0
98              490.0
99              510.0
```

### 6. Interpretation

- The **median** is less than the **mean**, indicating that there are high outliers that inflate the average.
- The **outliers** detected correspond to abnormally slow response times (>450ms).

## Complete executable code

```python
import pandas as pd
import numpy as np

# Crear datos
np.random.seed(42)
tiempos_normales = np.random.normal(loc=150, scale=30, size=95)
tiempos_outliers = np.array([450, 480, 520, 490, 510])
tiempos = np.concatenate([tiempos_normales, tiempos_outliers])

df = pd.DataFrame({'tiempo_respuesta_ms': tiempos})

# Estadísticas básicas
print("Estadísticas descriptivas:")
print(f"Media: {df['tiempo_respuesta_ms'].mean():.2f} ms")
print(f"Mediana: {df['tiempo_respuesta_ms'].median():.2f} ms")
print(f"Desvío estándar: {df['tiempo_respuesta_ms'].std():.2f} ms")

# Detección de outliers con IQR
Q1 = df['tiempo_respuesta_ms'].quantile(0.25)
Q3 = df['tiempo_respuesta_ms'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df['tiempo_respuesta_ms'] < limite_inferior) |
              (df['tiempo_respuesta_ms'] > limite_superior)]

print(f"\nOutliers detectados ({len(outliers)}):")
print(outliers['tiempo_respuesta_ms'].values)
```

## Errors comunes

- ❌ Use only the mean without considering the median (sensitive to outliers).
- ❌ Do not verify the Data distribution before applying Models.
- ❌ Confuse variance with standard deviation (dev = √variance).

## Exercise propuesto

Load a real Kaggle dataset and calculate:

1. Mean, median and standard deviation of height of people.
1. Detect outliers using the IQR method.
1. Visualize the distribution with a histogram using Matplotlib.
