# Example 01 — Cleaning de Dataset Sucio (Ventas de E-commerce)

## Contexto

Trabajas para un e-commerce que tiene un dataset de transacciones con múltiples Problems de calidad. Tu tarea es limpiar los Data antes de entrenar un Model de Prediction de churn.

## Dataset problemático

```python
import pandas as pd
import numpy as np

# Simular dataset con problemas típicos
np.random.seed(42)

data = {
    'customer_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
    'age': [25, 150, 35, np.nan, 42, 28, 'treinta', 55, 40, 32, 29, 38],  # Problema: outlier, NaN, string
    'income': [50000, 75000, 60000, 80000, np.nan, 55000, 70000, 65000, 72000, 58000, 62000, np.nan],  # NaN
    'purchase_date': ['2024-01-15', '2024-01-20', '2024-01-25', '2024-01-30',
                      '2024/02/05', 'invalid', '2024-02-15', '2024-02-20',
                      '2024-02-25', '2024-03-01', '2024-03-05', '2024-03-10'],  # Formato inconsistente
    'category': ['Electronics', 'electronics', 'FASHION', 'Fashion', 'Home', 'home',
                 'Electronics', 'Fashion', 'Home', 'Electronics', 'fashion', 'Home'],  # Inconsistencia de case
    'total_spent': [1200, -500, 800, 1500, 2000, 1100, 900, 1300, 1600, 1000, 850, 1400],  # Negativo
    'email': ['user1@mail.com', 'user2@mail.com', 'user3@mail.com', 'user3@mail.com',  # Duplicado
              'user5@mail.com', 'invalid_email', 'user7@mail.com', 'user8@mail.com',
              'user9@mail.com', 'user10@mail.com', 'user11@mail.com', 'user12@mail.com']
}

df = pd.DataFrame(data)
print("Dataset original:")
print(df)
print(f"\nDimensiones: {df.shape}")
```

## Salida inicial

```
Dataset original:
    customer_id  age   income purchase_date       category  total_spent            email
0           101   25  50000.0    2024-01-15    Electronics         1200   user1@mail.com
1           102  150  75000.0    2024-01-20    electronics         -500   user2@mail.com
2           103   35  60000.0    2024-01-25        FASHION          800   user3@mail.com
3           104  NaN  80000.0    2024-01-30        Fashion         1500   user3@mail.com
4           105   42      NaN    2024/02/05           Home         2000   user5@mail.com
5           106   28  55000.0       invalid           home         1100    invalid_email
6           107  tre  70000.0    2024-02-15    Electronics          900   user7@mail.com
7           108   55  65000.0    2024-02-20        Fashion         1300   user8@mail.com
8           109   40  72000.0    2024-02-25           Home         1600   user9@mail.com
9           110   32  58000.0    2024-03-01    Electronics         1000  user10@mail.com
10          111   29  62000.0    2024-03-05        fashion          850  user11@mail.com
11          112   38      NaN    2024-03-10           Home         1400  user12@mail.com

Dimensiones: (12, 7)
```

______________________________________________________________________

## 🔍 Paso 1: Diagnóstico inicial

```python
# Información general
print("=== INFORMACIÓN GENERAL ===")
print(df.info())
print("\n=== VALORES NULOS ===")
print(df.isnull().sum())
print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(df.describe())
print("\n=== DUPLICADOS ===")
print(f"Filas duplicadas (email): {df.duplicated(subset=['email']).sum()}")
```

**Salida:**

```
=== INFORMACIÓN GENERAL ===
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12 entries, 0 to 11
Data columns (total 7 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   customer_id    12 non-null     int64
 1   age            11 non-null     object  👈 debería ser int
 2   income         10 non-null     float64
 3   purchase_date  12 non-null     object  👈 debería ser datetime
 4   category       12 non-null     object
 5   total_spent    12 non-null     int64
 6   email          12 non-null     object
dtypes: float64(1), int64(2), object(4)

=== VALORES NULOS ===
customer_id      0
age              1  👈 NaN
income           2  👈 NaN
purchase_date    0
category         0
total_spent      0
email            0

=== DUPLICADOS ===
Filas duplicadas (email): 1  👈 user3@mail.com aparece 2 veces
```

### Problems detectados

1. **age:** Type object (debería ser int), tiene NaN y valor "treinta"
1. **income:** 2 valores nulos
1. **purchase_date:** formato inconsistente, valor "invalid"
1. **category:** inconsistencia de mayúsculas/minúsculas
1. **total_spent:** valor negativo (-500)
1. **email:** email duplicado y formato inválido
1. **Posible outlier:** age=150

______________________________________________________________________

## 🛠️ Paso 2: Cleaning sistemática

### 2.1 Remover duplicados

```python
# Identificar duplicados
print("Filas duplicadas:")
print(df[df.duplicated(subset=['email'], keep=False)])

# Opción 1: Mantener primera ocurrencia
df_clean = df.drop_duplicates(subset=['email'], keep='first')
print(f"\nFilas eliminadas: {len(df) - len(df_clean)}")
print(f"Nuevas dimensiones: {df_clean.shape}")
```

**Decisión:** Mantener primera ocurrencia de email duplicado (<user3@mail.com>).

### 2.2 Corregir columna `age`

```python
# Convertir a numérico (coerce convierte inválidos a NaN)
df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
print(f"Valores NaN en age: {df_clean['age'].isnull().sum()}")

# Detectar outliers con IQR
Q1 = df_clean['age'].quantile(0.25)
Q3 = df_clean['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['age'] < lower_bound) | (df_clean['age'] > upper_bound)]
print(f"\nOutliers detectados: {len(outliers)}")
print(outliers[['customer_id', 'age']])

# Opción conservadora: reemplazar outliers con mediana
df_clean['age'] = df_clean['age'].apply(
    lambda x: df_clean['age'].median() if (x < lower_bound or x > upper_bound) else x
)

# Imputar NaN con mediana
df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
print(f"\nNaN restantes en age: {df_clean['age'].isnull().sum()}")
```

**Decisión:**

- outlier (150) reemplazado con mediana
- NaN imputados con mediana

### 2.3 Corregir columna `income`

```python
# Estrategia: imputar con mediana del grupo (por category)
df_clean['income'] = df_clean.groupby('category')['income'].transform(
    lambda x: x.fillna(x.median())
)

# Si aún quedan NaN (grupo sin valores), usar mediana global
df_clean['income'].fillna(df_clean['income'].median(), inplace=True)
print(f"NaN restantes en income: {df_clean['income'].isnull().sum()}")
```

**Decisión:** Imputación por grupo (category) para mayor Precision.

### 2.4 Corregir columna `purchase_date`

```python
# Función para normalizar formato
def parse_date(date_str):
    try:
        # Intentar formato ISO
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except:
        try:
            # Intentar formato alternativo
            return pd.to_datetime(date_str, format='%Y/%m/%d')
        except:
            # Inválido: retornar NaT
            return pd.NaT

df_clean['purchase_date'] = df_clean['purchase_date'].apply(parse_date)
print(f"NaT (fechas inválidas): {df_clean['purchase_date'].isnull().sum()}")

# Eliminar filas con fechas inválidas (decisión de negocio)
df_clean = df_clean.dropna(subset=['purchase_date'])
print(f"Dimensiones tras eliminar fechas inválidas: {df_clean.shape}")
```

**Decisión:** Eliminar registros con fecha inválida (opción: imputar con fecha modal o eliminar).

### 2.5 Normalizar columna `category`

```python
# Convertir a lowercase y title case
df_clean['category'] = df_clean['category'].str.lower().str.title()
print("Categorías únicas:")
print(df_clean['category'].value_counts())
```

**Salida:**

```
Categorías únicas:
Electronics    3
Fashion        3
Home           3
```

### 2.6 Corregir Anomalies en `total_spent`

```python
# Detectar valores negativos
negative_mask = df_clean['total_spent'] < 0
print(f"Valores negativos: {negative_mask.sum()}")

# Opción 1: Convertir a valor absoluto (asumir error de signo)
# df_clean['total_spent'] = df_clean['total_spent'].abs()

# Opción 2: Eliminar transacciones negativas (refunds?)
df_clean = df_clean[df_clean['total_spent'] >= 0]
print(f"Dimensiones tras eliminar negativos: {df_clean.shape}")
```

**Decisión:** Eliminar transacciones negativas (interpretadas como refunds, fuera de alcance del Analysis).

### 2.7 Validar emails

```python
import re

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

df_clean['email_valid'] = df_clean['email'].apply(is_valid_email)
invalid_emails = df_clean[~df_clean['email_valid']]
print(f"Emails inválidos: {len(invalid_emails)}")

# Eliminar registros con emails inválidos
df_clean = df_clean[df_clean['email_valid']].drop('email_valid', axis=1)
print(f"Dimensiones finales: {df_clean.shape}")
```

______________________________________________________________________

## ✅ Paso 3: Dataset limpio final

```python
print("=== DATASET LIMPIO ===")
print(df_clean)
print("\n=== INFO ===")
print(df_clean.info())
print("\n=== NULOS ===")
print(df_clean.isnull().sum())
print("\n=== DUPLICADOS ===")
print(f"Duplicados: {df_clean.duplicated().sum()}")
```

**Salida limpia:**

```
=== DATASET LIMPIO ===
    customer_id   age   income purchase_date     category  total_spent            email
0           101  25.0  50000.0    2024-01-15  Electronics         1200   user1@mail.com
2           103  35.0  60000.0    2024-01-25      Fashion          800   user3@mail.com
4           105  42.0  67500.0    2024-02-05         Home         2000   user5@mail.com
6           107  35.0  70000.0    2024-02-15  Electronics          900   user7@mail.com
7           108  55.0  65000.0    2024-02-20      Fashion         1300   user8@mail.com
8           109  40.0  72000.0    2024-02-25         Home         1600   user9@mail.com
9           110  32.0  58000.0    2024-03-01  Electronics         1000  user10@mail.com
10          111  29.0  62000.0    2024-03-05      Fashion          850  user11@mail.com
11          112  38.0  67500.0    2024-03-10         Home         1400  user12@mail.com

Dimensiones finales: (9, 7)
```

**Resumen de transformaciones:**

- Filas originales: 12 → Filas finales: 9 (3 eliminadas)
- Nulos imputados: age (2), income (2)
- outliers corregidos: 1 (age=150)
- Duplicados eliminados: 1
- Fechas inválidas eliminadas: 1
- Emails inválidos eliminados: 1
- Valores negativos eliminados: 1
- Categorías normalizadas: 3 diferentes

______________________________________________________________________

## 📊 Paso 4: Visualization de calidad

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Distribución de age
axes[0, 0].hist(df_clean['age'], bins=10, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Distribución de Edad')
axes[0, 0].set_xlabel('Edad')
axes[0, 0].set_ylabel('Frecuencia')

# 2. Distribución de income
axes[0, 1].hist(df_clean['income'], bins=10, edgecolor='black', color='lightgreen')
axes[0, 1].set_title('Distribución de Ingresos')
axes[0, 1].set_xlabel('Ingresos')
axes[0, 1].set_ylabel('Frecuencia')

# 3. Transacciones por categoría
df_clean['category'].value_counts().plot(kind='bar', ax=axes[1, 0], color='coral')
axes[1, 0].set_title('Transacciones por Categoría')
axes[1, 0].set_xlabel('Categoría')
axes[1, 0].set_ylabel('Cantidad')
axes[1, 0].tick_params(axis='x', rotation=0)

# 4. Total spent vs age
axes[1, 1].scatter(df_clean['age'], df_clean['total_spent'], alpha=0.6, color='purple')
axes[1, 1].set_title('Gasto vs Edad')
axes[1, 1].set_xlabel('Edad')
axes[1, 1].set_ylabel('Total Gastado')

plt.tight_layout()
plt.show()
```

______________________________________________________________________

## 🎓 Lessons aprendidas

### ✅ Buenas Practices aplicadas

1. **Diagnóstico antes de actuar:** `info()`, `describe()`, `isnull()`
1. **Documentar decisiones:** ¿Por qué eliminar vs imputar?
1. **Validation contextual:** IQR para outliers, regex para emails
1. **Imputación inteligente:** Por grupo (category) en lugar de global
1. **Preservar trazabilidad:** Dataset original intacto, transformaciones en copia

### ⚠️ Consideraciones de negocio

- **Eliminar vs corregir:** Depende del % de Data afectados y disponibilidad de información adicional
- **Imputación:** Puede introducir sesgo. Documentar método usado
- **outliers:** No siempre son Errors. Validar con experto de dominio

### 🔧 Herramientas útiles

- `pd.to_numeric(errors='coerce')`: Conversión robusta
- `groupby().transform()`: Imputación por grupo
- `dropna()` vs `fillna()`: Trade-off entre pérdida de Data y calidad
- Regex para Validation de formatos
