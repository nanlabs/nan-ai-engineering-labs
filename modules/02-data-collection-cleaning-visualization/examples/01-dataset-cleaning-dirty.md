# Example 01 — Dirty Dataset Cleaning (E-commerce Sales)

## Context

You work for an e-commerce that has a transaction dataset with multiple quality problems. Your task is to clean the Data before training a Churn Prediction Model.

## Problematic dataset

```python
import pandas as pd
import numpy as np

# Similar dataset con problems typical
np.random.seed(42)

data = {
    'customer_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
    'age': [25, 150, 35, np.nan, 42, 28, 'treinta', 55, 40, 32, 29, 38],  # Problem: outlier, NaN, string
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

## Initial output

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

Dimensions: (12, 7)
```

______________________________________________________________________

## 🔍 Step 1: Initial diagnosis

```python
# Information general
print("=== INFORMATION GENERAL ===")
print(df.info())
print("\n=== VALORES NULOS ===")
print(df.isnull().sum())
print("\n=== STATISTICS DESCRIPTIVAS ===")
print(df.describe())
print("\n=== DUPLICADOS ===")
print(f"Filas duplicadas (email): {df.duplicated(subset=['email']).sum()}")
```

**Output:**

```
=== INFORMATION GENERAL ===
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12 entries, 0 to 11
Data columns (total 7 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   customer_id    12 non-null     int64
 1   age            11 non-null     object  👈 ought ser int
 2   income         10 non-null     float64
 3   purchase_date  12 non-null     object  👈 ought ser datetime
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

### Problems detected

1. **age:** Type object (should be int), has NaN and value "thirty"
1. **income:** 2 values nulls
1. **purchase_date:** inconsistent format, value "invalid"
1. **category:** case inconsistency
1. **total_spent:** negative value (-500)
1. **email:** duplicate email and invalid format
1. **Possible outlier:** age=150

______________________________________________________________________

## 🛠️ Step 2: Systematic Cleaning

### 2.1 Remove duplicates

```python
# Identify duplicates
print("Filas duplicadas:")
print(df[df.duplicated(subset=['email'], keep=False)])

# Option 1: Mantener primera ocurrencia
df_clean = df.drop_duplicates(subset=['email'], keep='first')
print(f"\nFilas eliminadas: {len(df) - len(df_clean)}")
print(f"Nuevas dimensions: {df_clean.shape}")
```

**Decision:** Keep first occurrence of duplicate email (<user3@mail.com>).

### 2.2 Correct `age` column

```python
# Convert a numeric (coerce convierte invalids a NaN)
df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
print(f"Valores NaN en age: {df_clean['age'].isnull().sum()}")

# Detect outliers con IQR
Q1 = df_clean['age'].quantile(0.25)
Q3 = df_clean['age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['age'] < lower_bound) | (df_clean['age'] > upper_bound)]
print(f"\nOutliers detected: {len(outliers)}")
print(outliers[['customer_id', 'age']])

# Option conservadora: reemplazar outliers con median
df_clean['age'] = df_clean['age'].apply(
    lambda x: df_clean['age'].median() if (x < lower_bound or x > upper_bound) else x
)

# Imputar NaN con median
df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
print(f"\nNaN restantes en age: {df_clean['age'].isnull().sum()}")
```

**Decision:**

- outlier (150) replaced with median
- NaN imputed with median

### 2.3 Correct `income` column

```python
# Estrategia: imputar con median del group (por category)
df_clean['income'] = df_clean.groupby('category')['income'].transform(
    lambda x: x.fillna(x.median())
)

# Si aún quedan NaN (group sin values), use median global
df_clean['income'].fillna(df_clean['income'].median(), inplace=True)
print(f"NaN restantes en income: {df_clean['income'].isnull().sum()}")
```

**Decision:** Imputation by group (category) for greater Precision.

### 2.4 Correct `purchase_date` column

```python
# Function para normalizar format
def parse_date(date_str):
    try:
        # Intentar format ISO
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except:
        try:
            # Intentar format alternativo
            return pd.to_datetime(date_str, format='%Y/%m/%d')
        except:
            # Invalid: retornar NaT
            return pd.NaT

df_clean['purchase_date'] = df_clean['purchase_date'].apply(parse_date)
print(f"NaT (fechas invalid): {df_clean['purchase_date'].isnull().sum()}")

# Eliminar rows con fechas invalid (decision de business)
df_clean = df_clean.dropna(subset=['purchase_date'])
print(f"Dimensions tras eliminar fechas invalid: {df_clean.shape}")
```

**Decision:** Delete records with invalid date (option: impute with modal date or delete).

### 2.5 Normalize `category` column

```python
# Convert a lowercase y title case
df_clean['category'] = df_clean['category'].str.lower().str.title()
print("Categories unique:")
print(df_clean['category'].value_counts())
```

**Output:**

```
Categories unique:
Electronics    3
Fashion        3
Home           3
```

### 2.6 Correct Anomalies in `total_spent`

```python
# Detect values negativos
negative_mask = df_clean['total_spent'] < 0
print(f"Valores negativos: {negative_mask.sum()}")

# Option 1: Convert a valor absoluto (asumir error de signo)
# df_clean['total_spent'] = df_clean['total_spent'].abs()

# Option 2: Eliminar transacciones negativas (refunds?)
df_clean = df_clean[df_clean['total_spent'] >= 0]
print(f"Dimensions tras eliminar negativos: {df_clean.shape}")
```

**Decision:** Eliminate negative transactions (interpreted as refunds, outside the scope of the Analysis).

### 2.7 Validate emails

```python
import re

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

df_clean['email_valid'] = df_clean['email'].apply(is_valid_email)
invalid_emails = df_clean[~df_clean['email_valid']]
print(f"Emails invalids: {len(invalid_emails)}")

# Eliminar registros con emails invalids
df_clean = df_clean[df_clean['email_valid']].drop('email_valid', axis=1)
print(f"Dimensions finales: {df_clean.shape}")
```

______________________________________________________________________

## ✅ Step 3: Final clean dataset

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

**Clean output:**

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

Dimensions finales: (9, 7)
```

**Summary of transformations:**

- Original rows: 12 → Final rows: 9 (3 removed)
- Nulls imputed: age (2), income (2)
- corrected outliers: 1 (age=150)
- Duplicates removed: 1
- Invalid dates removed: 1
- Invalid emails deleted: 1
- Negative values ​​removed: 1
- Normalized categories: 3 different

______________________________________________________________________

## 📊 Step 4: Quality visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Distribution de age
axes[0, 0].hist(df_clean['age'], bins=10, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Distribution de Edad')
axes[0, 0].set_xlabel('Edad')
axes[0, 0].set_ylabel('Frecuencia')

# 2. Distribution de income
axes[0, 1].hist(df_clean['income'], bins=10, edgecolor='black', color='lightgreen')
axes[0, 1].set_title('Distribution de Ingresos')
axes[0, 1].set_xlabel('Ingresos')
axes[0, 1].set_ylabel('Frecuencia')

# 3. Transacciones por category
df_clean['category'].value_counts().plot(kind='bar', ax=axes[1, 0], color='coral')
axes[1, 0].set_title('Transacciones por Category')
axes[1, 0].set_xlabel('Category')
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

## 🎓 Lessons learned

### ✅ Good Practices applied

1. **Diagnosis before acting:** `info()`, `describe()`, `isnull()`
1. **Document decisions:** Why delete vs impute?
1. **Contextual validation:** IQR for outliers, regex for emails
1. **Smart Imputation:** By group (category) instead of global
1. **Preserve traceability:** Original dataset intact, transformations in copy

### ⚠️ Business considerations

- **Delete vs correct:** Depends on the % of data affected and availability of additional information
- **Imputation:** May introduce bias. Document method used
- **outliers:** They are not always Errors. Validate with domain expert

### 🔧 Useful tools

- `pd.to_numeric(errors='coerce')`: Robust conversion
- `groupby().transform()`: Imputation by group
- `dropna()` vs `fillna()`: Trade-off between data loss and quality
- Regex for Validation of formats
