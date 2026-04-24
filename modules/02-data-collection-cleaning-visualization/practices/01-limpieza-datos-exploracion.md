# Práctica 01 — Limpieza de Datos y Exploración Inicial

## 🎯 Objetivos

- Detectar y manejar valores faltantes
- Identificar y tratar outliers
- Realizar transformaciones de datos
- Crear visualizaciones exploratorias

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Cargar y explorar dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset: ventas de tienda online
np.random.seed(42)

n_samples = 500

data = {
    'order_id': range(1, n_samples + 1),
    'customer_age': np.random.randint(18, 75, n_samples),
    'product_price': np.random.uniform(10, 500, n_samples),
    'quantity': np.random.randint(1, 10, n_samples),
    'shipping_cost': np.random.uniform(0, 50, n_samples),
    'discount_pct': np.random.uniform(0, 0.3, n_samples),
    'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer'], n_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
}

df = pd.DataFrame(data)

# Inyectar problemas de calidad
# 1. Valores faltantes
missing_indices = np.random.choice(df.index, size=30, replace=False)
df.loc[missing_indices[:10], 'customer_age'] = np.nan
df.loc[missing_indices[10:20], 'shipping_cost'] = np.nan
df.loc[missing_indices[20:30], 'payment_method'] = np.nan

# 2. Outliers
outlier_indices = np.random.choice(df.index, size=10, replace=False)
df.loc[outlier_indices, 'product_price'] = np.random.uniform(1000, 5000, 10)

# 3. Datos inconsistentes
df.loc[np.random.choice(df.index, 5), 'quantity'] = -1  # Cantidad negativa (error)

print("=== Información del Dataset ===")
print(df.info())
print("\n=== Primeras filas ===")
print(df.head())
```

**✅ Solución - Análisis inicial:**

```python
# Análisis de valores faltantes
print("\n=== Valores Faltantes ===")
missing_summary = pd.DataFrame({
    'columna': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_summary[missing_summary['missing_count'] > 0])

# Análisis de duplicados
print(f"\nDuplicados: {df.duplicated().sum()}")

# Estadísticas descriptivas
print("\n=== Estadísticas Descriptivas ===")
print(df.describe())
```

### Ejercicio 1.2: Manejar valores faltantes

```python
# Estrategias de imputación

# 1. Imputación por media (variables numéricas continuas)
df['customer_age_imputed'] = df['customer_age'].fillna(df['customer_age'].median())

# 2. Imputación por mediana (variables con outliers)
df['shipping_cost_imputed'] = df['shipping_cost'].fillna(df['shipping_cost'].median())

# 3. Imputación por moda (variables categóricas)
df['payment_method_imputed'] = df['payment_method'].fillna(df['payment_method'].mode()[0])

# Comparar antes y después
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Customer age
axes[0].hist(df['customer_age'].dropna(), bins=20, alpha=0.5, label='Original', color='blue')
axes[0].hist(df['customer_age_imputed'], bins=20, alpha=0.5, label='Imputado', color='red')
axes[0].set_xlabel('Customer Age')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de Edad (antes vs después)')
axes[0].legend()

# Shipping cost
axes[1].hist(df['shipping_cost'].dropna(), bins=20, alpha=0.5, label='Original', color='blue')
axes[1].hist(df['shipping_cost_imputed'], bins=20, alpha=0.5, label='Imputado', color='red')
axes[1].set_xlabel('Shipping Cost')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribución de Costo de Envío')
axes[1].legend()

plt.tight_layout()
plt.savefig('missing_values_imputation.png', dpi=150)
plt.show()

print("✅ Valores faltantes imputados")
```

### Ejercicio 1.3: Detectar y manejar outliers

```python
def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    Detecta outliers usando método IQR
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    return outliers, lower_bound, upper_bound

# Detectar outliers en product_price
outliers, lower, upper = detect_outliers_iqr(df, 'product_price')

print(f"\n=== Outliers en Product Price ===")
print(f"Límite inferior: ${lower:.2f}")
print(f"Límite superior: ${upper:.2f}")
print(f"Outliers detectados: {len(outliers)}")
print(f"\nEjemplos de outliers:")
print(outliers[['order_id', 'product_price']].head())

# Visualización con boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot antes
axes[0].boxplot(df['product_price'].dropna())
axes[0].set_ylabel('Product Price')
axes[0].set_title('Antes de remover outliers')
axes[0].grid(True, alpha=0.3)

# Tratamiento: Winsorización (cap en upper bound)
df['product_price_clean'] = df['product_price'].clip(upper=upper)

# Boxplot después
axes[1].boxplot(df['product_price_clean'])
axes[1].set_ylabel('Product Price')
axes[1].set_title('Después de winsorización')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outliers_treatment.png', dpi=150)
plt.show()

print("✅ Outliers tratados con winsorización")
```

### Ejercicio 1.4: Validar y corregir inconsistencias

```python
# Detectar cantidades negativas
invalid_qty = df[df['quantity'] < 0]

print(f"\n=== Valores Inválidos en Quantity ===")
print(f"Registros con cantidad negativa: {len(invalid_qty)}")

# Corrección: remover registros inválidos
df_clean = df[df['quantity'] > 0].copy()

print(f"Registros removidos: {len(df) - len(df_clean)}")
print(f"Dataset limpio: {len(df_clean)} registros")

# Calcular campo derivado
df_clean['total_price'] = (
    df_clean['product_price_clean'] * df_clean['quantity'] *
    (1 - df_clean['discount_pct']) + df_clean['shipping_cost_imputed']
)

print(f"\n✅ Dataset limpio:")
print(df_clean.head())
print(f"\nTotal price range: ${df_clean['total_price'].min():.2f} - ${df_clean['total_price'].max():.2f}")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Pipeline de Limpieza Completo

**Enunciado:**
Crea una función `clean_sales_data(df)` que:

1. Detecte e impute valores faltantes (estrategia apropiada por tipo de dato)
1. Identifique outliers en todas las columnas numéricas
1. Valide que no haya valores negativos en price/quantity
1. Retorne un DataFrame limpio y un reporte de calidad

**Validación esperada:**

```python
def clean_sales_data(df):
    # Tu código aquí
    pass

# Test
df_cleaned, quality_report = clean_sales_data(df)

assert df_cleaned.isnull().sum().sum() == 0, "Hay valores faltantes sin tratar"
assert (df_cleaned['quantity'] < 0).sum() == 0, "Hay cantidades negativas"
assert quality_report['rows_removed'] >= 0, "Reporte inválido"

print("✅ Tests pasados")
```

### Ejercicio 2.2: Análisis Exploratorio Multi-variable

**Enunciado:**
Genera un reporte exploratorio que incluya:

1. Distribución de ventas por región
1. Correlación entre precio, cantidad y descuento
1. Total de ventas por método de pago
1. Análisis de tendencia de edad vs precio promedio

**Validación esperada:**

- 4 visualizaciones (histogramas, scatter plots, barplots)
- Reporte con insights principales
- Correlaciones calculadas correctamente

### Ejercicio 2.3: Feature Engineering

**Enunciado:**
Crea nuevas features útiles para análisis:

1. `price_per_unit` = product_price / quantity
1. `is_high_value` = total_price > 75th percentile
1. `age_group` = bins de edad (18-30, 31-45, 46-60, 61+)
1. `discount_category` = sin descuento / bajo / medio / alto

**Validación esperada:**

```python
assert 'price_per_unit' in df_engineered.columns
assert 'is_high_value' in df_engineered.columns
assert df_engineered['is_high_value'].dtype == bool
assert df_engineered['age_group'].nunique() == 4
```

### Ejercicio 2.4: Detección de Anomalías

**Enunciado:**
Implementa un detector de pedidos anómalos usando:

1. Z-score para detectar precios extremos
1. Reglas de negocio (ej: descuento > 50% con precio > $300)
1. Valores imposibles (shipping_cost > product_price)

Marca registros con `is_anomalous` flag.

### Ejercicio 2.5: Visualización de Calidad de Datos

**Enunciado:**
Crea un dashboard de calidad con:

1. Heatmap de valores faltantes por columna
1. Distribución de outliers detectados
1. Matriz de correlación
1. Distribución de variables categóricas

Guarda como `data_quality_dashboard.png`.

______________________________________________________________________

## ✅ Checklist de Competencias

Después de completar esta práctica, deberías poder:

- [ ] Detectar valores faltantes y aplicar estrategias de imputación
- [ ] Identificar outliers con métodos estadísticos (IQR, Z-score)
- [ ] Aplicar transformaciones de datos (winsorización, clipping)
- [ ] Validar integridad de datos (rangos válidos, consistencia)
- [ ] Crear visualizaciones exploratorias efectivas
- [ ] Implementar pipelines de limpieza reproducibles
- [ ] Generar reportes de calidad de datos
- [ ] Calcular features derivadas

______________________________________________________________________

## 📚 Recursos Adicionales

**Librerías útiles:**

- `pandas-profiling`: EDA automático
- `missingno`: Visualización de missing data
- `PyOD`: Detección de outliers
- `Great Expectations`: Validación de datos

**Lecturas:**

- [Tidy Data por Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Data Cleaning Tutorial - Kaggle](https://www.kaggle.com/learn/data-cleaning)
