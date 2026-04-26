# Practice 01 — Data Cleaning and Initial Exploration

## 🎯 Objectives

- Detect and handle missing values
- Identify and treat outliers
- Perform data transformations
- Create exploratory visualizations

______________________________________________________________________

## 📚 Part 1: Guided Exercises

### Exercise 1.1: Load and explore dataset

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

# Inyectar problems de quality
# 1. Valores faltantes
missing_indices = np.random.choice(df.index, size=30, replace=False)
df.loc[missing_indices[:10], 'customer_age'] = np.nan
df.loc[missing_indices[10:20], 'shipping_cost'] = np.nan
df.loc[missing_indices[20:30], 'payment_method'] = np.nan

# 2. Outliers
outlier_indices = np.random.choice(df.index, size=10, replace=False)
df.loc[outlier_indices, 'product_price'] = np.random.uniform(1000, 5000, 10)

# 3. Data inconsistentes
df.loc[np.random.choice(df.index, 5), 'quantity'] = -1  # Cantidad negativa (error)

print("=== Information del Dataset ===")
print(df.info())
print("\n=== Primeras rows ===")
print(df.head())
```

**✅ Solution - Initial Analysis:**

```python
# Analysis de values faltantes
print("\n=== Valores Faltantes ===")
missing_summary = pd.DataFrame({
    'columna': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_summary[missing_summary['missing_count'] > 0])

# Analysis de duplicates
print(f"\nDuplicados: {df.duplicated().sum()}")

# Statistics descriptive
print("\n=== Statistics Descriptivas ===")
print(df.describe())
```

### Exercise 1.2: Handle missing values

```python
# Estrategias de imputation

# 1. Imputation por media (variables numerical continuas)
df['customer_age_imputed'] = df['customer_age'].fillna(df['customer_age'].median())

# 2. Imputation por median (variables con outliers)
df['shipping_cost_imputed'] = df['shipping_cost'].fillna(df['shipping_cost'].median())

# 3. Imputation por moda (variables categorical)
df['payment_method_imputed'] = df['payment_method'].fillna(df['payment_method'].mode()[0])

# Compare antes y after
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Customer age
axes[0].hist(df['customer_age'].dropna(), bins=20, alpha=0.5, label='Original', color='blue')
axes[0].hist(df['customer_age_imputed'], bins=20, alpha=0.5, label='Imputado', color='red')
axes[0].set_xlabel('Customer Age')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribution de Edad (antes vs after)')
axes[0].legend()

# Shipping cost
axes[1].hist(df['shipping_cost'].dropna(), bins=20, alpha=0.5, label='Original', color='blue')
axes[1].hist(df['shipping_cost_imputed'], bins=20, alpha=0.5, label='Imputado', color='red')
axes[1].set_xlabel('Shipping Cost')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribution de Costo de Shipment')
axes[1].legend()

plt.tight_layout()
plt.savefig('missing_values_imputation.png', dpi=150)
plt.show()

print("✅ Valores faltantes imputados")
```

### Exercise 1.3: Detect and manage outliers

```python
def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    Detecta outliers using method IQR
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    return outliers, lower_bound, upper_bound

# Detect outliers en product_price
outliers, lower, upper = detect_outliers_iqr(df, 'product_price')

print(f"\n=== Outliers en Product Price ===")
print(f"Limit inferior: ${lower:.2f}")
print(f"Limit superior: ${upper:.2f}")
print(f"Outliers detected: {len(outliers)}")
print(f"\nEjemplos de outliers:")
print(outliers[['order_id', 'product_price']].head())

# Visualization con boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot antes
axes[0].boxplot(df['product_price'].dropna())
axes[0].set_ylabel('Product Price')
axes[0].set_title('Antes de remover outliers')
axes[0].grid(True, alpha=0.3)

# Tratamiento: Winsorization (cap en upper bound)
df['product_price_clean'] = df['product_price'].clip(upper=upper)

# Boxplot after
axes[1].boxplot(df['product_price_clean'])
axes[1].set_ylabel('Product Price')
axes[1].set_title('After de winsorization')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outliers_treatment.png', dpi=150)
plt.show()

print("✅ Outliers tratados con winsorization")
```

### Exercise 1.4: Validate and correct inconsistencies

```python
# Detect cantidades negativas
invalid_qty = df[df['quantity'] < 0]

print(f"\n=== Valores Invalids en Quantity ===")
print(f"Registros con cantidad negativa: {len(invalid_qty)}")

# Correction: remover registros invalids
df_clean = df[df['quantity'] > 0].copy()

print(f"Registros removidos: {len(df) - len(df_clean)}")
print(f"Dataset clean: {len(df_clean)} registros")

# Calculate campo derivado
df_clean['total_price'] = (
    df_clean['product_price_clean'] * df_clean['quantity'] *
    (1 - df_clean['discount_pct']) + df_clean['shipping_cost_imputed']
)

print(f"\n✅ Dataset clean:")
print(df_clean.head())
print(f"\nTotal price range: ${df_clean['total_price'].min():.2f} - ${df_clean['total_price'].max():.2f}")
```

______________________________________________________________________

## 🚀 Part 2: Suggested Exercises

### Exercise 2.1: Complete Cleaning Pipeline

**Statement:**
Create a Function `clean_sales_data(df)` that:

1. Detected, Detect, Detects and impute missing values ​​(appropriate strategy per data type)
1. Identify outliers in all numerical columns
1. Validate that there are no negative values ​​in price/quantity
1. Return a clean DataFrame and a quality report

**Validation expected:**

```python
def clean_sales_data(df):
    # Tu code here
    pass

# Test
df_cleaned, quality_report = clean_sales_data(df)

assert df_cleaned.isnull().sum().sum() == 0, "Hay values faltantes sin tratar"
assert (df_cleaned['quantity'] < 0).sum() == 0, "Hay cantidades negativas"
assert quality_report['rows_removed'] >= 0, "Reporte invalid"

print("✅ Tests pasados")
```

### Exercise 2.2: Multi-variable Exploratory Analysis

**Statement:**
Generate an exploratory report that includes:

1. Distribution of sales by region
1. Correlation between price, quantity and discount
1. Total sales by payment method
1. Trend analysis of age vs price average

**Validation expected:**

- 4 visualizations (histograms, scatter plots, barplots)
- Report with insights principles
- Correlations calculated correctly

### Exercise 2.3: Feature Engineering

**Statement:**
Create new useful features for Analysis:

1. `price_per_unit` = product_price / quantity
1. `is_high_value` = total_price > 75th percentile
1. `age_group` = age bins (18-30, 31-45, 46-60, 61+)
1. `discount_category` = no discount / low / medium / high

**Validation expected:**

```python
assert 'price_per_unit' in df_engineered.columns
assert 'is_high_value' in df_engineered.columns
assert df_engineered['is_high_value'].dtype == bool
assert df_engineered['age_group'].nunique() == 4
```

### Exercise 2.4: Anomaly Detection

**Statement:**
Implement a failed order detector using:

1. Z-score to detect extreme prices
1. Business rules (e.g. discount > 50% with price > $300)
1. Impossible values ​​(shipping_cost > product_price)

Mark records with `is_anomalous` flag.

### Exercise 2.5: Data Quality Visualization

**Statement:**
Create a quality dashboard with:

1. Heatmap of missing values ​​per column
1. Distribution of outliers detected
1. Correlation Matrix
1. Distribution of categorical variables

Save as `data_quality_dashboard.png`.

______________________________________________________________________

## ✅ Skills Checklist

After completing this Practice, you should be able to:

- [ ] Detect missing values ​​and apply imputation strategies
- [ ] Identify outliers with statistical methods (IQR, Z-score)
- [ ] Apply Data transformations (winsorization, clipping)
- [ ] Validate Data integrity (valid ranges, consistency)
- [ ] Create effective exploratory visualizations
- [ ] Implement reproducible Cleaning pipelines
- [ ] Generate Data quality reporters
- [ ] Calculate derived features

______________________________________________________________________

## 📚 Additional Resources

**Useful libraries:**

- `pandas-profiling`: automatic EDA
- `missingno`: Display of missing data
- `PyOD`: Outlier detection
- `Great Expectations`: Data Validation

**Readings:**

- [Tidy Data by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Data Cleaning Tutorial - Kaggle](https://www.kaggle.com/learn/data-cleaning)
