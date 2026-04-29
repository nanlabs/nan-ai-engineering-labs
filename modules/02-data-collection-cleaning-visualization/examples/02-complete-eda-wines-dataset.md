# Example 02 — Complete EDA (Wine Dataset)

## Context

You have a dataset of wines with physical-chemical characteristics and quality evaluated by experts. Your objective is to perform a complete **Exploratory Data Analysis (EDA)** to understand:

- Distributions of variables
- Correlations between features
- Segmentation by quality
- Actionable Insights

## Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Wine Quality - UCI ML Repository)
# Version simplificada para example
np.random.seed(42)

data = {
    'fixed_acidity': np.random.uniform(4, 16, 100),
    'volatile_acidity': np.random.uniform(0.1, 1.6, 100),
    'citric_acid': np.random.uniform(0, 1, 100),
    'residual_sugar': np.random.uniform(0.9, 15, 100),
    'chlorides': np.random.uniform(0.01, 0.61, 100),
    'alcohol': np.random.uniform(8, 15, 100),
    'pH': np.random.uniform(2.7, 4.0, 100),
    'quality': np.random.choice([3, 4, 5, 6, 7, 8, 9], 100, p=[0.02, 0.10, 0.30, 0.35, 0.18, 0.04, 0.01])
}

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

print(f"Dimensions: {df.shape}")
print(df.head())
```

**Output:**

```
Dimensions: (1599, 12)

   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  ...  density    pH  sulphates  alcohol  quality
0            7.4              0.70         0.00             1.9      0.076  ...  0.99780  3.51       0.56      9.4        5
1            7.8              0.88         0.00             2.6      0.098  ...  0.99680  3.20       0.68      9.8        5
2            7.8              0.76         0.04             2.3      0.092  ...  0.99700  3.26       0.65      9.8        5
```

______________________________________________________________________

## 📊 Step 1: Initial inspection

### 1.1 General information

```python
print("=== INFORMATION DEL DATASET ===")
print(df.info())
```

**Output:**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1599 entries, 0 to 1598
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   fixed acidity         1599 non-null   float64
 1   volatile acidity      1599 non-null   float64
 2   citric acid           1599 non-null   float64
 3   residual sugar        1599 non-null   float64
 4   chlorides             1599 non-null   float64
 5   free sulfur dioxide   1599 non-null   float64
 6   total sulfur dioxide  1599 non-null   float64
 7   density               1599 non-null   float64
 8   pH                    1599 non-null   float64
 9   sulphates             1599 non-null   float64
 10  alcohol               1599 non-null   float64
 11  quality               1599 non-null   int64

✅ No hay values nulls
✅ Todos los types son numerical (correcto para este dataset)
```

### 1.2 Descriptive statistics

```python
print("\n=== STATISTICS DESCRIPTIVAS ===")
print(df.describe())
```

**Output:**

```
       fixed acidity  volatile acidity  citric acid  ...       pH  sulphates  alcohol   quality
count    1599.000000       1599.000000  1599.000000  ... 1599.000  1599.000  1599.000  1599.000
mean        8.319637          0.527821     0.270976  ...    3.311    0.658    10.423     5.636
std         1.741096          0.179060     0.194801  ...    0.154    0.169     1.066     0.808
min         4.600000          0.120000     0.000000  ...    2.740    0.330     8.400     3.000
25%         7.100000          0.390000     0.090000  ...    3.210    0.550     9.500     5.000
50%         7.900000          0.520000     0.260000  ...    3.310    0.620    10.200     6.000
75%         9.200000          0.640000     0.420000  ...    3.400    0.730    11.100     6.000
max        15.900000          1.580000     1.000000  ...    4.010    2.000    14.900     8.000
```

**Initial observations:**

- `quality`: range 3-8, mean ≈ 5.6 (scale of 0-10)
- `alcohol`: media ≈ 10.4%, range 8.4-14.9%
- `pH`: average ≈ 3.3 (acid, expected in wine)

______________________________________________________________________

## 📈 Step 2: Univariate analysis

### 2.1 variable Objective: `quality`

```python
# Distribution de quality
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histograma
axes[0].hist(df['quality'], bins=6, edgecolor='black', color='steelblue')
axes[0].set_title('Distribution de Calidad')
axes[0].set_xlabel('Calidad (3-8)')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(df['quality'].mean(), color='red', linestyle='--', label=f'Media: {df["quality"].mean():.2f}')
axes[0].legend()

# Graphic de barras
df['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Cantidad de Vinos por Calidad')
axes[1].set_xlabel('Calidad')
axes[1].set_ylabel('Cantidad')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

print("Distribution de quality:")
print(df['quality'].value_counts().sort_index())
```

**Output:**

```
Distribution de quality:
3      10
4      53
5     681   👈 Concentration en 5-6
6     638
7     199
8      18
```

**Insight:**

- Dataset **unbalanced**: 82% are quality 5-6
- Few wines of very low (3) or very high (8) quality
- Consider re-balancing or appropriate metrics (f1 instead of accuracy)

### 2.2 continuous variables: distributions

```python
# Seleccionar features numerical (excluir quality)
features = df.columns.drop('quality')

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(features):
    axes[idx].hist(df[col], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
    axes[idx].set_title(col)
    axes[idx].axvline(df[col].median(), color='red', linestyle='--', linewidth=1, label='Mediana')
    axes[idx].legend()

plt.tight_layout()
plt.show()
```

**Observations:**

- **alcohol:** approximately normal distribution, slight asymmetry to the right
- **volatile acidity:** long right tail (potential outliers)
- **citric acid:** many values ​​close to 0 (possible component absent in some wines)
- **residual sugar:** strongly skewed to the right (most wines are dry)

______________________________________________________________________

## 🔗 Step 3: Bivariate analysis

### 3.1 Correlation matrix

```python
# Calculate correlaciones
corr_matrix = df.corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matrix de Correlation', fontsize=16)
plt.tight_layout()
plt.show()

# Correlaciones con quality
print("\nCorrelaciones con 'quality' (ordenadas):")
quality_corr = corr_matrix['quality'].sort_values(ascending=False)
print(quality_corr)
```

**Output:**

```
Correlaciones con 'quality':
quality                 1.000000
alcohol                 0.476166  👈 Correlation positiva fuerte
sulphates               0.251397
citric acid             0.226373
fixed acidity           0.124052
residual sugar          0.013732
free sulfur dioxide    -0.050656
pH                     -0.057731
chlorides              -0.128907
density                -0.174919
total sulfur dioxide   -0.185100
volatile acidity       -0.390558  👈 Correlation negativa fuerte
```

**Insights clave:**

- **alcohol** has a stronger positive correlation with quality (0.48)
- **volatile acidity** has a strong negative correlation (-0.39)
- Wines of **higher quality** tend to have:
- More alcohol
- Less volatile acidity (associated with a vinegary taste)
- More citric acid

### 3.2 Scatter plots: features vs quality

```python
# Top 4 features correlacionadas con quality
top_features = quality_corr.drop('quality').abs().sort_values(ascending=False).head(4).index

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    # Scatter con jitter en quality para better visualization
    jitter = np.random.normal(0, 0.1, len(df))
    axes[idx].scatter(df[feature], df['quality'] + jitter, alpha=0.3, s=10)

    # Line de trend
    z = np.polyfit(df[feature], df['quality'], 1)
    p = np.poly1d(z)
    axes[idx].plot(df[feature].sort_values(), p(df[feature].sort_values()),
                   "r--", linewidth=2, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')

    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Quality')
    axes[idx].set_title(f'{feature} vs Quality (corr: {quality_corr[feature]:.2f})')
    axes[idx].legend()

plt.tight_layout()
plt.show()
```

**Observations:**

- Moderate linear relationships (not perfect, expected in real data)
- High variability: wines with the same alcohol can have very different qualities

______________________________________________________________________

## 📦 Step 4: Multivariate analysis

### 4.1 Segmentation by quality

```python
# Create categories: Baja (3-4), Media (5-6), Alta (7-8)
df['quality_category'] = pd.cut(df['quality'], bins=[2, 4, 6, 8],
                                 labels=['Baja', 'Media', 'Alta'])

print("Distribution por category:")
print(df['quality_category'].value_counts())
```

**Output:**

```
Media    1319  (82%)
Baja       63  (4%)
Alta      217  (14%)
```

### 4.2 Boxplots: comparison between groups

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

key_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']

for idx, feature in enumerate(key_features):
    df.boxplot(column=feature, by='quality_category', ax=axes[idx])
    axes[idx].set_title(f'{feature} por Calidad')
    axes[idx].set_xlabel('Category de Calidad')
    axes[idx].set_ylabel(feature)

plt.suptitle('')  # Remover qualification automatic
plt.tight_layout()
plt.show()
```

**Insights:**

- **Alcohol:** Medians clearly increasing with quality
- **Volatile acidity:** Decreasing medians with quality
- **Sulphates:** Slight increase with quality (preservative, prevents oxidation)

### 4.3 Pairplot (multiple relationships)

```python
# Seleccionar subset de features para legibilidad
selected_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'quality_category']

sns.pairplot(df[selected_features], hue='quality_category', palette='viridis',
             diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20})
plt.suptitle('Pairplot: Features Clave por Calidad', y=1.02, fontsize=16)
plt.show()
```

**Observations:**

- Visual separation between quality categories (although with overlap)
- Combinetions of features could improve Prediction (e.g. high alcohol + low volatile acidity)

______________________________________________________________________

## 📊 Step 5: Suggested Feature Engineering

### 5.1 Create derived features

```python
# Ratio de acidez fija / volatile (estabilidad)
df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 1e-5)

# Concentration total de acidez
df['total_acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']

# Ratio alcohol / sugar residual (dulzura relativa)
df['alcohol_sugar_ratio'] = df['alcohol'] / (df['residual sugar'] + 1)

print("Nuevas features:")
print(df[['acidity_ratio', 'total_acidity', 'alcohol_sugar_ratio']].describe())
```

### 5.2 Validate correlation of new features

```python
new_features = ['acidity_ratio', 'total_acidity', 'alcohol_sugar_ratio']
new_corr = df[new_features + ['quality']].corr()['quality'].drop('quality')

print("\nCorrelaciones de nuevas features con quality:")
print(new_corr.sort_values(ascending=False))
```

______________________________________________________________________

## 📝Executive summary

### ✅ Key findings

1. **variable Objective:**

   - Unbalanced dataset (82% quality average)
   - Few instances of extreme quality

1. **Main predictors:**

   - **Alcohol** (+): Mayor alcohol → mayor quality
- **Volatile acidity** (-): More volatile acidity → lower quality
- **Sulphates** (+): Preservative, positive correlation with quality
   - **Citric acid** (+): Freshness of the wine

1. **outliers:**

   - Presents in multiple features (especially `residual sugar`, `chlorides`)
   - Require treatment before modeling

1. **Non-linear relationships:**

   - Scatter plots suggest non-perfectly linear relationships
   - Consider nonlinear Models (Random Forest, XGBoost)

### 🚀 Recommendations for modeling

1. **Preprocessing:**

   - Scaling features (StandardScaler)
   - Handle outliers (IQR clipping or Robust Scaler)
   - Consider re-balancing (SMOTE, class weights)

1. **Features:**

   - Use correlated top features
   - Test derived features (ratios)
   - Consider PCA if multicollinearity is present

1. **Models:**

   - Baseline: Logistic Regression
   - Nonlinear models: Random Forest, XGBoost
   - Metric: f1-score macro (unbalanced dataset)

1. **Validation:**

   - Stratified K-Fold (preserve quality distribution)
- Validate performance by quality category

______________________________________________________________________

## 🎓 Lessons learned

### EDA Workflow

1. **Inspection** → `info()`, `describe()`, `isnull()`
1. **Univariate** → Distributions, outliers, asymmetries
1. **Bivariate** → Correlations, scatter plots
1. **Multivariate** → Segmentation, pairplots, interactions
1. **Feature Engineering** → Create derived features and validate

### Effective visualizations

- **Heatmap:** Multiple correlations at a glance
- **Boxplots:** Comparisons between groups
- **Pairplots:** Multidimensional relationships
- **Scatter + Trend:** Validate linear relationships

### Considerations

- EDA guides preprocessing and modeling decisions
- Do not jump into modeling without understanding the Data
- Document findings to communicate to non-technical stakeholders
