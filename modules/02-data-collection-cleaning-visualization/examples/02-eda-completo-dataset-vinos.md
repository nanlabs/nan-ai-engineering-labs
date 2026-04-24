# Ejemplo 02 — EDA Completo (Dataset de Vinos)

## Contexto

Tienes un dataset de vinos con características físico-químicas y calidad evaluada por expertos. Tu objetivo es realizar un **Análisis Exploratorio de Datos (EDA)** completo para entender:

- Distribuciones de variables
- Correlaciones entre features
- Segmentación por calidad
- Insights accionables

## Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset (Wine Quality - UCI ML Repository)
# Versión simplificada para ejemplo
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

print(f"Dimensiones: {df.shape}")
print(df.head())
```

**Salida:**

```
Dimensiones: (1599, 12)

   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  ...  density    pH  sulphates  alcohol  quality
0            7.4              0.70         0.00             1.9      0.076  ...  0.99780  3.51       0.56      9.4        5
1            7.8              0.88         0.00             2.6      0.098  ...  0.99680  3.20       0.68      9.8        5
2            7.8              0.76         0.04             2.3      0.092  ...  0.99700  3.26       0.65      9.8        5
```

______________________________________________________________________

## 📊 Paso 1: Inspección inicial

### 1.1 Información general

```python
print("=== INFORMACIÓN DEL DATASET ===")
print(df.info())
```

**Salida:**

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

✅ No hay valores nulos
✅ Todos los tipos son numéricos (correcto para este dataset)
```

### 1.2 Estadísticas descriptivas

```python
print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(df.describe())
```

**Salida:**

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

**Observaciones iniciales:**

- `quality`: rango 3-8, media ≈ 5.6 (escala de 0-10)
- `alcohol`: media ≈ 10.4%, rango 8.4-14.9%
- `pH`: media ≈ 3.3 (ácido, esperado en vino)

______________________________________________________________________

## 📈 Paso 2: Análisis univariado

### 2.1 Variable objetivo: `quality`

```python
# Distribución de calidad
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histograma
axes[0].hist(df['quality'], bins=6, edgecolor='black', color='steelblue')
axes[0].set_title('Distribución de Calidad')
axes[0].set_xlabel('Calidad (3-8)')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(df['quality'].mean(), color='red', linestyle='--', label=f'Media: {df["quality"].mean():.2f}')
axes[0].legend()

# Gráfico de barras
df['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Cantidad de Vinos por Calidad')
axes[1].set_xlabel('Calidad')
axes[1].set_ylabel('Cantidad')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

print("Distribución de calidad:")
print(df['quality'].value_counts().sort_index())
```

**Salida:**

```
Distribución de calidad:
3      10
4      53
5     681   👈 Concentración en 5-6
6     638
7     199
8      18
```

**Insight:**

- Dataset **desbalanceado**: 82% son calidad 5-6
- Pocos vinos de calidad muy baja (3) o muy alta (8)
- Considerar re-balanceo o métricas apropiadas (F1 en lugar de accuracy)

### 2.2 Variables continuas: distribuciones

```python
# Seleccionar features numéricas (excluir quality)
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

**Observaciones:**

- **alcohol:** distribución aproximadamente normal, ligera asimetría a derecha
- **volatile acidity:** cola larga a derecha (outliers potenciales)
- **citric acid:** muchos valores cercanos a 0 (posible componente ausente en algunos vinos)
- **residual sugar:** fuertemente sesgada a derecha (la mayoría de vinos son secos)

______________________________________________________________________

## 🔗 Paso 3: Análisis bivariado

### 3.1 Matriz de correlación

```python
# Calcular correlaciones
corr_matrix = df.corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación', fontsize=16)
plt.tight_layout()
plt.show()

# Correlaciones con quality
print("\nCorrelaciones con 'quality' (ordenadas):")
quality_corr = corr_matrix['quality'].sort_values(ascending=False)
print(quality_corr)
```

**Salida:**

```
Correlaciones con 'quality':
quality                 1.000000
alcohol                 0.476166  👈 Correlación positiva fuerte
sulphates               0.251397
citric acid             0.226373
fixed acidity           0.124052
residual sugar          0.013732
free sulfur dioxide    -0.050656
pH                     -0.057731
chlorides              -0.128907
density                -0.174919
total sulfur dioxide   -0.185100
volatile acidity       -0.390558  👈 Correlación negativa fuerte
```

**Insights clave:**

- **alcohol** tiene correlación positiva más fuerte con calidad (0.48)
- **volatile acidity** tiene correlación negativa fuerte (-0.39)
- Vinos de **mayor calidad** tienden a tener:
  - Más alcohol
  - Menos acidez volátil (asociada a sabor avinagrado)
  - Más ácido cítrico

### 3.2 Scatter plots: features vs quality

```python
# Top 4 features correlacionadas con quality
top_features = quality_corr.drop('quality').abs().sort_values(ascending=False).head(4).index

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    # Scatter con jitter en quality para mejor visualización
    jitter = np.random.normal(0, 0.1, len(df))
    axes[idx].scatter(df[feature], df['quality'] + jitter, alpha=0.3, s=10)

    # Línea de tendencia
    z = np.polyfit(df[feature], df['quality'], 1)
    p = np.poly1d(z)
    axes[idx].plot(df[feature].sort_values(), p(df[feature].sort_values()),
                   "r--", linewidth=2, label=f'Tendencia: {z[0]:.2f}x + {z[1]:.2f}')

    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Quality')
    axes[idx].set_title(f'{feature} vs Quality (corr: {quality_corr[feature]:.2f})')
    axes[idx].legend()

plt.tight_layout()
plt.show()
```

**Observaciones:**

- Relaciones lineales moderadas (no perfectas, esperable en datos reales)
- Alta variabilidad: vinos con mismo alcohol pueden tener calidades muy diferentes

______________________________________________________________________

## 📦 Paso 4: Análisis multivariado

### 4.1 Segmentación por calidad

```python
# Crear categorías: Baja (3-4), Media (5-6), Alta (7-8)
df['quality_category'] = pd.cut(df['quality'], bins=[2, 4, 6, 8],
                                 labels=['Baja', 'Media', 'Alta'])

print("Distribución por categoría:")
print(df['quality_category'].value_counts())
```

**Salida:**

```
Media    1319  (82%)
Baja       63  (4%)
Alta      217  (14%)
```

### 4.2 Boxplots: comparación entre grupos

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

key_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']

for idx, feature in enumerate(key_features):
    df.boxplot(column=feature, by='quality_category', ax=axes[idx])
    axes[idx].set_title(f'{feature} por Calidad')
    axes[idx].set_xlabel('Categoría de Calidad')
    axes[idx].set_ylabel(feature)

plt.suptitle('')  # Remover título automático
plt.tight_layout()
plt.show()
```

**Insights:**

- **Alcohol:** Medianas claramente crecientes con calidad
- **Volatile acidity:** Medianas decrecientes con calidad
- **Sulphates:** Ligero incremento con calidad (conservante, previene oxidación)

### 4.3 Pairplot (relaciones múltiples)

```python
# Seleccionar subset de features para legibilidad
selected_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'quality_category']

sns.pairplot(df[selected_features], hue='quality_category', palette='viridis',
             diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20})
plt.suptitle('Pairplot: Features Clave por Calidad', y=1.02, fontsize=16)
plt.show()
```

**Observaciones:**

- Separación visual entre categorías de calidad (aunque con overlap)
- Combinaciones de features podrían mejorar predicción (ej: alcohol alto + volatile acidity baja)

______________________________________________________________________

## 📊 Paso 5: Feature Engineering sugerido

### 5.1 Crear features derivadas

```python
# Ratio de acidez fija / volátil (estabilidad)
df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 1e-5)

# Concentración total de acidez
df['total_acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']

# Ratio alcohol / azúcar residual (dulzura relativa)
df['alcohol_sugar_ratio'] = df['alcohol'] / (df['residual sugar'] + 1)

print("Nuevas features:")
print(df[['acidity_ratio', 'total_acidity', 'alcohol_sugar_ratio']].describe())
```

### 5.2 Validar correlación de nuevas features

```python
new_features = ['acidity_ratio', 'total_acidity', 'alcohol_sugar_ratio']
new_corr = df[new_features + ['quality']].corr()['quality'].drop('quality')

print("\nCorrelaciones de nuevas features con quality:")
print(new_corr.sort_values(ascending=False))
```

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Hallazgos clave

1. **Variable objetivo:**

   - Dataset desbalanceado (82% calidad media)
   - Pocas instancias de calidad extrema

1. **Predictores principales:**

   - **Alcohol** (+): Mayor alcohol → mayor calidad
   - **Volatile acidity** (-): Más acidez volátil → menor calidad
   - **Sulphates** (+): Conservante, correlación positiva con calidad
   - **Citric acid** (+): Frescura del vino

1. **Outliers:**

   - Presentes en multiple features (especialmente `residual sugar`, `chlorides`)
   - Requieren tratamiento antes de modelado

1. **Relaciones no lineales:**

   - Scatter plots sugieren relaciones no perfectamente lineales
   - Considerar modelos no lineales (Random Forest, XGBoost)

### 🚀 Recomendaciones para modelado

1. **Preprocesamiento:**

   - Escalar features (StandardScaler)
   - Manejar outliers (IQR clipping o Robust Scaler)
   - Considerar re-balanceo (SMOTE, class weights)

1. **Features:**

   - Usar top features correlacionadas
   - Probar features derivadas (ratios)
   - Considerar PCA si hay multicolinealidad

1. **Modelos:**

   - Baseline: Logistic Regression
   - Modelos no lineales: Random Forest, XGBoost
   - Métrica: F1-score macro (dataset desbalanceado)

1. **Validación:**

   - Stratified K-Fold (preservar distribución de quality)
   - Validar performance por categoría de calidad

______________________________________________________________________

## 🎓 Lecciones aprendidas

### Workflow de EDA

1. **Inspección** → `info()`, `describe()`, `isnull()`
1. **Univariado** → Distribuciones, outliers, asimetrías
1. **Bivariado** → Correlaciones, scatter plots
1. **Multivariado** → Segmentación, pairplots, interacciones
1. **Feature Engineering** → Crear features derivadas y validar

### Visualizaciones efectivas

- **Heatmap:** Correlaciones múltiples de un vistazo
- **Boxplots:** Comparaciones entre grupos
- **Pairplots:** Relaciones multidimensionales
- **Scatter + tendencia:** Validar relaciones lineales

### Consideraciones

- EDA guía decisiones de preprocesamiento y modelado
- No saltar a modelar sin entender los datos
- Documentar hallazgos para comunicar a stakeholders no técnicos
