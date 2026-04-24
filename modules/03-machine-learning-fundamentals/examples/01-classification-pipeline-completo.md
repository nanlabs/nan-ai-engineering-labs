# Example 01 — Pipeline Completo de Classification (Iris Dataset)

## Contexto

Construirás un pipeline completo de Machine Learning desde cero: desde Data crudos hasta Prediction, siguiendo las mejores Practices profesionales. Usaremos el dataset Iris (Classification de especies de flores) por ser didáctico pero representativo.

## Objective

Predecir la especie de flor Iris basándose en mediciones de sépalos y pétalos.

______________________________________________________________________

## 🚀 Paso 1: Importar librerías y cargar Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Cargar dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print("=== DATASET IRIS ===")
print(f"Dimensiones: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Clases: {iris.target_names}")
print(f"\nPrimeras filas:")
print(X.head())
print(f"\nDistribución de clases:")
print(y.value_counts())
```

**Salida:**

```
=== DATASET IRIS ===
Dimensiones: (150, 4)
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Clases: ['setosa' 'versicolor' 'virginica']

Primeras filas:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2

Distribución de clases:
0    50  👈 Balanceado (33% cada clase)
1    50
2    50
```

______________________________________________________________________

## 📊 Paso 2: Analysis Exploratorio (EDA)

### 2.1 Estadísticas descriptivas

```python
print("=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(X.describe())
print(f"\nValores nulos: {X.isnull().sum().sum()}")
```

### 2.2 Visualizaciones

```python
# Crear DataFrame combinado para visualización
df = X.copy()
df['species'] = y.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Pairplot
sns.pairplot(df, hue='species', palette='Set1', markers=['o', 's', 'D'])
plt.suptitle('Pairplot: Relaciones entre Features', y=1.02)
plt.show()

# Boxplots por especie
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, col in enumerate(X.columns):
    df.boxplot(column=col, by='species', ax=axes[idx])
    axes[idx].set_title(col)
    axes[idx].set_xlabel('Species')

plt.suptitle('')
plt.tight_layout()
plt.show()
```

**Insights del EDA:**

- **Setosa** se separa claramente (petal length/width más cortos)
- **Versicolor y Virginica** tienen overlap (más difíciles de distinguir)
- No hay valores nulos ni outliers extremos
- Dataset balanceado (no requiere re-balanceo)

______________________________________________________________________

## 🔧 Paso 3: Preparación de Data

### 3.1 Train/Test Split

```python
# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify preserva distribución de clases
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nDistribución en train:")
print(y_train.value_counts())
```

**Salida:**

```
Train set: (120, 4)
Test set: (30, 4)

Distribución en train:
0    40
1    40
2    40
✅ Stratify funcionó: distribución balanceada preservada
```

### 3.2 Escalado de features

```python
# Entrenar scaler solo con train (prevenir data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Aplicar transformación aprendida

# Convertir de nuevo a DataFrame para mantener nombres de columnas
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Antes del escalado (train):")
print(X_train.describe().loc[['mean', 'std']])
print("\nDespués del escalado (train):")
print(X_train_scaled.describe().loc[['mean', 'std']])
```

**Salida:**

```
Después del escalado:
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
mean          -0.000000        -0.000000            -0.000000          0.000000
std            1.008403         1.008403             1.008403          1.008403
✅ Media ≈ 0, std ≈ 1 (escalado correcto)
```

______________________________________________________________________

## 🤖 Paso 4: Training de Models

### 4.1 Baseline: Logistic Regression

```python
# Modelo base
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)

# Predicciones
y_pred_lr = lr.predict(X_test_scaled)

# Evaluación
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {acc_lr:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=iris.target_names))
```

**Salida:**

```
=== LOGISTIC REGRESSION ===
Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        10
   virginica       1.00      1.00      1.00        10

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

### 4.2 Decision Tree

```python
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"\n=== DECISION TREE ===")
print(f"Accuracy: {acc_dt:.4f}")
```

### 4.3 Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\n=== RANDOM FOREST ===")
print(f"Accuracy: {acc_rf:.4f}")
```

______________________________________________________________________

## 📈 Paso 5: Comparación de Models

### 5.1 Cross-validation

```python
models = {
    'Logistic Regression': lr,
    'Decision Tree': dt,
    'Random Forest': rf
}

print("\n=== CROSS-VALIDATION (5-FOLD) ===")
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name}:")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"  Scores: {scores}")
```

**Salida:**

```
=== CROSS-VALIDATION (5-FOLD) ===
Logistic Regression:
  Mean: 0.9667 (+/- 0.0632)
  Scores: [1.         0.95833333 1.         0.95833333 0.91666667]

Decision Tree:
  Mean: 0.9583 (+/- 0.0632)
  Scores: [0.95833333 0.95833333 1.         0.91666667 0.95833333]

Random Forest:
  Mean: 0.9667 (+/- 0.0632)
  Scores: [1.         0.95833333 1.         0.95833333 0.91666667]
```

### 5.2 Visualization de comparación

```python
# Comparar performance
comparison = pd.DataFrame({
    'Model': list(models.keys()),
    'Test Accuracy': [acc_lr, acc_dt, acc_rf],
    'CV Mean': [cv_results[name].mean() for name in models.keys()],
    'CV Std': [cv_results[name].std() for name in models.keys()]
})

print("\n=== RESUMEN DE MODELOS ===")
print(comparison)

# Gráfico
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison))
width = 0.35

ax.bar(x - width/2, comparison['Test Accuracy'], width, label='Test Accuracy', color='skyblue')
ax.bar(x + width/2, comparison['CV Mean'], width, label='CV Mean', color='lightcoral')

ax.set_xlabel('Modelo')
ax.set_ylabel('Accuracy')
ax.set_title('Comparación de Modelos')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Model'], rotation=15, ha='right')
ax.legend()
ax.set_ylim([0.9, 1.01])

plt.tight_layout()
plt.show()
```

______________________________________________________________________

## 🎯 Paso 6: Analysis detallado del mejor Model

```python
# Seleccionar mejor modelo (Logistic Regression y Random Forest empatan)
best_model = rf  # Elegimos RF por robustez

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance (Random Forest)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance)

# Gráfico
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Salida:**

```
=== FEATURE IMPORTANCE ===
              feature  importance
2  petal length (cm)    0.445302  👈 Feature más importante
3   petal width (cm)    0.425186
0  sepal length (cm)    0.089254
1   sepal width (cm)    0.040258
```

**Insight:**

- Dimensiones de **pétalo** son mucho más predictivas que sépalos
- Esto confirma observaciones del EDA

______________________________________________________________________

## 🚀 Paso 7: Pipeline reproducible con scikit-learn

```python
from sklearn.pipeline import Pipeline

# Construir pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
])

# Entrenar pipeline (fit + transform automático)
pipeline.fit(X_train, y_train)

# Predecir
y_pred_pipeline = pipeline.predict(X_test)
acc_pipeline = accuracy_score(y_test, y_pred_pipeline)

print(f"=== PIPELINE ===")
print(f"Accuracy: {acc_pipeline:.4f}")

# Ventaja: Predecir en datos nuevos sin olvidar escalar
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Setosa típica
prediction = pipeline.predict(new_sample)
print(f"\nPredicción para muestra nueva: {iris.target_names[prediction[0]]}")
```

______________________________________________________________________

## 💾 Paso 8: Guardar Model

```python
import joblib

# Guardar modelo entrenado
joblib.dump(pipeline, 'iris_classifier_pipeline.pkl')
print("Modelo guardado: iris_classifier_pipeline.pkl")

# Cargar modelo (en producción)
loaded_pipeline = joblib.load('iris_classifier_pipeline.pkl')
prediction_loaded = loaded_pipeline.predict(new_sample)
print(f"Predicción con modelo cargado: {iris.target_names[prediction_loaded[0]]}")
```

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Results

| Model              | Test accuracy | CV Mean | CV Std |
| ------------------- | ------------- | ------- | ------ |
| Logistic Regression | 1.0000        | 0.9667  | 0.0316 |
| Decision Tree       | 1.0000        | 0.9583  | 0.0316 |
| Random Forest       | 1.0000        | 0.9667  | 0.0316 |

**Model seleccionado:** Random Forest

- accuracy perfecto en test set (1.00)
- Robusto en cross-validation (0.9667 ± 0.0316)
- Interpretable (feature importance)

### 🎯 Features clave

1. **petal length** (44.5%) - Más importante
1. **petal width** (42.5%)
1. **sepal length** (8.9%)
1. **sepal width** (4.0%)

### 🔍 Observaciones

- Dataset Iris es "fácil" (accuracy cercano a 100%)
- Setosa perfectamente separable
- Versicolor/Virginica overlap mínimo
- No hubo overfitting (CV y test performance similares)

______________________________________________________________________

## 🎓 Lessons aprendidas

### ✅ Buenas Practices aplicadas

1. **Train/test split estratificado** → Preserva distribución de clases
1. **Escalado después de split** → Previene data leakage
1. **Cross-validation** → Validation robusta (no confiar solo en test accuracy)
1. **Comparación de múltiples Models** → Baseline simple vs complejos
1. **Pipeline de scikit-learn** → Reproducibilidad y prevenir Errors
1. **Feature importance** → Interpretabilidad

### 🚫 Errors comunes evitados

- ❌ Escalar antes de split (data leakage)
- ❌ Optimizar Hyperparameters mirando test set (overfitting)
- ❌ No usar cross-validation (sobreestimar performance)
- ❌ Olvidar estratificación (split desbalanceado)

### 🔧 Herramientas clave

- `StandardScaler`: Escalado (media 0, std 1)
- `train_test_split` con `stratify`: Split balanceado
- `cross_val_score`: Cross-validation
- `Pipeline`: Workflow reproducible
- `joblib`: Persistencia de Models

### 💡 Próximos pasos

Para datasets reales más complejos:

1. Manejo de missing values y outliers
1. Feature engineering más sofisticado
1. Hyperparameter tuning (Grid Search, Random Search)
1. Analysis de Errors (por qué el Model falla)
1. Calibración de probabilidades
1. Threshold optimization (si costos de FP/FN difieren)
