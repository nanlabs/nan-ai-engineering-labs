# Práctica 01 — Regresión Lineal y Logística

## 🎯 Objetivos

- Implementar regresión lineal desde cero y con sklearn
- Entrenar modelo de regresión logística
- Evaluar modelos con métricas apropiadas
- Interpretar coeficientes y hacer predicciones

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Regresión Lineal Simple

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset: predicción de precios de casas
np.random.seed(42)
n = 200

# Feature: tamaño en m²
size = np.random.uniform(50, 250, n)

# Target: precio (relación lineal + ruido)
price = 50000 + size * 1500 + np.random.normal(0, 20000, n)

df = pd.DataFrame({'size_m2': size, 'price': price})

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(df['size_m2'], df['price'], alpha=0.6, color='steelblue')
plt.xlabel('Size (m²)')
plt.ylabel('Price ($)')
plt.title('House Price vs Size')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_size_price.png', dpi=150)
plt.show()

print("Dataset creado:", df.shape)
print(df.head())
```

**✅ Solución - Entrenar modelo:**

```python
# Split
X = df[['size_m2']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar
model = LinearRegression()
model.fit(X_train, y_train)

# Coeficientes
print(f"\n=== Modelo Entrenado ===")
print(f"Intercept (β₀): ${model.intercept_:,.2f}")
print(f"Coefficient (β₁): ${model.coef_[0]:,.2f} per m²")

# Predicción
y_pred = model.predict(X_test)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Performance ===")
print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²: {r2:.4f}")

# Visualizar predicciones
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.6, label='Real', color='steelblue')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicción')
plt.xlabel('Size (m²)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: Predictions vs Real')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('linear_regression_fit.png', dpi=150)
plt.show()
```

### Ejercicio 1.2: Regresión Múltiple

```python
# Agregar más features
df['bedrooms'] = np.random.randint(1, 6, n)
df['age_years'] = np.random.randint(0, 50, n)
df['distance_center_km'] = np.random.uniform(1, 30, n)

# Recalcular precio con múltiples factores
df['price'] = (
    50000 +
    df['size_m2'] * 1500 +
    df['bedrooms'] * 15000 +
    -df['age_years'] * 1000 +
    -df['distance_center_km'] * 2000 +
    np.random.normal(0, 20000, n)
)

# Entrenar modelo múltiple
feature_cols = ['size_m2', 'bedrooms', 'age_years', 'distance_center_km']
X = df[feature_cols]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Coeficientes
print("\n=== Regresión Múltiple ===")
for feature, coef in zip(feature_cols, model_multi.coef_):
    print(f"{feature}: ${coef:,.2f}")
print(f"Intercept: ${model_multi.intercept_:,.2f}")

# Performance
y_pred_multi = model_multi.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))

print(f"\nR²: {r2_multi:.4f}")
print(f"RMSE: ${rmse_multi:,.2f}")

# Gráfico de predicciones vs reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_multi, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Real Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Multiple Linear Regression: Predicted vs Real')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('multiple_regression_predictions.png', dpi=150)
plt.show()
```

### Ejercicio 1.3: Regresión Logística (Clasificación Binaria)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset: aprobación de crédito
np.random.seed(42)
n = 500

df_credit = pd.DataFrame({
    'income': np.random.uniform(20000, 120000, n),
    'credit_score': np.random.randint(300, 850, n),
    'debt': np.random.uniform(0, 50000, n),
    'age': np.random.randint(18, 70, n)
})

# Target: aprobación basada en reglas
df_credit['approved'] = (
    (df_credit['income'] > 40000) &
    (df_credit['credit_score'] > 600) &
    (df_credit['debt'] < 30000)
).astype(int)

print("=== Credit Approval Dataset ===")
print(df_credit.head())
print(f"\nApproval rate: {df_credit['approved'].mean():.2%}")

# Split
X = df_credit[['income', 'credit_score', 'debt', 'age']]
y = df_credit['approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

# Predicciones
y_pred = log_model.predict(X_test)
y_proba = log_model.predict_proba(X_test)[:, 1]

# Evaluación
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n=== Logistic Regression Performance ===")
print(f"Accuracy: {acc:.4f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizar coeficientes
plt.figure(figsize=(10, 6))
features = X.columns
coefs = log_model.coef_[0]
plt.barh(features, coefs, color='steelblue', alpha=0.7)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Coefficients')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('logistic_coefficients.png', dpi=150)
plt.show()
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Regresión Lineal desde Cero

**Enunciado:**
Implementa regresión lineal sin sklearn usando gradient descent:

```python
def linear_regression_gd(X, y, learning_rate=0.01, epochs=1000):
    """
    Implementa regresión lineal con gradient descent

    Returns:
        weights, bias, loss_history
    """
    # Tu código aquí
    pass
```

Valida que los coeficientes sean similares a sklearn.

### Ejercicio 2.2: Regularización (Ridge y Lasso)

**Enunciado:**
Compara modelos con regularización:

1. Linear Regression (sin regularización)
1. Ridge (L2 regularization)
1. Lasso (L1 regularization)

Usa `Ridge` y `Lasso` de sklearn con diferentes valores de `alpha`.
Visualiza cómo cambian los coeficientes.

### Ejercicio 2.3: Polynomial Regression

**Enunciado:**
Ajusta regresión polinomial de grados 1, 2, 3, 5:

- Usa `PolynomialFeatures` de sklearn
- Compara RMSE y R² en test set
- Visualiza overfitting en grado 5

### Ejercicio 2.4: Threshold Tuning en Logistic Regression

**Enunciado:**
Por defecto, threshold = 0.5 para clasificación.
Varía threshold de 0.1 a 0.9 y grafica:

- Precision vs Threshold
- Recall vs Threshold
- F1-Score vs Threshold

Encuentra threshold óptimo para maximizar F1.

### Ejercicio 2.5: ROC Curve y AUC

**Enunciado:**
Genera ROC curve para modelo logístico:

1. Calcula TPR y FPR para diferentes thresholds
1. Grafica ROC curve
1. Calcula AUC (área bajo la curva)
1. Compara con `roc_curve` y `roc_auc_score` de sklearn

______________________________________________________________________

## ✅ Checklist de Competencias

- [ ] Entrenar regresión lineal simple y múltiple
- [ ] Interpretar coeficientes (β₀, β₁, ...)
- [ ] Evaluar con MSE, RMSE, R²
- [ ] Detectar overfitting comparando train vs test
- [ ] Implementar regresión logística para clasificación
- [ ] Calcular métricas: accuracy, precision, recall, F1
- [ ] Interpretar confusion matrix
- [ ] Ajustar threshold de clasificación
- [ ] Aplicar regularización (Ridge, Lasso)

______________________________________________________________________

## 📚 Recursos

- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
