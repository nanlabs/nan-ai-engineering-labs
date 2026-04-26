# Practice 01 — Linear and Logistic Regression

## 🎯 Objectives

- Implement linear regression from zero and with sklearn
- Logistic regression train model
- Evaluate Models with appropriate Metrics
- Interpret coefficients and make predictions

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Simple Linear Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset: prediction de precios de casas
np.random.seed(42)
n = 200

# Feature: size en m²
size = np.random.uniform(50, 250, n)

# Target: price (relationship lineal + noise)
price = 50000 + size * 1500 + np.random.normal(0, 20000, n)

df = pd.DataFrame({'size_m2': size, 'price': price})

# Visualization
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

**✅ Solution - Train Model:**

```python
# Split
X = df[['size_m2']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Coeficientes
print(f"\n=== Model Entrenado ===")
print(f"Intercept (β₀): ${model.intercept_:,.2f}")
print(f"Coefficient (β₁): ${model.coef_[0]:,.2f} per m²")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Performance ===")
print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²: {r2:.4f}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.6, label='Real', color='steelblue')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prediction')
plt.xlabel('Size (m²)')
plt.ylabel('Price ($)')
plt.title('Linear Regression: Predictions vs Real')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('linear_regression_fit.png', dpi=150)
plt.show()
```

### Exercise 1.2: Multiple Regression

```python
# Add más features
df['bedrooms'] = np.random.randint(1, 6, n)
df['age_years'] = np.random.randint(0, 50, n)
df['distance_center_km'] = np.random.uniform(1, 30, n)

# Recalcular price con multiple factors
df['price'] = (
    50000 +
    df['size_m2'] * 1500 +
    df['bedrooms'] * 15000 +
    -df['age_years'] * 1000 +
    -df['distance_center_km'] * 2000 +
    np.random.normal(0, 20000, n)
)

# Train model multiple
feature_cols = ['size_m2', 'bedrooms', 'age_years', 'distance_center_km']
X = df[feature_cols]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Coeficientes
print("\n=== Regression Multiple ===")
for feature, coef in zip(feature_cols, model_multi.coef_):
    print(f"{feature}: ${coef:,.2f}")
print(f"Intercept: ${model_multi.intercept_:,.2f}")

# Performance
y_pred_multi = model_multi.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))

print(f"\nR²: {r2_multi:.4f}")
print(f"RMSE: ${rmse_multi:,.2f}")

# Graphic de predictions vs real
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

### Exercise 1.3: Logistic Regression (Binary Classification)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset: approval de credit
np.random.seed(42)
n = 500

df_credit = pd.DataFrame({
    'income': np.random.uniform(20000, 120000, n),
    'credit_score': np.random.randint(300, 850, n),
    'debt': np.random.uniform(0, 50000, n),
    'age': np.random.randint(18, 70, n)
})

# Target: approval basada en reglas
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

# Train
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

# Predictions
y_pred = log_model.predict(X_test)
y_proba = log_model.predict_proba(X_test)[:, 1]

# Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n=== Logistic Regression Performance ===")
print(f"Accuracy: {acc:.4f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize coeficientes
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

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Linear Regression from Zero

**Statement:**
Implement regression lineal sin sklearn using gradient descent:

```python
def linear_regression_gd(X, y, learning_rate=0.01, epochs=1000):
    """
    Implement regression lineal con gradient descent

    Returns:
        weights, bias, loss_history
    """
    # Tu code here
    pass
```

Validates that the coefficients are similar to sklearn.

### Exercise 2.2: Regularization (Ridge and Lasso)

**Statement:**
Compare Models with Regularization:

1. Linear Regression (sin Regularization)
1. Ridge (L2 regularization)
1. Lasso (L1 regularization)

Use sklearn's `Ridge` and `Lasso` with different values ​​of `alpha`.
Visualize how the coefficients change.

### Exercise 2.3: Polynomial Regression

**Statement:**
Fit polynomial regression of degrees 1, 2, 3, 5:

- Use sklearn's `PolynomialFeatures`
- Compare RMSE and R² in test set
- Visualize overfitting in grade 5

### Exercise 2.4: Threshold Tuning in Logistic Regression

**Statement:**
By default, threshold = 0.5 for Classification.
Vary threshold from 0.1 to 0.9 and graph:

- Precision vs Threshold
- recall vs Threshold
- f1-Score vs Threshold

Find optimal threshold to maximize f1.

### Exercise 2.5: ROC Curve and auc

**Statement:**
Generate ROC curve for logistic Model:

1. Calculate TPR and FPR for different thresholds
1. Grafica ROC curve
1. Calculate auc (area under the curve)
1. Compare with sklearn's `roc_curve` and `roc_auc_score`

______________________________________________________________________

## ✅ Skills Checklist

- [ ] Simple and multiple linear train regression
- [ ] Interpret coeficientes (β₀, β₁, ...)
- [ ] Evaluate with MSE, RMSE, R²
- [ ] Detect overfitting comparando train vs test
- [ ] Implement logistic regression for Classification
- [ ] Calculate Metrics: accuracy, precision, recall, f1
- [ ] Interpret confusion matrix
- [ ] Adjust Classification threshold
- [ ] Apply Regularization (Ridge, Lasso)

______________________________________________________________________

## 📚 Resources

- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
