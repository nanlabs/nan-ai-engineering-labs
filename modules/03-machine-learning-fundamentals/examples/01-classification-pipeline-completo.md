# Example 01 — Complete Classification Pipeline (Iris Dataset)

## Context

You will build a complete Machine Learning pipeline from scratch: from raw Data to Prediction, following professional best practices. We will use the Iris dataset (Classification of flower species) because it is didactic but representative.

## Objective

Predict the Iris flower species based on measurements of sepals and petals.

______________________________________________________________________

## 🚀 Step 1: Import libraries and load Data

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

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print("=== DATASET IRIS ===")
print(f"Dimensions: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Cases, Clashes, Classes: {iris.target_names}")
print(f"\nPrimeras rows:")
print(X.head())
print(f"\nDistribución de classes:")
print(y.value_counts())
```

**Output:**

```
=== DATASET IRIS ===
Dimensions: (150, 4)
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Cases, Clashes, Classes: ['setosa' 'versicolor' 'virginica']

Primeras rows:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2

Distribution de classes:
0    50  👈 Balanced (33% each clause, class)
1    50
2    50
```

______________________________________________________________________

## 📊 Step 2: Exploratory Analysis (EDA)

### 2.1 Descriptive statistics

```python
print("=== STATISTICS DESCRIPTIVAS ===")
print(X.describe())
print(f"\nValores nulls: {X.isnull().sum().sum()}")
```

### 2.2 Visualizations

```python
# Create DataFrame combinado para visualization
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

**EDA Insights:**

- **Setosa** separates clearly (shorter petal length/width)
- **Versicolor and Virginica** have overlap (more difficult to distinguish)
- There are no null values ​​or extreme outliers
- Balanced dataset (does not require re-balancing)

______________________________________________________________________

## 🔧 Step 3: Data Preparation

### 3.1 Train/Test Split

```python
# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify preserva distribution de classes
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nDistribución en train:")
print(y_train.value_counts())
```

**Output:**

```
Train set: (120, 4)
Test set: (30, 4)

Distribution en train:
0    40
1    40
2    40
✅ Stratify it worked: distribution balanced preservada
```

### 3.2 Feature scaling

```python
# Train scaler solo con train (prevenir data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply transformation aprendida

# Convertir de nuevo a DataFrame para mantener nombres de columns
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Antes del escalado (train):")
print(X_train.describe().loc[['mean', 'std']])
print("\nDespués del escalado (train):")
print(X_train_scaled.describe().loc[['mean', 'std']])
```

**Output:**

```
After del escalado:
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
mean          -0.000000        -0.000000            -0.000000          0.000000
std            1.008403         1.008403             1.008403          1.008403
✅ Media ≈ 0, std ≈ 1 (escalado correcto)
```

______________________________________________________________________

## 🤖 Step 4: Model Training

### 4.1 Baseline: Logistic Regression

```python
# Model base
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr.predict(X_test_scaled)

# Evaluation
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {acc_lr:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=iris.target_names))
```

**Output:**

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

## 📈 Step 5: Comparison of Models

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

**Output:**

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

### 5.2 Comparison display

```python
# Compare performance
comparison = pd.DataFrame({
    'Model': list(models.keys()),
    'Test Accuracy': [acc_lr, acc_dt, acc_rf],
    'CV Mean': [cv_results[name].mean() for name in models.keys()],
    'CV Std': [cv_results[name].std() for name in models.keys()]
})

print("\n=== RESUMEN DE MODELOS ===")
print(comparison)

# Graphic
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arrange(len(comparison))
width = 0.35

ax.bar(x - width/2, comparison['Test Accuracy'], width, label='Test Accuracy', color='skyblue')
ax.bar(x + width/2, comparison['CV Mean'], width, label='CV Mean', color='lightcoral')

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison de Models')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Model'], rotation=15, ha='right')
ax.legend()
ax.set_ylim([0.9, 1.01])

plt.tight_layout()
plt.show()
```

______________________________________________________________________

## 🎯 Step 6: Detailed analysis of the better Model

```python
# Seleccionar better model (Logistic Regression y Random Forest empatan)
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

# Graphic
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

**Output:**

```
=== FEATURE IMPORTANCE ===
              feature  importance
2  petal length (cm)    0.445302  👈 Feature más important
3   petal width (cm)    0.425186
0  sepal length (cm)    0.089254
1   sepal width (cm)    0.040258
```

**Insight:**

- **Petal** dimensions are much more predictive than sepals
- This confirms EDA observations

______________________________________________________________________

## 🚀 Step 7: Reproducible pipeline with scikit-learn

```python
from sklearn.pipeline import Pipeline

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
])

# Train pipeline (fit + transform automatic)
pipeline.fit(X_train, y_train)

# Predict
y_pred_pipeline = pipeline.predict(X_test)
acc_pipeline = accuracy_score(y_test, y_pred_pipeline)

print(f"=== PIPELINE ===")
print(f"Accuracy: {acc_pipeline:.4f}")

# Advantage: Predict en data nuevos sin olvidar escalar
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Setosa typical
prediction = pipeline.predict(new_sample)
print(f"\nPredicción para muestra nueva: {iris.target_names[prediction[0]]}")
```

______________________________________________________________________

## 💾 Paso 8: Save Model

```python
import joblib

# Save model entrenado
joblib.dump(pipeline, 'iris_classifier_pipeline.pkl')
print("Model guardado: iris_classifier_pipeline.pkl")

# Load model (en production)
loaded_pipeline = joblib.load('iris_classifier_pipeline.pkl')
prediction_loaded = loaded_pipeline.predict(new_sample)
print(f"Prediction con model cargado: {iris.target_names[prediction_loaded[0]]}")
```

______________________________________________________________________

## 📝 Executive summary

### ✅ Results

| Model | Test accuracy | CV Mean | CV Std |
| ------------------- | ------------- | ------- | ------ |
| Logistic Regression | 1.0000 | 0.9667 | 0.0316 |
| Decision Tree | 1.0000 | 0.9583 | 0.0316 |
| Random Forest | 1.0000 | 0.9667 | 0.0316 |

**Selected model:** Random Forest

- perfect accuracy in test set (1.00)
- Robust in cross-validation (0.9667 ± 0.0316)
- Interpretable (feature importance)

### 🎯 Features clave

1. **petal length** (44.5%) - Most important
1. **petal width** (42.5%)
1. **sepal length** (8.9%)
1. **sepal width** (4.0%)

### 🔍 Observations

- Dataset Iris is "easy" (accuracy close to 100%)
- Perfectly separable setosa
- Versicolor/Virginica overlap minimum
- There was no overfitting (similar CV and test performance)

______________________________________________________________________

## 🎓 Lessons learned

### ✅ Good Practices applied

1. **Train/test split stratified** → Preserve class distribution
1. **Scaling after split** → Prevents data leakage
1. **Cross-validation** → Robust validation (do not rely only on test accuracy)
1. **Comparison of multiple Models** → Baseline simple vs complex
1. **Scikit-learn Pipeline** → Reproducibility and preventing Errors
1. **Feature importance** → Interpretability

### 🚫 Common errors avoided

- ❌ Scale before split (data leakage)
- ❌ Optimize Hyperparameters by looking at test set (overfitting)
- ❌ Do not use cross-validation (overestimate performance)
- ❌ Forget stratification (unbalanced split)

### 🔧 Key tools

- `StandardScaler`: Scaling (mean 0, std 1)
- `train_test_split` with `stratify`: Split balanced
- `cross_val_score`: Cross-validation
- `Pipeline`: Workflow reproducible
- `joblib`: Model Persistence

### 💡 Next steps

For more complex real datasets:

1. Handling missing values ​​and outliers
1. More sophisticated feature engineering
1. Hyperparameter tuning (Grid Search, Random Search)
1. Error Analysis (why the Model fails)
1. Probability calibration
1. Threshold optimization (if FP/FN costs differ)
