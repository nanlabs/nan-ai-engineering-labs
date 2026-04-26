# Example 02 — Regression with Overfitting Diagnosis

## Context

You will build a regression model to predict home prices, with emphasis on **detecting and mitigating overfitting**. You will learn to diagnose Problems using Learning curves and adjust Model complexity.

## Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset California Housing
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='Price')

print("=== CALIFORNIA HOUSING DATASET ===")
print(f"Dimensions: {X.shape}")
print(f"Target: Precio mediano de casa (en $100,000)")
print(f"\nFeatures:")
print(X.columns.tolist())
print(f"\nPrimeras rows:")
print(X.head())
```

**Output:**

```
=== CALIFORNIA HOUSING DATASET ===
Dimensions: (20640, 8)
Target: Precio mediano de casa (en $100,000)

Features:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
```

______________________________________________________________________

## 📊 Step 1: EDA and preparation

### 1.1 Initial inspection

```python
print("=== STATISTICS ===")
print(X.describe())
print(f"\nDistribución del target (Price):")
print(y.describe())

# Visualize distribution del target
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(y, bins=50, edgecolor='black', color='skyblue')
plt.xlabel('Price ($100k)')
plt.ylabel('Frequency')
plt.title('Distribution de Precios')

plt.subplot(1, 2, 2)
plt.boxplot(y, vert=False)
plt.xlabel('Price ($100k)')
plt.title('Boxplot de Precios')

plt.tight_layout()
plt.show()
```

**Observation:** Distribution with outliers (some very expensive houses).

### 1.2 Correlations

```python
# Add target al DataFrame para correlaciones
df = X.copy()
df['Price'] = y

# Matrix de correlation
corr = df.corr()['Price'].sort_values(ascending=False)
print("\n=== CORRELACIONES CON PRICE ===")
print(corr)

# Visualize top features
plt.figure(figsize=(10, 6))
corr.drop('Price').plot(kind='barh', color='steelblue')
plt.xlabel('Correlation with Price')
plt.title('Feature Correlations')
plt.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()
```

**Output:**

```
=== CORRELACIONES CON PRICE ===
Price         1.000000
MedInc        0.688075  👈 Correlation más fuerte
AveRooms      0.151948
HouseAge      0.105623
...
```

### 1.3 Train/Test Split

```python
# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

______________________________________________________________________

## 🤖 Paso 2: Baseline - Linear Regression

```python
# Train model base
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = lr.predict(X_train_scaled)
y_test_pred = lr.predict(X_test_scaled)

# Metrics
def evaluate_model(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {dataset_name} ===")
    print(f"MAE:  ${mae:.4f} x 100k = ${mae * 100000:.0f}")
    print(f"RMSE: ${rmse:.4f} x 100k = ${rmse * 100000:.0f}")
    print(f"R²:   {r2:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

train_metrics = evaluate_model(y_train, y_train_pred, "TRAIN - Linear Regression")
test_metrics = evaluate_model(y_test, y_test_pred, "TEST - Linear Regression")
```

**Output:**

```
=== TRAIN - Linear Regression ===
MAE:  $0.5264 x 100k = $52640
RMSE: $0.7272 x 100k = $72720
R²:   0.6021

=== TEST - Linear Regression ===
MAE:  $0.5329 x 100k = $53290
RMSE: $0.7405 x 100k = $74050
R²:   0.5757

✅ Train y Test performance similar → No overfitting
```

______________________________________________________________________

## 📈 Step 3: Detect overfitting with Decision Tree

### 3.1 Unconstrained Decision Tree (expected overfit)

```python
# Tree profundo (sin max_depth)
dt_overfit = DecisionTreeRegressor(random_state=42)
dt_overfit.fit(X_train_scaled, y_train)

y_train_pred_dt = dt_overfit.predict(X_train_scaled)
y_test_pred_dt = dt_overfit.predict(X_test_scaled)

train_metrics_dt = evaluate_model(y_train, y_train_pred_dt, "TRAIN - Decision Tree (sin limit)")
test_metrics_dt = evaluate_model(y_test, y_test_pred_dt, "TEST - Decision Tree (sin limit)")
```

**Output:**

```
=== TRAIN - Decision Tree (sin limit) ===
MAE:  $0.0000 x 100k = $0
RMSE: $0.0000 x 100k = $0
R²:   1.0000  👈 Perfecto en train (memorization)

=== TEST - Decision Tree (sin limit) ===
MAE:  $0.7241 x 100k = $72410
RMSE: $1.0269 x 100k = $102690
R²:   0.2328  👈 Peor que Linear Regression

🚨 OVERFITTING DETECTADO: R² train (1.0) >> R² test (0.23)
```

### 3.2 Learning Curves

```python
def plot_learning_curve(model, X, y, title):
    """
    Graficar curves de learning para diagnosticar overfitting.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )

    # Convertir a RMSE
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)

    # Promedios y desviaciones
    train_mean = train_rmse.mean(axis=1)
    train_std = train_rmse.std(axis=1)
    val_mean = val_rmse.mean(axis=1)
    val_std = val_rmse.std(axis=1)

    # Graphic
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training RMSE', marker='o', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, val_mean, label='Validation RMSE', marker='s', color='red')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')

    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Compare Linear Regression vs Decision Tree
plot_learning_curve(lr, X_train_scaled, y_train, 'Learning Curve - Linear Regression')
plot_learning_curve(dt_overfit, X_train_scaled, y_train, 'Learning Curve - Decision Tree (Overfit)')
```

**Interpretation of curves:**

**Linear Regression:**

- Train and Validation RMSE converge (small gap)
- ✅ No overfitting
- ⚠️ High bias (both curves high plateau)

**Decision Tree:**

- Train RMSE very low (almost 0)
- Validation RMSE alto
- Large gap between train and validation
- 🚨 overfitting claro

______________________________________________________________________

## 🛠️ Step 4: Mitigate overfitting

### 4.1 Decision Tree with max_depth

```python
# Probar different profundidades
depths = [2, 4, 6, 8, 10, 15, 20, None]
results = []

for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, dt.predict(X_train_scaled))
    test_r2 = r2_score(y_test, dt.predict(X_test_scaled))

    results.append({
        'max_depth': depth if depth else 'None',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': train_r2 - test_r2
    })

results_df = pd.DataFrame(results)
print("\n=== IMPACTO DE max_depth ===")
print(results_df)

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(results_df))
ax.plot(x, results_df['train_r2'], marker='o', label='Train R²', color='blue')
ax.plot(x, results_df['test_r2'], marker='s', label='Test R²', color='red')
ax.set_xlabel('max_depth')
ax.set_ylabel('R² Score')
ax.set_title('R² vs max_depth')
ax.set_xticks(x)
ax.set_xticklabels(results_df['max_depth'])
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Output:**

```
=== IMPACTO DE max_depth ===
  max_depth  train_r2  test_r2      gap
0         2    0.4562   0.4489   0.0073  👈 Underfitting
1         4    0.5834   0.5456   0.0378
2         6    0.6694   0.5892   0.0802  👈 Balance optimum
3         8    0.7438   0.5754   0.1684
4        10    0.8126   0.5489   0.2637
5        15    0.9144   0.4782   0.4362
6        20    0.9694   0.3654   0.6040
7      None    1.0000   0.2328   0.7672  👈 Overfitting severo
```

**Conclusion:** `max_depth=6` offers better balance (higher R² test, moderate gap).

### 4.2 Random Forest (ensemble to reduce variance)

```python
# Random Forest con hyperparameters balanceados
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

y_train_pred_rf = rf.predict(X_train_scaled)
y_test_pred_rf = rf.predict(X_test_scaled)

train_metrics_rf = evaluate_model(y_train, y_train_pred_rf, "TRAIN - Random Forest")
test_metrics_rf = evaluate_model(y_test, y_test_pred_rf, "TEST - Random Forest")
```

**Output:**

```
=== TRAIN - Random Forest ===
MAE:  $0.3268 x 100k = $32680
RMSE: $0.4812 x 100k = $48120
R²:   0.8321

=== TEST - Random Forest ===
MAE:  $0.4949 x 100k = $49490
RMSE: $0.6882 x 100k = $68820
R²:   0.6349  👈 Mejor que Linear Regression

✅ Gap razonable: train R² (0.83) - test R² (0.63) = 0.20
✅ Mejor performance en test que Decision Tree individual
```

### 4.3 Regularization: Ridge Regression

```python
from sklearn.linear_model import Ridge

# Ridge con different values de alpha
alphas = [0.01, 0.1, 1, 10, 100]
ridge_results = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, ridge.predict(X_train_scaled))
    test_r2 = r2_score(y_test, ridge.predict(X_test_scaled))

    ridge_results.append({
        'alpha': alpha,
        'train_r2': train_r2,
        'test_r2': test_r2
    })

ridge_df = pd.DataFrame(ridge_results)
print("\n=== RIDGE REGRESSION (Regularization L2) ===")
print(ridge_df)
```

**Output:**

```
=== RIDGE REGRESSION (Regularization L2) ===
   alpha  train_r2  test_r2
0   0.01    0.6021   0.5757
1   0.10    0.6021   0.5757
2   1.00    0.6020   0.5758  👈 Ligera improvement en test
3  10.00    0.6000   0.5767  👈 Mejor test R²
4 100.00    0.5670   0.5609
```

**Insight:** Ridge with alpha=10 improved slightly test R² by penalizing large coefficients.

______________________________________________________________________

## 📊 Step 5: Final Comparison of Models

```python
# Summary de todos los models
comparison = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Decision Tree (no limit)',
        'Decision Tree (depth=6)',
        'Random Forest',
        'Ridge (alpha=10)'
    ],
    'Train R²': [
        train_metrics['R2'],
        train_metrics_dt['R2'],
        0.6694,
        train_metrics_rf['R2'],
        0.6000
    ],
    'Test R²': [
        test_metrics['R2'],
        test_metrics_dt['R2'],
        0.5892,
        test_metrics_rf['R2'],
        0.5767
    ],
    'Test RMSE': [
        test_metrics['RMSE'],
        test_metrics_dt['RMSE'],
        0.7115,
        test_metrics_rf['RMSE'],
        0.7223
    ]
})

comparison['Gap (Train-Test)'] = comparison['Train R²'] - comparison['Test R²']
comparison = comparison.sort_values('Test R²', ascending=False)

print("\n=== COMPARISON FINAL ===")
print(comparison)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
x = range(len(comparison))
axes[0].bar([i - 0.2 for i in x], comparison['Train R²'], width=0.4, label='Train R²', color='skyblue')
axes[0].bar([i + 0.2 for i in x], comparison['Test R²'], width=0.4, label='Test R²', color='coral')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison['Model'], rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Gap (overfitting indicator)
axes[1].barh(comparison['Model'], comparison['Gap (Train-Test)'], color='indianred')
axes[1].set_xlabel('Gap (Train R² - Test R²)')
axes[1].set_title('Overfitting Gap (menor es better)')
axes[1].axvline(x=0.1, color='green', linestyle='--', label='Threshold (0.1)')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
```

**Ranking:**

1. **Random Forest** - Best Test R² (0.6349), acceptable gap
1. **Decision Tree (depth=6)** - Decent balance, more interpretable
1. **Ridge (alpha=10)** - Slight improvement about Linear Regression
1. **Linear Regression** - Solid Baseline
1. **Decision Tree (no limit)** - Severe overfitting

______________________________________________________________________

## 🎓 Lessons learned

### ✅ How to detect overfitting

1. **Compare train vs test metrics:**

   - Train R² >> Test R² → overfitting
   - Train R² ≈ Test R² → Good balance

1. **Learning Curves:**

   - Large gap between train and validation → overfitting
   - Both atlas curves → underfitting

1. **Cross-validation:**

   - High variance in CV scores → Instability

### 🛠️ Techniques to mitigate overfitting

1. **Regularization:**

   - Ridge (L2): Penalizes large coefficients
- Lasso (L1): Automatic feature selection
   - Elastic Net: Combine L1 and L2

1. **Limit Model complexity:**

- `max_depth`, `min_samples_split` in trees
   - `n_estimators`, `max_features` in Random Forest

1. **More Data:**

- Increase Training size (if possible)

1. **Ensembles:**

- Random Forest reduces variance of individual trees
   - Bagging, Boosting

### 📐 Regression metrics

- **MAE:** average error in original units (interpretable)
- **RMSE:** Penalizes large errors more strongly
- **R²:** % of variance explained (0-1, higher the better)

### 💡 Rules Practices

- Gap train-test < 0.1 → Good balance
- Gap train-test > 0.3 → overfitting probable
- Test performance improvement with Regularization → Model was overfit
- Test performance worsens with Regularization → Model was underfit
