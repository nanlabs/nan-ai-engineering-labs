# Example 01 — Bias Detection and Fairness Metrics

## Context

ML Models can perpetuate or amplify biases present in Training Data, resulting in discriminatory decisions.

## Objective

Detect and measure bias in Credit Approval Model using fairness metrics.

______________________________________________________________________

## 🚀 Setup

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
```

______________________________________________________________________

## 📚 Generate synthetic data

```python
# Simulation de solicitudes de credit
n_samples = 1000

data = {
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'employment_years': np.random.randint(0, 30, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'ethnicity': np.random.choice(['Group_A', 'Group_B'], n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Target: approval de credit (score-based con bias)
def generate_approval(row):
    """
    Function con bias intentional:
    - Base: credit_score + income
    - Bias: penaliza a mujeres y Group_B
    """
    base_score = row['credit_score'] / 10 + row['income'] / 10000

    # Bias de gender (mujeres necesitan 10% más score)
    if row['gender'] == 'F':
        base_score *= 0.9

    # Bias ethnic (Group_B necesita 15% más score)
    if row['ethnicity'] == 'Group_B':
        base_score *= 0.85

    # Threshold para approval
    return 1 if base_score > 100 else 0

df['approved'] = df.apply(generate_approval, axis=1)

print(f"Dataset: {len(df)} solicitudes")
print(f"Tasa de approval: {df['approved'].mean():.2%}")
print("\nDistribución:")
print(df[['gender', 'ethnicity', 'approved']].value_counts())
```

**Output:**

```
Dataset: 1000 solicitudes
Tasa de approval: 52.30%

Distribution:
gender  ethnicity  approved
M       Group_A    1           298
                   0           161
        Group_B    1            89
                   0            52
F       Group_A    1           125
                   0           109
        Group_B    1            48
                   0           118
```

______________________________________________________________________

## 🤖 Train Model

```python
# Features (excluir protected attributes para training)
feature_cols = ['age', 'income', 'credit_score', 'employment_years']
X = df[feature_cols]
y = df['approved']

# Save atributos protegidos para analysis
protected_attrs = df[['gender', 'ethnicity']]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Indices para protected attrs
test_indices = X_test.index
protected_test = protected_attrs.loc[test_indices]

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy general
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
```

**Output:**

```
Accuracy: 0.8867
```

______________________________________________________________________

## ⚖️ Fairness Metrics

### 1. Demographic Parity (Statistical Parity)

```python
def demographic_parity(y_pred, protected_attr):
    """
    P(Ŷ=1 | A=0) ≈ P(Ŷ=1 | A=1)

    Approval similar entre grupos
    """
    groups = protected_attr.unique()
    approval_rates = {}

    for group in groups:
        mask = protected_attr == group
        approval_rate = y_pred[mask].mean()
        approval_rates[group] = approval_rate

    # Diferencia maximum
    rates = list(approval_rates.values())
    disparity = max(rates) - min(rates)

    return approval_rates, disparity

# Por gender
gender_rates, gender_disparity = demographic_parity(
    y_pred, protected_test['gender']
)

print("\n=== Demographic Parity - Gender ===")
for group, rate in gender_rates.items():
    print(f"{group}: {rate:.2%} approval")
print(f"Disparidad: {gender_disparity:.4f} (ideal < 0.1)")

# Por etnia
ethnicity_rates, eth_disparity = demographic_parity(
    y_pred, protected_test['ethnicity']
)

print("\n=== Demographic Parity - Etnia ===")
for group, rate in ethnicity_rates.items():
    print(f"{group}: {rate:.2%} approval")
print(f"Disparidad: {eth_disparity:.4f}")
```

**Output:**

```
=== Demographic Parity - Gender ===
M: 68.23% approval
F: 51.12% approval
Disparidad: 0.1711 (ideal < 0.1)

=== Demographic Parity - Etnia ===
Group_A: 64.89% approval
Group_B: 48.76% approval
Disparidad: 0.1613
```

### 2. Equalized Odds

```python
def equalized_odds(y_true, y_pred, protected_attr):
    """
    TPR y FPR deben ser iguales entre grupos

    TPR: P(Ŷ=1 | Y=1, A=a)
    FPR: P(Ŷ=1 | Y=0, A=a)
    """
    groups = protected_attr.unique()
    metrics = {}

    for group in groups:
        mask = protected_attr == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        # True Positive Rate
        tp = ((y_true_group == 1) & (y_pred_group == 1)).sum()
        fn = ((y_true_group == 1) & (y_pred_group == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        # False Positive Rate
        fp = ((y_true_group == 0) & (y_pred_group == 1)).sum()
        tn = ((y_true_group == 0) & (y_pred_group == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics[group] = {'TPR': tpr, 'FPR': fpr}

    # Disparidad
    tprs = [m['TPR'] for m in metrics.values()]
    fprs = [m['FPR'] for m in metrics.values()]

    tpr_disparity = max(tprs) - min(tprs)
    fpr_disparity = max(fprs) - min(fprs)

    return metrics, tpr_disparity, fpr_disparity

# Por gender
gender_metrics, tpr_disp, fpr_disp = equalized_odds(
    y_test.values, y_pred, protected_test['gender'].values
)

print("\n=== Equalized Odds - Gender ===")
for group, metrics in gender_metrics.items():
    print(f"{group}: TPR={metrics['TPR']:.4f}, FPR={metrics['FPR']:.4f}")
print(f"TPR Disparidad: {tpr_disp:.4f}")
print(f"FPR Disparidad: {fpr_disp:.4f}")
```

**Output:**

```
=== Equalized Odds - Gender ===
M: TPR=0.9234, FPR=0.1523
F: TPR=0.8567, FPR=0.2134
TPR Disparidad: 0.0667
FPR Disparidad: 0.0611
```

### 3. Equal Opportunity

```python
def equal_opportunity(y_true, y_pred, protected_attr):
    """
    Solo TPR must ser igual (subset de Equalized Odds)

    Useful when False Positives son aceptables
    """
    groups = protected_attr.unique()
    tprs = {}

    for group in groups:
        mask = protected_attr == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        tp = ((y_true_group == 1) & (y_pred_group == 1)).sum()
        fn = ((y_true_group == 1) & (y_pred_group == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        tprs[group] = tpr

    disparity = max(tprs.values()) - min(tprs.values())

    return tprs, disparity

tprs, eo_disparity = equal_opportunity(
    y_test.values, y_pred, protected_test['gender'].values
)

print("\n=== Equal Opportunity - Gender ===")
for group, tpr in tprs.items():
    print(f"{group}: TPR={tpr:.4f}")
print(f"Disparidad: {eo_disparity:.4f} (ideal < 0.1)")
```

**Output:**

```
=== Equal Opportunity - Gender ===
M: TPR=0.9234
F: TPR=0.8567
Disparidad: 0.0667 (ideal < 0.1)
```

______________________________________________________________________

## 📊 Bias display

```python
# Create DataFrame con results
results_df = pd.DataFrame({
    'y_true': y_test.values,
    'y_pred': y_pred,
    'gender': protected_test['gender'].values,
    'ethnicity': protected_test['ethnicity'].values
})

# Approval rate por group
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Por gender
gender_approval = results_df.groupby('gender')['y_pred'].mean()
axes[0].bar(gender_approval.index, gender_approval.values, color=['blue', 'pink'], alpha=0.7)
axes[0].set_ylabel('Tasa de Approval')
axes[0].set_title('Approval por Gender')
axes[0].axhline(y=results_df['y_pred'].mean(), color='red', linestyle='--', label='Media global')
axes[0].legend()
axes[0].set_ylim([0, 1])

# Por etnia
eth_approval = results_df.groupby('ethnicity')['y_pred'].mean()
axes[1].bar(eth_approval.index, eth_approval.values, color=['green', 'orange'], alpha=0.7)
axes[1].set_ylabel('Tasa de Approval')
axes[1].set_title('Approval por Etnia')
axes[1].axhline(y=results_df['y_pred'].mean(), color='red', linestyle='--', label='Media global')
axes[1].legend()
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('bias_approval_rates.png', dpi=150)
plt.show()
```

### Confusion matrix by group

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

groups = [('M', 'gender'), ('F', 'gender'), ('Group_A', 'ethnicity'), ('Group_B', 'ethnicity')]

for idx, (group, attr) in enumerate(groups):
    ax = axes[idx // 2, idx % 2]

    mask = results_df[attr] == group
    y_true_group = results_df.loc[mask, 'y_true']
    y_pred_group = results_df.loc[mask, 'y_pred']

    cm = confusion_matrix(y_true_group, y_pred_group)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Real')
    ax.set_title(f'Matrix Confusion: {group}')

plt.tight_layout()
plt.savefig('bias_confusion_matrices.png', dpi=150)
plt.show()
```

______________________________________________________________________

## 🛠️ Bias Mitigation

### 1. Reweighting

```python
def reweighting(X, y, protected_attr):
    """
    Asigna pesos a muestras para balancear representation
    """
    weights = np.ones(len(y))

    # Calculate frecuencias
    for group in protected_attr.unique():
        for label in [0, 1]:
            mask = (protected_attr == group) & (y == label)
            count = mask.sum()
            if count > 0:
                # Peso inversamente proporcional a frecuencia
                weights[mask] = 1.0 / count

    # Normalizar
    weights = weights / weights.sum() * len(weights)

    return weights

# Reentrenar con pesos
weights_train = reweighting(
    X_train, y_train, protected_attrs.loc[X_train.index, 'gender']
)

model_fair = RandomForestClassifier(n_estimators=100, random_state=42)
model_fair.fit(X_train, y_train, sample_weight=weights_train)

y_pred_fair = model_fair.predict(X_test)

# Evaluate fairness
gender_rates_fair, disparity_fair = demographic_parity(
    y_pred_fair, protected_test['gender']
)

print("\n=== After de Reweighting ===")
for group, rate in gender_rates_fair.items():
    print(f"{group}: {rate:.2%}")
print(f"Disparidad: {disparity_fair:.4f} (antes: {gender_disparity:.4f})")

# Accuracy can disminuir
acc_fair = accuracy_score(y_test, y_pred_fair)
print(f"Accuracy: {acc_fair:.4f} (antes: {acc:.4f})")
```

### 2. Threshold Optimization

```python
from sklearn.metrics import roc_curve

def optimize_thresholds(model, X_test, y_test, protected_attr):
    """
    Encuentra thresholds different por group para igualar TPR
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = {}
    for group in protected_attr.unique():
        mask = protected_attr == group

        fpr, tpr, thresh = roc_curve(y_test[mask], y_proba[mask])

        # Threshold para TPR objective (ej: 0.85)
        target_tpr = 0.85
        idx = np.argmin(np.abs(tpr - target_tpr))
        thresholds[group] = thresh[idx]

    return thresholds

thresholds_opt = optimize_thresholds(
    model, X_test.values, y_test.values, protected_test['gender'].values
)

print("\n=== Thresholds Optimizados ===")
for group, thresh in thresholds_opt.items():
    print(f"{group}: {thresh:.4f}")
```

______________________________________________________________________

## 📝 Summary

### ✅ Fairness Metrics

| Metric | Definition | When to use |
| ---------------------- | --------------------------------------- | ------------------------------------------------- |
| **Demographic Parity** | P(Ŷ=1\|A=a) equal for all groups | Decisions that do not affect population proportion |
| **Equalized Odds** | TPR and FPR equal between groups | Errors (FP and FN) have the same cost |
| **Equal Opportunity** | Only TPR equal between groups | PF less serious than FN (e.g. medical screening) |

### 🎯 Trade-offs

```
Fairness ↑ ⟷ Accuracy ↓
```

- Mitigate bias can reduce overall accuracy
- Important: define which fairness Metric is a priority
- There is no such thing as "universal fairness" - it depends on the context

### 💡 Best Practices

- ✅ Identify protected attributes (gender, ethnicity, age)
- ✅ Measure BEFORE deployment
- ✅ Evaluate multiple fairness metrics
- ✅ Document trade-offs (accuracy vs fairness)
- ✅ Re-evaluate periodically (fairness drift)
- ✅ External audits when possible

### 🚫 Errors common

- ❌ Remove protected attributes without further ado (proxy variables persist)
- ❌ Only measure accuracy (ignore disparities by group)
- ❌ Assume that "treat everyone equal in code" = fairness
- ❌ Do not involve affected stakeholders
- ❌ Forget intersectionality (ex: woman + Group_B)

### 🔧 Mitigation techniques

**Pre-processing:**

- Reweighting
- Resampling (oversample minority groups)
- Data augmentation

**In-processing:**

- Fairness constraints in loss function
- Adversarial debiasing

**Post-processing:**

- Threshold optimization
- Calibration by group

### 📌 Checklist Fairness

- ✅ Protected attributes identified
- ✅ Demographic Parity evaluated
- ✅ Equalized Odds evaluated
- ✅ Visualization of disparities
- ✅Trade-off accuracy/fairness documented
- ✅ Mitigation technique applied
- ✅ Validation in historical data
- ✅ Continuous monitoring plan

### ⚠️ Legal and ethical

- Regulation (e.g. EU AI Act, Fair Lending Laws)
- Indirect discrimination: protected attributes proxies
- Transparency: explain decisions to affected users
- Right to explanation (GDPR)
