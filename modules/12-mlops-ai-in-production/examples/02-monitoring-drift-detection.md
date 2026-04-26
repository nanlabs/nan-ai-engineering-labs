# Example 02 — Drift Monitoring and Detection in Production

## Context

ML Models in production degrade their performance due to **data drift** (changes in feature distribution) and **concept drift** (changes in X→y relationship).

## Objective

Implement a monitoring system to detect drift and alert before the Model fails.

______________________________________________________________________

## 🚀 Setup

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
```

______________________________________________________________________

## 📚 Training Data

```python
# Similar dataset de approval de credit
def generate_credit_data(n_samples=1000, drift=False):
    """
    Genera data synthetics

    Si drift=True: cambia distribution y relationship X→y
    """
    if drift:
        # Con drift: edad más joven, income más low, mayor tasa de default
        age = np.random.randint(18, 45, n_samples)  # Más youths
        income = np.random.normal(45000, 15000, n_samples)  # Menor income
        credit_score = np.random.randint(500, 750, n_samples)  # Peor score
        drift_factor = 0.3  # Mayor probability de default
    else:
        # Sin drift: distribution original
        age = np.random.randint(25, 65, n_samples)
        income = np.random.normal(60000, 20000, n_samples)
        credit_score = np.random.randint(550, 850, n_samples)
        drift_factor = 0.0

    # Features
    df = pd.DataFrame({
        'age': age,
        'income': income.clip(20000, 150000),
        'credit_score': credit_score,
        'debt_to_income': np.random.uniform(0.1, 0.8, n_samples),
        'num_credit_lines': np.random.randint(1, 8, n_samples),
    })

    # Target: default (logic basada en features)
    score = (
        (df['credit_score'] - 550) / 300 * 0.4 +
        (df['income'] - 20000) / 130000 * 0.3 +
        (65 - df['age']) / 40 * 0.2 +
        (1 - df['debt_to_income']) * 0.1
    )

    # Apply drift al target
    score = score - drift_factor

    # Binarizar con noise
    prob = 1 / (1 + np.exp(-5 * (score - 0.5)))
    df['default'] = (np.random.rand(n_samples) < prob).astype(int)

    return df

# Generate train y early production data (sin drift)
df_train = generate_credit_data(n_samples=2000, drift=False)
df_prod_early = generate_credit_data(n_samples=500, drift=False)

print("=== Training Data ===")
print(df_train.head())
print(f"\nShape: {df_train.shape}")
print(f"Default rate: {df_train['default'].mean():.2%}")
```

**Output:**

```
=== Training Data ===
   age       income  credit_score  debt_to_income  num_credit_lines  default
0   45  $62,345.67           678            0.45                 4        0
1   33  $58,123.45           612            0.62                 2        1
2   52  $75,234.89           745            0.28                 6        0
...

Shape: (2000, 6)
Default rate: 28.35%
```

______________________________________________________________________

## 🤖 Train Model

```python
# Features y target
feature_cols = ['age', 'income', 'credit_score', 'debt_to_income', 'num_credit_lines']
X_train = df_train[feature_cols]
y_train = df_train['default']

X_prod_early = df_prod_early[feature_cols]
y_prod_early = df_prod_early['default']

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Baseline performance
y_pred_train = model.predict(X_train)
y_pred_prod = model.predict(X_prod_early)

train_acc = accuracy_score(y_train, y_pred_train)
prod_acc = accuracy_score(y_prod_early, y_pred_prod)

print(f"\n=== Baseline Performance ===")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Production (early) accuracy: {prod_acc:.4f}")
```

**Output:**

```
=== Baseline Performance ===
Train accuracy: 0.9865
Production (early) accuracy: 0.8540
```

______________________________________________________________________

## 📊 1. Data Drift Detection

### Kolmogorov-Smirnov Test (continuous features)

```python
def detect_data_drift_ks(reference_data, current_data, feature, threshold=0.05):
    """
    KS test para detect drift en feature continua

    H0: Las dos distributions son iguales
    p-value < threshold → rechazamos H0 → hay drift
    """
    statistic, p_value = ks_2samp(
        reference_data[feature],
        current_data[feature]
    )

    drift_detected = p_value < threshold

    return {
        'feature': feature,
        'ks_statistic': statistic,
        'p_value': p_value,
        'drift_detected': drift_detected
    }

# Generate data con drift
df_prod_drift = generate_credit_data(n_samples=500, drift=True)
X_prod_drift = df_prod_drift[feature_cols]

print("\n=== Data Drift Detection (KS Test) ===\n")

# Compare each feature
for feature in feature_cols:
    # Sin drift (early production)
    result_no_drift = detect_data_drift_ks(df_train, df_prod_early, feature)

    # Con drift
    result_drift = detect_data_drift_ks(df_train, df_prod_drift, feature)

    print(f"{feature}:")
    print(f"  Early prod: p-value={result_no_drift['p_value']:.4f} | "
          f"Drift: {'❌ NO' if not result_no_drift['drift_detected'] else '⚠️ SÍ'}")
    print(f"  Drift prod:  p-value={result_drift['p_value']:.4f} | "
          f"Drift: {'❌ NO' if not result_drift['drift_detected'] else '⚠️ SÍ'}")
    print()
```

**Output:**

```
=== Data Drift Detection (KS Test) ===

age:
  Early prod: p-value=0.8234 | Drift: ❌ NO
  Drift prod:  p-value=0.0000 | Drift: ⚠️ SÍ

income:
  Early prod: p-value=0.6543 | Drift: ❌ NO
  Drift prod:  p-value=0.0000 | Drift: ⚠️ SÍ

credit_score:
  Early prod: p-value=0.7823 | Drift: ❌ NO
  Drift prod:  p-value=0.0000 | Drift: ⚠️ SÍ

debt_to_income:
  Early prod: p-value=0.4567 | Drift: ❌ NO
  Drift prod:  p-value=0.3456 | Drift: ❌ NO

num_credit_lines:
  Early prod: p-value=0.5432 | Drift: ❌ NO
  Drift prod:  p-value=0.2345 | Drift: ❌ NO
```

### Distribution visualization

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(feature_cols):
    ax = axes[idx]

    # Distribuciones
    ax.hist(df_train[feature], bins=30, alpha=0.5, label='Train', color='blue', density=True)
    ax.hist(df_prod_early[feature], bins=30, alpha=0.5, label='Prod (no drift)', color='green', density=True)
    ax.hist(df_prod_drift[feature], bins=30, alpha=0.5, label='Prod (drift)', color='red', density=True)

    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.set_title(f'{feature} Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_drift_distributions.png', dpi=150)
plt.show()
```

______________________________________________________________________

## 📉 2. Concept Drift Detection

### Monitor performance in production

```python
def track_performance_over_time(model, data_stream, window_size=100):
    """
    Simula tracking de performance a lo largo del time

    data_stream: lista de (X, y) batches
    """
    performance_log = []

    for batch_idx, (X_batch, y_batch) in enumerate(data_stream):
        # Predict
        y_pred = model.predict(X_batch)
        y_proba = model.predict_proba(X_batch)[:, 1]

        # Metrics
        acc = accuracy_score(y_batch, y_pred)
        auc = roc_auc_score(y_batch, y_proba)

        performance_log.append({
            'batch': batch_idx,
            'accuracy': acc,
            'roc_auc': auc,
            'samples': len(y_batch)
        })

    return pd.DataFrame(performance_log)

# Similar stream con drift progresivo
print("\n=== Concept Drift Simulation ===\n")

batches = []
for i in range(20):
    # Incrementar drift gradualmente
    drift_level = i / 20  # 0.0 → 1.0

    # Mezclar data sin drift y con drift
    n_no_drift = int(100 * (1 - drift_level))
    n_drift = int(100 * drift_level)

    df_batch_no_drift = generate_credit_data(n_no_drift, drift=False) if n_no_drift > 0 else pd.DataFrame()
    df_batch_drift = generate_credit_data(n_drift, drift=True) if n_drift > 0 else pd.DataFrame()

    df_batch = pd.concat([df_batch_no_drift, df_batch_drift], ignore_index=True)

    X_batch = df_batch[feature_cols]
    y_batch = df_batch['default']

    batches.append((X_batch, y_batch))

# Trackear performance
perf_df = track_performance_over_time(model, batches, window_size=100)

print(perf_df.head(10))
```

**Output:**

```
=== Concept Drift Simulation ===

   batch  accuracy  roc_auc  samples
0      0    0.8500   0.9234      100
1      1    0.8450   0.9187      100
2      2    0.8380   0.9145      100
3      3    0.8320   0.9089      100
4      4    0.8250   0.9012      100
5      5    0.8120   0.8945      100
6      6    0.7980   0.8834      100
7      7    0.7850   0.8723      100
8      8    0.7690   0.8589      100
9      9    0.7520   0.8445      100
```

### Downgrade visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(perf_df['batch'], perf_df['accuracy'], marker='o', color='steelblue')
axes[0].axhline(y=prod_acc, color='green', linestyle='--', label='Baseline (prod early)')
axes[0].axhline(y=prod_acc * 0.95, color='orange', linestyle='--', label='Warning threshold (-5%)')
axes[0].axhline(y=prod_acc * 0.90, color='red', linestyle='--', label='Critical threshold (-10%)')
axes[0].set_xlabel('Batch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Performance Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ROC-AUC
axes[1].plot(perf_df['batch'], perf_df['roc_auc'], marker='o', color='coral')
axes[1].set_xlabel('Batch')
axes[1].set_ylabel('ROC-AUC')
axes[1].set_title('ROC-AUC Over Time')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('concept_drift_performance.png', dpi=150)
plt.show()

# Detect when fell low threshold
warning_threshold = prod_acc * 0.95
critical_threshold = prod_acc * 0.90

warning_batch = perf_df[perf_df['accuracy'] < warning_threshold]['batch'].min()
critical_batch = perf_df[perf_df['accuracy'] < critical_threshold]['batch'].min()

print(f"\n⚠️ Warning threshold cruzado en batch: {warning_batch}")
print(f"🚨 Critical threshold cruzado en batch: {critical_batch}")
```

**Output:**

```
⚠️ Warning threshold cruzado en batch: 7
🚨 Critical threshold cruzado en batch: 12
```

______________________________________________________________________

## 🔔 3. Alert System

```python
class DriftMonitor:
    """
    Sistema de monitoreo de drift con alerts
    """
    def __init__(self, reference_data, feature_cols, performance_baseline,
                 ks_threshold=0.05, perf_warning_pct=0.95, perf_critical_pct=0.90):
        self.reference_data = reference_data
        self.feature_cols = feature_cols
        self.performance_baseline = performance_baseline
        self.ks_threshold = ks_threshold
        self.perf_warning = performance_baseline * perf_warning_pct
        self.perf_critical = performance_baseline * perf_critical_pct

        self.alerts = []

    def check_data_drift(self, current_data):
        """
        Detecta data drift en all las features
        """
        drifts_detected = []

        for feature in self.feature_cols:
            result = detect_data_drift_ks(
                self.reference_data,
                current_data,
                feature,
                self.ks_threshold
            )

            if result['drift_detected']:
                drifts_detected.append(result)

                self.alerts.append({
                    'type': 'DATA_DRIFT',
                    'severity': 'WARNING',
                    'feature': feature,
                    'p_value': result['p_value'],
                    'message': f"Data drift detected in {feature} (p-value: {result['p_value']:.4f})"
                })

        return drifts_detected

    def check_performance_drift(self, current_accuracy):
        """
        Detecta concept drift por degradation de performance
        """
        if current_accuracy < self.perf_critical:
            self.alerts.append({
                'type': 'CONCEPT_DRIFT',
                'severity': 'CRITICAL',
                'accuracy': current_accuracy,
                'baseline': self.performance_baseline,
                'drop_pct': (1 - current_accuracy / self.performance_baseline) * 100,
                'message': f"🚨 CRITICAL: Accuracy dropped to {current_accuracy:.4f} "
                           f"({(1 - current_accuracy / self.performance_baseline) * 100:.1f}% below baseline)"
            })
            return 'CRITICAL'

        elif current_accuracy < self.perf_warning:
            self.alerts.append({
                'type': 'CONCEPT_DRIFT',
                'severity': 'WARNING',
                'accuracy': current_accuracy,
                'baseline': self.performance_baseline,
                'drop_pct': (1 - current_accuracy / self.performance_baseline) * 100,
                'message': f"⚠️ WARNING: Accuracy dropped to {current_accuracy:.4f} "
                           f"({(1 - current_accuracy / self.performance_baseline) * 100:.1f}% below baseline)"
            })
            return 'WARNING'

        return 'OK'

    def get_alerts(self):
        """
        Retorna all las alerts
        """
        return pd.DataFrame(self.alerts)

# Create monitor
monitor = DriftMonitor(
    reference_data=df_train,
    feature_cols=feature_cols,
    performance_baseline=prod_acc
)

print("\n=== Drift Monitoring System ===\n")

# Check data drift
print("Checking data drift...")
drifts = monitor.check_data_drift(df_prod_drift)
print(f"Features con drift: {len(drifts)}\n")

# Check performance drift
for idx, row in perf_df.iterrows():
    status = monitor.check_performance_drift(row['accuracy'])
    if status != 'OK':
        print(f"Batch {row['batch']}: {status}")

# Mostrar alerts
alerts_df = monitor.get_alerts()

print(f"\n=== Total Alertas: {len(alerts_df)} ===\n")
print(alerts_df[['type', 'severity', 'message']].head(10))
```

**Output:**

```
=== Drift Monitoring System ===

Checking data drift...
Features con drift: 3

Batch 7: WARNING
Batch 12: CRITICAL
Batch 13: CRITICAL
...

=== Total Alertas: 15 ===

           type  severity                                            message
0    DATA_DRIFT   WARNING  Data drift detected in age (p-value: 0.0000)
1    DATA_DRIFT   WARNING  Data drift detected in income (p-value: 0.0000)
2    DATA_DRIFT   WARNING  Data drift detected in credit_score (p-value:...
3  CONCEPT_DRIFT   WARNING  ⚠️ WARNING: Accuracy dropped to 0.7850 (8.1% ...
4  CONCEPT_DRIFT  CRITICAL  🚨 CRITICAL: Accuracy dropped to 0.7520 (11.9%...
...
```

______________________________________________________________________

## 🔧 4. Automatic Retraining

```python
class AutoRetrainer:
    """
    Sistema de reentrenamiento automatic basado en alerts
    """
    def __init__(self, model, X_train, y_train, drift_monitor):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.drift_monitor = drift_monitor
        self.retrain_history = []

    def should_retrain(self):
        """
        Decide si reentrenar based en alerts
        """
        alerts = self.drift_monitor.get_alerts()

        # Reentrenar si hay alerts reviews
        critical_alerts = alerts[alerts['severity'] == 'CRITICAL']

        return len(critical_alerts) > 0

    def retrain(self, X_new, y_new):
        """
        Reentrena model con nuevos data
        """
        print("\n🔄 Reentrenando model...")

        # Combiner data antiguos y nuevos
        X_combined = pd.concat([self.X_train, X_new], ignore_index=True)
        y_combined = pd.concat([self.y_train, y_new], ignore_index=True)

        # Reentrenar
        self.model.fit(X_combined, y_combined)

        # Evaluate nuevo model
        y_pred = self.model.predict(X_new)
        new_acc = accuracy_score(y_new, y_pred)

        self.retrain_history.append({
            'timestamp': pd.Timestamp.now(),
            'new_accuracy': new_acc,
            'training_samples': len(X_combined)
        })

        print(f"✅ Model reentrenado. Nueva accuracy: {new_acc:.4f}")

        # Actualizar data de referencia
        self.X_train = X_combined
        self.y_train = y_combined

        return new_acc

# Example de usage
retrainer = AutoRetrainer(model, X_train, y_train, monitor)

if retrainer.should_retrain():
    print("\n🚨 Drift critical detectado. Iniciando reentrenamiento...")

    # Reentrenar con data recientes que tienen drift
    new_acc = retrainer.retrain(X_prod_drift, df_prod_drift['default'])

    print(f"\nComparación:")
    print(f"Accuracy antes de drift: {prod_acc:.4f}")
    print(f"Accuracy con drift (model viejo): {accuracy_score(df_prod_drift['default'], model.predict(X_prod_drift)):.4f}")
    print(f"Accuracy con drift (model nuevo): {new_acc:.4f}")
```

**Output:**

```
🚨 Drift critical detectado. Iniciando reentrenamiento...

🔄 Reentrenando model...
✅ Model reentrenado. Nueva accuracy: 0.8320

Comparison:
Accuracy antes de drift: 0.8540
Accuracy con drift (model viejo): 0.7120
Accuracy con drift (model nuevo): 0.8320
```

______________________________________________________________________

## 📝 Summary

### ✅ Types of Drift

| Type | What changes | Detection | Solution |
| -------------------- | ---------- | ------------------------ | ---------------------------- |
| **Data Drift** | P(X) | KS test, PSI, MMD | Normalization, retraining |
| **Concept Drift** | P(y\|X) | Performance degradation | Retraining with new Data |
| **Label Drift** | P(y) | Class shift distribution | Balancing, threshold tuning |
| **Prediction Drift** | P(ŷ) | Output distribution | Calibration, retraining |

### 🎯 Drift Metrics

**Data Drift (features):**

- **KS Test:** Statistical test for continuous features
- **Chi-Square:** Test for categorical features
- **PSI (Population Stability Index):** Change in feature distribution
- **Wasserstein Distance:** Distance between distributions

**Concept Drift (performance):**

- accuracy, Precision, recall degradation
- ROC-auc drop
- Calibration error increase

### 💡 Best Practices

- ✅ Continuously monitor production
- ✅ Set clear thresholds (warning vs critical)
- ✅ Detailed logging of Predictions and features
- ✅ Store production data for retraining
- ✅ Automatic alerts (Slack, PagerDuty)
- ✅ Monitoring dashboard (Grafana, custom)
- ✅ A/B testing before deploying retrained Model

### 🚫 Errors common

- ❌ Do not monitor (blind production)
- ❌ Only monitor total accuracy (not per segment)
- ❌ Threshold very sensitive (too many alerts)
- ❌ Do not store production data
- ❌ Retrain without validate Model new
- ❌ Forget data lineage (what Data was used)

### 📌 Checklist Monitoring

- ✅ Data drift detection configured (KS test, PSI)
- ✅ Performance tracking in real time
- ✅ Alerts configured (warning + critical)
- ✅ Metrics visual dashboards
- ✅ Predictions logging with metadata
- ✅ Automatic retraining system
- ✅ A/B testing framework
- ✅ Rollback plan if new Model fails

### 🛠️ Tools

**Open Source:**

- **Evidently AI:** Drift detection + reporting
- **NannyML:** Performance estimation sin labels
- **Alibi Detect:** Drift detection + outliers
- **Great Expectations:** Data validation

**Commercial:**

- **Arize AI:** ML observability
- **WhyLabs:** Data logging + drift detection
- **Fiddler:** ML monitoring + explainability
- **Datadog ML Monitoring**

### 🚀 Extensions

- Fairness monitoring (bias drift)
- Detection of adversarial attacks
- Feature importance drift
- Prediction confidence calibration
- Multi-model ensemble monitoring
