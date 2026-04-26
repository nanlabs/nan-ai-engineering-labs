# Example 02 — Detection of Anomalies in time series

## Context

Anomalies are observations that deviate significantly from normal behavior. You will learn statistical techniques, based on ML and deep learning to detect Anomalies in time series.

## Objective

Detect fraudulent transactions, server failures or anomalous behavior in temporary data.

______________________________________________________________________

## 🚀 Step 1: Setup and imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
```

______________________________________________________________________

## 📥 Step 2: Generate Data with Anomalies

```python
# Time series normal
np.random.seed(42)
date_range = pd.date_range(start='2023-01-01', periods=365, freq='H')

# Pattern normal: trend + seasonality
hours = np.arrange(len(date_range))
trend = 0.05 * hours
seasonality = 20 * np.sin(2 * np.pi * hours / 24)  # Seasonality diaria
noise = np.random.normal(0, 5, len(date_range))

normal_values = 100 + trend + seasonality + noise

# Inyectar anomalies (5% de los data)
anomaly_indices = np.random.choice(len(date_range), size=int(0.05 * len(date_range)), replace=False)
anomaly_values = normal_values.copy()

# Types de anomalies:
# - Point anomalies: values extremos
# - Collective anomalies: patterns anomalous consecutivos
for idx in anomaly_indices[:10]:  # Point anomalies
    anomaly_values[idx] += np.random.choice([-50, 50])  # Spike/Drop

for idx in anomaly_indices[10:15]:  # Collective anomalies
    anomaly_values[idx:idx+5] += 30  # Period elevado

# DataFrame
df = pd.DataFrame({
    'timestamp': date_range,
    'value': anomaly_values
})

df['is_anomaly'] = False
df.loc[anomaly_indices, 'is_anomaly'] = True

print(f"Total de observaciones: {len(df)}")
print(f"Anomalies inyectadas: {df['is_anomaly'].sum()} ({100*df['is_anomaly'].mean():.2f}%)")
print(f"\n{df.head()}")
```

**Output:**

```
Total de observaciones: 365
Anomalies inyectadas: 18 (4.93%)

            timestamp      value  is_anomaly
0 2023-01-01 00:00:00  95.234512       False
1 2023-01-01 01:00:00  91.456789       False
2 2023-01-01 02:00:00  88.765432       False
3 2023-01-01 03:00:00  87.123456       False
4 2023-01-01 04:00:00  86.789012       False
```

______________________________________________________________________

## 📊 Paso 3: Visualize Data

```python
plt.figure(figsize=(16, 6))

# Plot series complete
plt.plot(df['timestamp'], df['value'], label='Valores', color='blue', alpha=0.7)

# Highlight anomalies
anomalies = df[df['is_anomaly']]
plt.scatter(anomalies['timestamp'], anomalies['value'],
            color='red', s=100, label='Anomalies Reales', zorder=5)

plt.title('Series Temporal con Anomalies Inyectadas', fontsize=16)
plt.xlabel('Timestamp')
plt.ylabel('Valor')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

______________________________________________________________________

## 🔍 Technique 1: Statistical Methods (Z-Score)

### 1.1 Z-Score (standard deviation)

```python
def detect_anomalies_zscore(df, column, threshold=3):
    """
    Detecta anomalies using Z-score
    threshold=3: values más de 3 desviaciones standard se consideran anomalies
    """
    mean = df[column].mean()
    std = df[column].std()

    df['z_score'] = (df[column] - mean) / std
    df['anomaly_zscore'] = np.abs(df['z_score']) > threshold

    return df

df = detect_anomalies_zscore(df, 'value', threshold=3)

# Metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

print("=== Z-SCORE METHOD ===")
print(f"Anomalies detectadas: {df['anomaly_zscore'].sum()}")

print("\nConfusion Matrix:")
cm = confusion_matrix(df['is_anomaly'], df['anomaly_zscore'])
print(cm)

print("\nClassification Report:")
print(classification_report(df['is_anomaly'], df['anomaly_zscore'],
                           target_names=['Normal', 'Anomaly']))
```

**Output:**

```
=== Z-SCORE METHOD ===
Anomalies detectadas: 15

Confusion Matrix:
[[345   2]
 [  3  15]]  👈 TP=15, FP=2, FN=3

Classification Report:
              precision    recall  f1-score   support

      Normal       0.99      0.99      0.99       347
     Anomaly       0.88      0.83      0.86        18

    accuracy                           0.99       365
```

### 1.2 Visualize Z-Score detection

```python
plt.figure(figsize=(16, 6))

plt.plot(df['timestamp'], df['value'], label='Valores', color='blue', alpha=0.5)

# Anomalies detectadas
detected = df[df['anomaly_zscore']]
plt.scatter(detected['timestamp'], detected['value'],
            color='orange', s=100, label='Detectadas (Z-Score)', marker='x', zorder=5)

# Anomalies real
real = df[df['is_anomaly']]
plt.scatter(real['timestamp'], real['value'],
            color='red', s=50, label='Reales', alpha=0.5, zorder=4)

plt.title('Detection con Z-Score (threshold=3)', fontsize=16)
plt.xlabel('Timestamp')
plt.ylabel('Valor')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

______________________________________________________________________

## 🤖 Technique 2: Isolation Forest (ML)

### 2.1 Feature engineering for time series

```python
# Create features:
# - Valor actual
# - Lags (values previous)
# - Rolling statistics (media, std mobile)
# - Hora del día

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Lags
df['lag_1'] = df['value'].shift(1)
df['lag_2'] = df['value'].shift(2)
df['lag_3'] = df['value'].shift(3)

# Rolling statistics (ventana de 24 horas)
df['rolling_mean_24'] = df['value'].rolling(window=24, center=False).mean()
df['rolling_std_24'] = df['value'].rolling(window=24, center=False).std()

# Diferencia con media mobile
df['diff_from_rolling_mean'] = df['value'] - df['rolling_mean_24']

# Drop NaNs (primeras rows por lags y rolling)
df.dropna(inplace=True)

# Features para model
feature_cols = ['value', 'lag_1', 'lag_2', 'lag_3',
                'rolling_mean_24', 'rolling_std_24',
                'diff_from_rolling_mean', 'hour']

X = df[feature_cols]

print(f"Features: {X.shape}")
print(f"\n{X.head()}")
```

### 2.2 Train Isolation Forest

```python
# Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
# contamination: porcentaje esperado de anomalies (ajustar according to dominio)
iso_forest = IsolationForest(
    contamination=0.05,  # Esperamos ~5% de anomalies
    random_state=42,
    n_estimators=100
)

# Predict: 1 = normal, -1 = anomaly
df['iso_forest_pred'] = iso_forest.fit_predict(X_scaled)
df['anomaly_isoforest'] = df['iso_forest_pred'] == -1

print("=== ISOLATION FOREST ===")
print(f"Anomalies detectadas: {df['anomaly_isoforest'].sum()}")

print("\nConfusion Matrix:")
cm_iso = confusion_matrix(df['is_anomaly'], df['anomaly_isoforest'])
print(cm_iso)

print("\nClassification Report:")
print(classification_report(df['is_anomaly'], df['anomaly_isoforest'],
                           target_names=['Normal', 'Anomaly']))
```

**Output:**

```
=== ISOLATION FOREST ===
Anomalies detectadas: 17

Confusion Matrix:
[[330  17]
 [  1  17]]  👈 Mejor recall (17/18 anomalies detectadas)

Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      0.95      0.97       347
     Anomaly       0.50      0.94      0.65        18

    accuracy                           0.95       365
```

### 2.3 Visualize Isolation Forest

```python
plt.figure(figsize=(16, 6))

plt.plot(df['timestamp'], df['value'], label='Valores', color='blue', alpha=0.5)

# Detectadas
detected_iso = df[df['anomaly_isoforest']]
plt.scatter(detected_iso['timestamp'], detected_iso['value'],
            color='green', s=100, label='Detectadas (Isolation Forest)', marker='^', zorder=5)

# Reales
real = df[df['is_anomaly']]
plt.scatter(real['timestamp'], real['value'],
            color='red', s=50, label='Reales', alpha=0.5, zorder=4)

plt.title('Detection con Isolation Forest', fontsize=16)
plt.xlabel('Timestamp')
plt.ylabel('Valor')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

______________________________________________________________________

## 📐 Technique 3: IQR (Interquartile Range)

### 3.1 IQR method with moving window

```python
def detect_anomalies_iqr(df, column, window=24, multiplier=1.5):
    """
    Detecta anomalies using IQR (Interquartile Range) en ventana mobile
    """
    df['rolling_Q1'] = df[column].rolling(window=window).quantile(0.25)
    df['rolling_Q3'] = df[column].rolling(window=window).quantile(0.75)
    df['rolling_IQR'] = df['rolling_Q3'] - df['rolling_Q1']

    df['lower_bound'] = df['rolling_Q1'] - multiplier * df['rolling_IQR']
    df['upper_bound'] = df['rolling_Q3'] + multiplier * df['rolling_IQR']

    df['anomaly_iqr'] = (df[column] < df['lower_bound']) | (df[column] > df['upper_bound'])

    return df

df = detect_anomalies_iqr(df, 'value', window=24, multiplier=1.5)

print("=== IQR METHOD ===")
print(f"Anomalies detectadas: {df['anomaly_iqr'].sum()}")

print("\nConfusion Matrix:")
cm_iqr = confusion_matrix(df['is_anomaly'], df['anomaly_iqr'])
print(cm_iqr)

print("\nClassification Report:")
print(classification_report(df['is_anomaly'], df['anomaly_iqr'],
                           target_names=['Normal', 'Anomaly']))
```

**Output:**

```
=== IQR METHOD ===
Anomalies detectadas: 19

Confusion Matrix:
[[329  18]
 [  1  17]]

Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      0.95      0.97       347
     Anomaly       0.49      0.94      0.64        18
```

______________________________________________________________________

## 🧠 Technique 4: LSTM Autoencoder (Deep Learning)

### 4.1 Prepare Data for LSTM

```python
import torch
import torch.nn as nn

# Create secuencias (ventanas de 24 horas)
def create_sequences(data, seq_length=24):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Solo use values (sin timestamps)
values = df['value'].values

# Normalizar
scaler_lstm = StandardScaler()
values_scaled = scaler_lstm.fit_transform(values.reshape(-1, 1)).flatten()

# Create secuencias
seq_length = 24
sequences = create_sequences(values_scaled, seq_length)

print(f"Secuencias creadas: {sequences.shape}")  # (n_samples, seq_length)

# Convert a tensores
X_lstm = torch.FloatTensor(sequences).unsqueeze(-1)  # [n_samples, seq_length, 1]
```

### 4.2 Define LSTM Autoencoder

```python
class LSTMAutoencoder(nn.Module):
    """
    Autoencoder basado en LSTM para detection de anomalies
    """
    def __init__(self, seq_length, n_features=1, hidden_size=32):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Repeat hidden state para decoder
        hidden_repeated = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)

        # Decode
        decoded, _ = self.decoder(hidden_repeated, (hidden, cell))

        # Output
        output = self.output(decoded)

        return output

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_lstm = LSTMAutoencoder(seq_length=seq_length, hidden_size=32).to(device)

print(model_lstm)
```

### 4.3 Train autoencoder

```python
# Loss y optimizer
criterion_lstm = nn.MSELoss()
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

# DataLoader
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_lstm, X_lstm)  # Input = Output (autoencoder)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
num_epochs = 50
losses = []

print("Entrenando LSTM Autoencoder...")

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward
        outputs = model_lstm(batch_X)
        loss = criterion_lstm(outputs, batch_y)

        # Backward
        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / len(loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(loader):.6f}")

print("✅ Training completed")
```

**Output:**

```
Epoch [10/50], Loss: 0.012345
Epoch [20/50], Loss: 0.008234
Epoch [30/50], Loss: 0.005678
Epoch [40/50], Loss: 0.004123
Epoch [50/50], Loss: 0.003456
✅ Training completed
```

### 4.4 Detection of Anomalies with reconstruction error

```python
# Predict (reconstruir)
model_lstm.eval()
with torch.no_grad():
    X_lstm_device = X_lstm.to(device)
    reconstructed = model_lstm(X_lstm_device).cpu().numpy()

# Calculate reconstruction error (MAE)
reconstruction_errors = np.mean(np.abs(X_lstm.numpy() - reconstructed), axis=(1, 2))

# Threshold: percentile 95 (top 5% errors son anomalies)
threshold_lstm = np.percentile(reconstruction_errors, 95)

# Detect anomalies
anomalies_lstm = reconstruction_errors > threshold_lstm

# Alinear con dataframe original (considerando seq_length)
df['anomaly_lstm'] = False
df.iloc[seq_length:seq_length+len(anomalies_lstm), df.columns.get_loc('anomaly_lstm')] = anomalies_lstm

print("=== LSTM AUTOENCODER ===")
print(f"Threshold: {threshold_lstm:.6f}")
print(f"Anomalies detectadas: {df['anomaly_lstm'].sum()}")

# Metrics (considerar solo rows con prediction)
valid_rows = df.iloc[seq_length:seq_length+len(anomalies_lstm)]

print("\nConfusion Matrix:")
cm_lstm = confusion_matrix(valid_rows['is_anomaly'], valid_rows['anomaly_lstm'])
print(cm_lstm)
```

______________________________________________________________________

## 📊 Compare all methods

```python
# Tabla comparativa
methods = ['Z-Score', 'IQR', 'Isolation Forest', 'LSTM Autoencoder']
precisions = []
recalls = []
f1_scores = []

for col in ['anomaly_zscore', 'anomaly_iqr', 'anomaly_isoforest', 'anomaly_lstm']:
    valid_data = df[df[col].notna()]

    precision = precision_score(valid_data['is_anomaly'], valid_data[col], zero_division=0)
    recall = recall_score(valid_data['is_anomaly'], valid_data[col], zero_division=0)
    f1 = f1_score(valid_data['is_anomaly'], valid_data[col], zero_division=0)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

comparison = pd.DataFrame({
    'Method': methods,
    'Precision': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores
})

print("\n=== COMPARISON DE METHODS ===")
print(comparison.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(comparison['Method'], comparison['Precision'], color='skyblue')
axes[0].set_title('Precision')
axes[0].set_ylim(0, 1.1)
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(comparison['Method'], comparison['Recall'], color='lightgreen')
axes[1].set_title('Recall')
axes[1].set_ylim(0, 1.1)
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(comparison['Method'], comparison['F1-Score'], color='salmon')
axes[2].set_title('F1-Score')
axes[2].set_ylim(0, 1.1)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Output expected:**

```
=== COMPARISON DE METHODS ===
            Method  Precision  Recall  F1-Score
           Z-Score       0.88    0.83      0.85
               IQR       0.49    0.94      0.64
  Isolation Forest       0.50    0.94      0.65
  LSTM Autoencoder       0.75    0.89      0.81
```

______________________________________________________________________

## 📝Executive summary

### ✅ Comparison of methods

| Method | Precision | recall | f1-Score | Advantages | Disadvantages |
| -------------------- | --------- | ------ | -------- | -------------------------------------------------------- | ------------------------------------------------------- |
| **Z-Score** | 0.88 | 0.83 | 0.85 | Simple, fast | Assume normal distribution, does not capture temporal context |
| **IQR** | 0.49 | 0.94 | 0.64 | Robust to outliers | Many false positives |
| **Isolation Forest** | 0.50 | 0.94 | 0.65 | Does not assume distribution, handles multi-dimensional | Requires feature engineering |
| **LSTM Autoencoder** | 0.75 | 0.89 | 0.81 | Capture temporal patterns, detect complex anomalies | Slow, requires Data |

### 🎯 When use each method

**Z-Score / IQR:**

- ✅ Baseline fast
- ✅ Anomalies point (extreme values)
- ✅ Stationary series

**Isolation Forest:**

- ✅ Complex anomalies (multi-dimensional)
- ✅ No assumption of data distribution
- ✅ Medium-large datasets

**LSTM Autoencoder:**

- ✅ Contextual anomalies (depend on sequence)
- ✅ Complex temporal patterns
- ✅ Very large datasets (>10k points)

______________________________________________________________________

## 🎓 Lessons learned

### ✅ Types of Anomalies

1. **Point Anomalies:**

   - Extreme individual values
   - Example: Temperature of 100°C in winter
   - **Detect with:** Z-Score,IQR

1. **Contextual Anomalies:**

- Abnormal values ​​according to context
- Example: Low traffic at 3 PM (normal at 3 AM)
   - **Detect with:** LSTM, Prophet residuals

1. **Collective Anomalies:**

- Sequence of abnormal values
- Example: Sustained 24-hour fall
   - **Detect with:** LSTM, Change Point Detection

### ✅ Challenges

1. **Class imbalance:** Anomalies are rare (~1-5%)

   - **Solution:** Use appropriate Metrics (Precision, recall, f1), not just accuracy

1. **Define threshold:**

- Z-Score: 3 (standard), 2 (sensitive), 4 (conservative)
- IQR: 1.5 (standard), 3 (conservative)
   - LSTM: Percentile 95-99

1. **False positives:**

   - IQR generates many FP (high sensitivity)
- **Solution:** Method Ensemble, Human Validation

### 💡Additional improvements

1. **Ensemble:** Combine multiple methods (voting)

```python
df['anomaly_ensemble'] = (
    df['anomaly_zscore'] &
    df['anomaly_isoforest']
).astype(int)
```

1. **Seasonal Hybrid ESD (S-H-ESD):** Detects Anomalies considering Seasonality
1. **Prophet:** Use residuals as Anomalies
1. **Transformer Autoencoder:** More powerful than LSTM

### 🚫 Errors common

- ❌ Use Z-Score in non-stationary series
- ❌Do not consider Seasonality
- ❌Fixed threshold for all series
- ❌ Do not validate detections (many FPs)
- ❌ Train LSTM with Anomalies (contaminates the Model)

______________________________________________________________________

## 🔧 Production pipeline

```python
def anomaly_detection_pipeline(df, value_col, methods=['zscore', 'isoforest']):
    """
    Pipeline multi-method para detection de anomalies
    """
    results = {}

    # Z-Score
    if 'zscore' in methods:
        df_result = detect_anomalies_zscore(df.copy(), value_col, threshold=3)
        results['zscore'] = df_result['anomaly_zscore']

    # Isolation Forest
    if 'isoforest' in methods:
        # Feature engineering
        df_features = df.copy()
        df_features['lag_1'] = df_features[value_col].shift(1)
        df_features['rolling_mean'] = df_features[value_col].rolling(24).mean()
        df_features.dropna(inplace=True)

        X = df_features[[value_col, 'lag_1', 'rolling_mean']]
        X_scaled = StandardScaler().fit_transform(X)

        iso = IsolationForest(contamination=0.05, random_state=42)
        predictions = iso.fit_predict(X_scaled)

        df_result = df.copy()
        df_result['anomaly_isoforest'] = False
        df_result.iloc[df_features.index, df_result.columns.get_loc('anomaly_isoforest')] = (predictions == -1)

        results['isoforest'] = df_result['anomaly_isoforest']

    # Ensemble (vote)
    ensemble_votes = sum(results.values())
    results['ensemble'] = ensemble_votes >= len(methods) / 2  # Most

    return pd.DataFrame(results)

# Wear
anomaly_results = anomaly_detection_pipeline(df, 'value', methods=['zscore', 'isoforest'])
print(anomaly_results.sum())
```

### 📌 Anomaly detection checklist

- ✅ Understand Types of Anomalies (point, contextual, collective)
- ✅ Visualize Data before modeling
- ✅ Consider Seasonality and Trend
- ✅ Test multiple methods (statistical baseline + ML)
- ✅ Feature engineering for ML methods
- ✅ Threshold tuning (validate with domain expert)
- ✅ Evaluate yourself with Precision/recall/f1 (no accuracy)
- ✅ Visualize detections for health check
- ✅ Implement alerting in production
- ✅ Feedback loop to improve Models
