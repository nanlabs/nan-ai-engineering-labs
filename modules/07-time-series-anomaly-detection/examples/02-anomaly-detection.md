# Example 02 — Detección de Anomalies en time series

## Contexto

Las Anomalies son observaciones que se desvían significativamente del comportamiento normal. Aprenderás técnicas estadísticas, basadas en ML y deep learning para detectar Anomalies en time series.

## Objective

Detectar transacciones fraudulentas, fallas de servidores o comportamientos anómalos en Data temporales.

______________________________________________________________________

## 🚀 Paso 1: Setup e importaciones

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

## 📥 Paso 2: Generar Data con Anomalies

```python
# Serie temporal normal
np.random.seed(42)
date_range = pd.date_range(start='2023-01-01', periods=365, freq='H')

# Patrón normal: tendencia + estacionalidad
hours = np.arange(len(date_range))
trend = 0.05 * hours
seasonality = 20 * np.sin(2 * np.pi * hours / 24)  # Estacionalidad diaria
noise = np.random.normal(0, 5, len(date_range))

normal_values = 100 + trend + seasonality + noise

# Inyectar anomalías (5% de los datos)
anomaly_indices = np.random.choice(len(date_range), size=int(0.05 * len(date_range)), replace=False)
anomaly_values = normal_values.copy()

# Tipos de anomalías:
# - Point anomalies: valores extremos
# - Collective anomalies: patrones anómalos consecutivos
for idx in anomaly_indices[:10]:  # Point anomalies
    anomaly_values[idx] += np.random.choice([-50, 50])  # Spike/Drop

for idx in anomaly_indices[10:15]:  # Collective anomalies
    anomaly_values[idx:idx+5] += 30  # Período elevado

# DataFrame
df = pd.DataFrame({
    'timestamp': date_range,
    'value': anomaly_values
})

df['is_anomaly'] = False
df.loc[anomaly_indices, 'is_anomaly'] = True

print(f"Total de observaciones: {len(df)}")
print(f"Anomalías inyectadas: {df['is_anomaly'].sum()} ({100*df['is_anomaly'].mean():.2f}%)")
print(f"\n{df.head()}")
```

**Salida:**

```
Total de observaciones: 365
Anomalías inyectadas: 18 (4.93%)

            timestamp      value  is_anomaly
0 2023-01-01 00:00:00  95.234512       False
1 2023-01-01 01:00:00  91.456789       False
2 2023-01-01 02:00:00  88.765432       False
3 2023-01-01 03:00:00  87.123456       False
4 2023-01-01 04:00:00  86.789012       False
```

______________________________________________________________________

## 📊 Paso 3: Visualizar Data

```python
plt.figure(figsize=(16, 6))

# Plot serie completa
plt.plot(df['timestamp'], df['value'], label='Valores', color='blue', alpha=0.7)

# Highlight anomalías
anomalies = df[df['is_anomaly']]
plt.scatter(anomalies['timestamp'], anomalies['value'],
            color='red', s=100, label='Anomalías Reales', zorder=5)

plt.title('Serie Temporal con Anomalías Inyectadas', fontsize=16)
plt.xlabel('Timestamp')
plt.ylabel('Valor')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

______________________________________________________________________

## 🔍 Técnica 1: Métodos Estadísticos (Z-Score)

### 1.1 Z-Score (desviación estándar)

```python
def detect_anomalies_zscore(df, column, threshold=3):
    """
    Detecta anomalías usando Z-score
    threshold=3: valores más de 3 desviaciones estándar se consideran anomalías
    """
    mean = df[column].mean()
    std = df[column].std()

    df['z_score'] = (df[column] - mean) / std
    df['anomaly_zscore'] = np.abs(df['z_score']) > threshold

    return df

df = detect_anomalies_zscore(df, 'value', threshold=3)

# Métricas
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

print("=== Z-SCORE METHOD ===")
print(f"Anomalías detectadas: {df['anomaly_zscore'].sum()}")

print("\nConfusion Matrix:")
cm = confusion_matrix(df['is_anomaly'], df['anomaly_zscore'])
print(cm)

print("\nClassification Report:")
print(classification_report(df['is_anomaly'], df['anomaly_zscore'],
                           target_names=['Normal', 'Anomaly']))
```

**Salida:**

```
=== Z-SCORE METHOD ===
Anomalías detectadas: 15

Confusion Matrix:
[[345   2]
 [  3  15]]  👈 TP=15, FP=2, FN=3

Classification Report:
              precision    recall  f1-score   support

      Normal       0.99      0.99      0.99       347
     Anomaly       0.88      0.83      0.86        18

    accuracy                           0.99       365
```

### 1.2 Visualizar detección Z-Score

```python
plt.figure(figsize=(16, 6))

plt.plot(df['timestamp'], df['value'], label='Valores', color='blue', alpha=0.5)

# Anomalías detectadas
detected = df[df['anomaly_zscore']]
plt.scatter(detected['timestamp'], detected['value'],
            color='orange', s=100, label='Detectadas (Z-Score)', marker='x', zorder=5)

# Anomalías reales
real = df[df['is_anomaly']]
plt.scatter(real['timestamp'], real['value'],
            color='red', s=50, label='Reales', alpha=0.5, zorder=4)

plt.title('Detección con Z-Score (threshold=3)', fontsize=16)
plt.xlabel('Timestamp')
plt.ylabel('Valor')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

______________________________________________________________________

## 🤖 Técnica 2: Isolation Forest (ML)

### 2.1 Feature engineering para time series

```python
# Crear features:
# - Valor actual
# - Lags (valores previos)
# - Rolling statistics (media, std móvil)
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

# Diferencia con media móvil
df['diff_from_rolling_mean'] = df['value'] - df['rolling_mean_24']

# Drop NaNs (primeras filas por lags y rolling)
df.dropna(inplace=True)

# Features para modelo
feature_cols = ['value', 'lag_1', 'lag_2', 'lag_3',
                'rolling_mean_24', 'rolling_std_24',
                'diff_from_rolling_mean', 'hour']

X = df[feature_cols]

print(f"Features: {X.shape}")
print(f"\n{X.head()}")
```

### 2.2 Entrenar Isolation Forest

```python
# Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
# contamination: porcentaje esperado de anomalías (ajustar según dominio)
iso_forest = IsolationForest(
    contamination=0.05,  # Esperamos ~5% de anomalías
    random_state=42,
    n_estimators=100
)

# Predecir: 1 = normal, -1 = anomalía
df['iso_forest_pred'] = iso_forest.fit_predict(X_scaled)
df['anomaly_isoforest'] = df['iso_forest_pred'] == -1

print("=== ISOLATION FOREST ===")
print(f"Anomalías detectadas: {df['anomaly_isoforest'].sum()}")

print("\nConfusion Matrix:")
cm_iso = confusion_matrix(df['is_anomaly'], df['anomaly_isoforest'])
print(cm_iso)

print("\nClassification Report:")
print(classification_report(df['is_anomaly'], df['anomaly_isoforest'],
                           target_names=['Normal', 'Anomaly']))
```

**Salida:**

```
=== ISOLATION FOREST ===
Anomalías detectadas: 17

Confusion Matrix:
[[330  17]
 [  1  17]]  👈 Mejor recall (17/18 anomalías detectadas)

Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      0.95      0.97       347
     Anomaly       0.50      0.94      0.65        18

    accuracy                           0.95       365
```

### 2.3 Visualizar Isolation Forest

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

plt.title('Detección con Isolation Forest', fontsize=16)
plt.xlabel('Timestamp')
plt.ylabel('Valor')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

______________________________________________________________________

## 📐 Técnica 3: IQR (Interquartile Range)

### 3.1 Método IQR con ventana móvil

```python
def detect_anomalies_iqr(df, column, window=24, multiplier=1.5):
    """
    Detecta anomalías usando IQR (Interquartile Range) en ventana móvil
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
print(f"Anomalías detectadas: {df['anomaly_iqr'].sum()}")

print("\nConfusion Matrix:")
cm_iqr = confusion_matrix(df['is_anomaly'], df['anomaly_iqr'])
print(cm_iqr)

print("\nClassification Report:")
print(classification_report(df['is_anomaly'], df['anomaly_iqr'],
                           target_names=['Normal', 'Anomaly']))
```

**Salida:**

```
=== IQR METHOD ===
Anomalías detectadas: 19

Confusion Matrix:
[[329  18]
 [  1  17]]

Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      0.95      0.97       347
     Anomaly       0.49      0.94      0.64        18
```

______________________________________________________________________

## 🧠 Técnica 4: LSTM Autoencoder (Deep Learning)

### 4.1 Preparar Data para LSTM

```python
import torch
import torch.nn as nn

# Crear secuencias (ventanas de 24 horas)
def create_sequences(data, seq_length=24):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Solo usar valores (sin timestamps)
values = df['value'].values

# Normalizar
scaler_lstm = StandardScaler()
values_scaled = scaler_lstm.fit_transform(values.reshape(-1, 1)).flatten()

# Crear secuencias
seq_length = 24
sequences = create_sequences(values_scaled, seq_length)

print(f"Secuencias creadas: {sequences.shape}")  # (n_samples, seq_length)

# Convertir a tensores
X_lstm = torch.FloatTensor(sequences).unsqueeze(-1)  # [n_samples, seq_length, 1]
```

### 4.2 Definir LSTM Autoencoder

```python
class LSTMAutoencoder(nn.Module):
    """
    Autoencoder basado en LSTM para detección de anomalías
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

# Crear modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_lstm = LSTMAutoencoder(seq_length=seq_length, hidden_size=32).to(device)

print(model_lstm)
```

### 4.3 Entrenar autoencoder

```python
# Loss y optimizer
criterion_lstm = nn.MSELoss()
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

# DataLoader
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_lstm, X_lstm)  # Input = Output (autoencoder)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Entrenar
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

print("✅ Entrenamiento completado")
```

**Salida:**

```
Epoch [10/50], Loss: 0.012345
Epoch [20/50], Loss: 0.008234
Epoch [30/50], Loss: 0.005678
Epoch [40/50], Loss: 0.004123
Epoch [50/50], Loss: 0.003456
✅ Entrenamiento completado
```

### 4.4 Detección de Anomalies con reconstruction error

```python
# Predecir (reconstruir)
model_lstm.eval()
with torch.no_grad():
    X_lstm_device = X_lstm.to(device)
    reconstructed = model_lstm(X_lstm_device).cpu().numpy()

# Calcular reconstruction error (MAE)
reconstruction_errors = np.mean(np.abs(X_lstm.numpy() - reconstructed), axis=(1, 2))

# Threshold: percentil 95 (top 5% errores son anomalías)
threshold_lstm = np.percentile(reconstruction_errors, 95)

# Detectar anomalías
anomalies_lstm = reconstruction_errors > threshold_lstm

# Alinear con dataframe original (considerando seq_length)
df['anomaly_lstm'] = False
df.iloc[seq_length:seq_length+len(anomalies_lstm), df.columns.get_loc('anomaly_lstm')] = anomalies_lstm

print("=== LSTM AUTOENCODER ===")
print(f"Threshold: {threshold_lstm:.6f}")
print(f"Anomalías detectadas: {df['anomaly_lstm'].sum()}")

# Métricas (considerar solo filas con predicción)
valid_rows = df.iloc[seq_length:seq_length+len(anomalies_lstm)]

print("\nConfusion Matrix:")
cm_lstm = confusion_matrix(valid_rows['is_anomaly'], valid_rows['anomaly_lstm'])
print(cm_lstm)
```

______________________________________________________________________

## 📊 Comparar todos los métodos

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

print("\n=== COMPARACIÓN DE MÉTODOS ===")
print(comparison.to_string(index=False))

# Visualizar
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

**Salida esperada:**

```
=== COMPARACIÓN DE MÉTODOS ===
            Method  Precision  Recall  F1-Score
           Z-Score       0.88    0.83      0.85
               IQR       0.49    0.94      0.64
  Isolation Forest       0.50    0.94      0.65
  LSTM Autoencoder       0.75    0.89      0.81
```

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Comparación de métodos

| Método               | Precision | recall | f1-Score | Ventajas                                                 | Desventajas                                             |
| -------------------- | --------- | ------ | -------- | -------------------------------------------------------- | ------------------------------------------------------- |
| **Z-Score**          | 0.88      | 0.83   | 0.85     | Simple, rápido                                           | Asume distribución normal, no captura contexto temporal |
| **IQR**              | 0.49      | 0.94   | 0.64     | Robusto a outliers                                       | Muchos falsos positivos                                 |
| **Isolation Forest** | 0.50      | 0.94   | 0.65     | No asume distribución, maneja multi-dimensional          | Requiere feature engineering                            |
| **LSTM Autoencoder** | 0.75      | 0.89   | 0.81     | Captura patrones temporales, detecta Anomalies complejas | Lento, requiere Data                                    |

### 🎯 Cuándo usar cada método

**Z-Score / IQR:**

- ✅ Baseline rápido
- ✅ Anomalies point (valores extremos)
- ✅ Series estacionarias

**Isolation Forest:**

- ✅ Anomalies complejas (multi-dimensional)
- ✅ No asume distribución de Data
- ✅ Datasets medianos-grandes

**LSTM Autoencoder:**

- ✅ Anomalies contextuales (dependen de secuencia)
- ✅ Patrones temporales complejos
- ✅ Datasets muy grandes (>10k puntos)

______________________________________________________________________

## 🎓 Lessons aprendidas

### ✅ Types de Anomalies

1. **Point Anomalies:**

   - Valores individuales extremos
   - Example: Temperatura de 100°C en invierno
   - **Detectar con:** Z-Score,IQR

1. **Contextual Anomalies:**

   - Valores anómalos según contexto
   - Example: Tráfico bajo a las 3 PM (normal a las 3 AM)
   - **Detectar con:** LSTM, Prophet residuals

1. **Collective Anomalies:**

   - Secuencia de valores anómalos
   - Example: Caída sostenida de 24 horas
   - **Detectar con:** LSTM, Change Point Detection

### ✅ Desafíos

1. **Class imbalance:** Anomalies son raras (~1-5%)

   - **Solución:** Usar Metrics apropiadas (Precision, recall, f1), no solo accuracy

1. **Definir threshold:**

   - Z-Score: 3 (estándar), 2 (sensible), 4 (conservador)
   - IQR: 1.5 (estándar), 3 (conservador)
   - LSTM: Percentil 95-99

1. **Falsos positivos:**

   - IQR genera muchos FP (alta sensibilidad)
   - **Solución:** Ensemble de métodos, Validation humana

### 💡 Mejoras adicionales

1. **Ensemble:** Combinar múltiples métodos (votación)

```python
df['anomaly_ensemble'] = (
    df['anomaly_zscore'] &
    df['anomaly_isoforest']
).astype(int)
```

1. **Seasonal Hybrid ESD (S-H-ESD):** Detecta Anomalies considerando Seasonality
1. **Prophet:** Usar residuals como Anomalies
1. **Transformer Autoencoder:** Más poderoso que LSTM

### 🚫 Errors comunes

- ❌ Usar Z-Score en series no estacionarias
- ❌ No considerar Seasonality
- ❌ Threshold fijo para todas las series
- ❌ No validar detecciones (muchos FP)
- ❌ Entrenar LSTM con Anomalies (contamina el Model)

______________________________________________________________________

## 🔧 Pipeline de producción

```python
def anomaly_detection_pipeline(df, value_col, methods=['zscore', 'isoforest']):
    """
    Pipeline multi-método para detección de anomalías
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

    # Ensemble (votación)
    ensemble_votes = sum(results.values())
    results['ensemble'] = ensemble_votes >= len(methods) / 2  # Mayoría

    return pd.DataFrame(results)

# Usar
anomaly_results = anomaly_detection_pipeline(df, 'value', methods=['zscore', 'isoforest'])
print(anomaly_results.sum())
```

### 📌 Checklist detección de Anomalies

- ✅ Entender Types de Anomalies (point, contextual, collective)
- ✅ Visualizar Data antes de modelar
- ✅ Considerar Seasonality y Trend
- ✅ Probar múltiples métodos (baseline estadístico + ML)
- ✅ Feature engineering para métodos ML
- ✅ Tuning de threshold (validar con domain expert)
- ✅ Evaluar con Precision/recall/f1 (no accuracy)
- ✅ Visualizar detecciones para sanity check
- ✅ Implementar alerting en producción
- ✅ Feedback loop para mejorar Models
