# Practice 02 — Anomaly Detection

## 🎯 Objectives

- Implement detection algorithms
- Identify outliers temporales
- Evaluate yourself with precision/recall
- Apply in production

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Statistical Methods

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Data con anomalies
np.random.seed(42)
normal = np.random.normal(100, 10, 950)
anomalies = np.random.uniform(150, 200, 50)
data = np.concatenate([normal, anomalies])
np.random.shuffle(data)

# Z-score
z_scores = (data - data.mean()) / data.std()
anomalies_z = np.abs(z_scores) > 3

print(f"Anomalies detectadas (Z-score): {anomalies_z.sum()}")

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
predictions = iso_forest.fit_predict(data.reshape(-1, 1))
anomalies_iso = predictions == -1

print(f"Anomalies detectadas (IsoForest): {anomalies_iso.sum()}")
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: LSTM Autoencoder

**Statement:**
Implement autoencoder for Anomalies:

- Encoder comprime Time series
- Decoder reconstruye
- Threshold on reconstruction error

### Exercise 2.2: Sliding Window

**Statement:**
Detect Anomalies contextuales:

- 24 hour sliding window
- Compare with historical windows
- Flag if deviation > 3σ

### Exercise 2.3: Ensemble Detector

**Statement:**
Combine multiple detectores:

- Z-score + IQR + IsolationForest + LSTM
- Voting: majority marks as Anomaly
- Compare with individual detectors

### Exercise 2.4: Real-Time Detection

**Statement:**
Streaming system:

- Process Data point by point
- Actualiza detectores incrementalmente
- Latencia < 100ms

### Exercise 2.5: Anomaly Explanation

**Statement:**
Explain why it is Anomaly:

- Feature importance
- Comparison with normal distribution
- Contextual info (what features are off)

______________________________________________________________________

## ✅ Checklist

- [ ] Z-score e IQR
- [ ] Isolation Forest
- [ ] LSTM Autoencoder
- [ ] Ensemble methods
- [ ] Real-time detection

______________________________________________________________________

## 📚 Resources

- [PyOD Library](https://github.com/yzhao062/pyod)
- [Anomaly Detection Papers](https://paperswithcode.com/task/anomaly-detection)
