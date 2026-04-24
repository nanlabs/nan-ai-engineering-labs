# Práctica 02 — Detección de Anomalías

## 🎯 Objetivos

- Implementar algoritmos de detección
- Identificar outliers temporales
- Evaluar con precision/recall
- Aplicar en producción

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Statistical Methods

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Datos con anomalías
np.random.seed(42)
normal = np.random.normal(100, 10, 950)
anomalies = np.random.uniform(150, 200, 50)
data = np.concatenate([normal, anomalies])
np.random.shuffle(data)

# Z-score
z_scores = (data - data.mean()) / data.std()
anomalies_z = np.abs(z_scores) > 3

print(f"Anomalías detectadas (Z-score): {anomalies_z.sum()}")

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
predictions = iso_forest.fit_predict(data.reshape(-1, 1))
anomalies_iso = predictions == -1

print(f"Anomalías detectadas (IsoForest): {anomalies_iso.sum()}")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: LSTM Autoencoder

**Enunciado:**
Implementa autoencoder para anomalías:

- Encoder comprime serie temporal
- Decoder reconstruye
- Threshold en reconstruction error

### Ejercicio 2.2: Sliding Window

**Enunciado:**
Detecta anomalías contextuales:

- Ventana deslizante de 24 horas
- Compara con ventanas históricas
- Flag si desviación > 3σ

### Ejercicio 2.3: Ensemble Detector

**Enunciado:**
Combina múltiples detectores:

- Z-score + IQR + IsolationForest + LSTM
- Voting: mayoría marca como anomalía
- Compara con detectores individuales

### Ejercicio 2.4: Real-Time Detection

**Enunciado:**
Sistema de streaming:

- Procesa datos punto por punto
- Actualiza detectores incrementalmente
- Latencia < 100ms

### Ejercicio 2.5: Anomaly Explanation

**Enunciado:**
Explica por qué es anomalía:

- Feature importance
- Comparación con distribución normal
- Contextual info (qué features están off)

______________________________________________________________________

## ✅ Checklist

- [ ] Z-score e IQR
- [ ] Isolation Forest
- [ ] LSTM Autoencoder
- [ ] Ensemble methods
- [ ] Real-time detection

______________________________________________________________________

## 📚 Recursos

- [PyOD Library](https://github.com/yzhao062/pyod)
- [Anomaly Detection Papers](https://paperswithcode.com/task/anomaly-detection)
