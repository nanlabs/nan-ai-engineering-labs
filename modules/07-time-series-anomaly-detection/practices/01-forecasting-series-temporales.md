# Práctica 01 — Forecasting de Series Temporales

## 🎯 Objetivos

- Descomponer series temporales
- Implementar modelos ARIMA y Prophet
- Validar forecasts con métricas
- Detectar estacionalidad

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Análisis Exploratorio

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Simular serie temporal
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 200, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 5, 365)
values = trend + seasonal + noise

ts = pd.Series(values, index=dates)

# Descomposición
decomposition = seasonal_decompose(ts, model='additive', period=30)

fig, axes = plt.subplots(4, 1, figsize=(12, 8))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.savefig('decomposition.png')
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: ARIMA Manual

**Enunciado:**

1. Aplica ADF test para estacionariedad
1. Diferencia serie si no estacionaria
1. Determina (p,d,q) con ACF/PACF
1. Entrena ARIMA manualmente
1. Forecast 30 días

### Ejercicio 2.2: Prophet con Holidays

**Enunciado:**
Usa Facebook Prophet:

- Añade holidays personalizados
- Modela cambios de tendencia
- Evalúa forecast con MAE

### Ejercicio 2.3: LSTM para Forecasting

**Enunciado:**
Implementa LSTM en PyTorch:

- Ventanas deslizantes (window=30)
- Predict siguiente valor
- Compara con ARIMA

### Ejercicio 2.4: Cross-Validation Temporal

**Enunciado:**
Implementa time series CV:

- Training set creciente
- Test set fijo
- Rolling window validation

### Ejercicio 2.5: Multi-Step Forecast

**Enunciado:**
Forecast múltiples pasos:

- Direct multi-step
- Recursive one-step
- Compara ambos approaches

______________________________________________________________________

## ✅ Checklist

- [ ] Descomponer series temporales
- [ ] Test de estacionariedad
- [ ] ARIMA y Prophet
- [ ] Cross-validation temporal
- [ ] Multi-step forecasting

______________________________________________________________________

## 📚 Recursos

- [statsmodels TSA](https://www.statsmodels.org/stable/tsa.html)
- [Prophet Docs](https://facebook.github.io/prophet/)
