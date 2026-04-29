# Practice 01 — Time series forecasting

## 🎯 Objectives

- Descomponer time series
- Implement Models ARIMA and Prophet
- Validate forecasts with Metrics
- Detect Seasonality

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Analysis Exploratorio

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Similar series temporal
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 200, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arrange(365) / 365)
noise = np.random.normal(0, 5, 365)
values = trend + seasonal + noise

ts = pd.Series(values, index=dates)

# Decomposition
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

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: ARIMA Manual

**Statement:**

1. Apply ADF test for stationarity
1. Diferencia series si no estacionaria
1. Determine (p,d,q) with ACF/PACF
1. Train ARIMA manualmente
1. Forecast 30 days

### Exercise 2.2: Prophet with Holidays

**Statement:**
Usa Facebook Prophet:

- Add custom holidays
- Model trend changes
- Evaluate forecast with MAE

### Exercise 2.3: LSTM for Forecasting

**Statement:**
Implement LSTM in PyTorch:

- Ventanas deslizantes (window=30)
- Predict next valor
- Compare with ARIMA

### Exercise 2.4: Cross-Validation Temporal

**Statement:**
Implement time series CV:

- Training set creciente
- Test set fijo
- Rolling window validation

### Exercise 2.5: Multi-Step Forecast

**Statement:**
Forecast multiple steps:

- Direct multi-step
- Recursive one-step
- Compare ambos approaches

______________________________________________________________________

## ✅ Checklist

- [ ] Descomponer time series
- [ ] Stationarity test
- [ ] ARIMA and Prophet
- [ ] Cross-validation temporal
- [ ] Multi-step forecasting

______________________________________________________________________

## 📚 Resources

- [statsmodels TSA](https://www.statsmodels.org/stable/tsa.html)
- [Prophet Docs](https://facebook.github.io/prophet/)
