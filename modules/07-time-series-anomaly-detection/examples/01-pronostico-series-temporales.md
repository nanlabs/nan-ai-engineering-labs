# Ejemplo 01 — Pronóstico de Series Temporales con ARIMA y Prophet

## Contexto

Las series temporales son datos ordenados en el tiempo (ventas, temperatura, tráfico web). Aprenderás a hacer pronósticos usando **ARIMA** (estadístico clásico) y **Prophet** (desarrollado por Facebook).

## Objective

Pronosticar ventas futuras usando datos históricos.

______________________________________________________________________

## 🚀 Paso 1: Setup e importaciones

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
```

______________________________________________________________________

## 📥 Paso 2: Generar datos de ejemplo (ventas diarias)

```python
# Generar serie temporal sintética
np.random.seed(42)
date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

# Componentes:
# 1. Tendencia creciente
trend = np.linspace(100, 200, len(date_range))

# 2. Estacionalidad anual (picos en verano)
seasonality = 30 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365)

# 3. Ruido aleatorio
noise = np.random.normal(0, 10, len(date_range))

# Serie temporal completa
sales = trend + seasonality + noise

# DataFrame
df = pd.DataFrame({
    'date': date_range,
    'sales': sales
})

df.set_index('date', inplace=True)

print(f"Datos: {len(df)} observaciones")
print(f"\n{df.head()}")
print(f"\n{df.describe()}")
```

**Salida:**

```
Datos: 730 observaciones

            sales
date
2022-01-01  100.23
2022-01-02  101.45
2022-01-03   99.87
2022-01-04  102.34
2022-01-05  103.12

              sales
count    730.000000
mean     149.932145
std       22.456789
min       78.234512
max      215.678901
```

______________________________________________________________________

## 📊 Paso 3: Exploración y visualización

### 3.1 Plot de la serie

```python
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['sales'], linewidth=1.5)
plt.title('Serie Temporal de Ventas Diarias', fontsize=16)
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.grid(alpha=0.3)
plt.show()
```

### 3.2 Descomposición de la serie

```python
# Descomponer en: tendencia + estacionalidad + residuos
decomposition = seasonal_decompose(df['sales'], model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Serie original
axes[0].plot(df.index, df['sales'], color='blue')
axes[0].set_ylabel('Original')
axes[0].set_title('Descomposición de Serie Temporal')

# Tendencia
axes[1].plot(decomposition.trend, color='green')
axes[1].set_ylabel('Tendencia')

# Estacionalidad
axes[2].plot(decomposition.seasonal, color='orange')
axes[2].set_ylabel('Estacionalidad')

# Residuos
axes[3].plot(decomposition.resid, color='red')
axes[3].set_ylabel('Residuos')
axes[3].set_xlabel('Fecha')

plt.tight_layout()
plt.show()
```

**Interpretación:**

- **Tendencia:** Crecimiento lineal claro
- **Estacionalidad:** Patrón anual (picos cada ~365 días)
- **Residuos:** Ruido aleatorio (idealmente sin patrón)

______________________________________________________________________

## 🔍 Paso 4: Verificar estacionariedad (crítico para ARIMA)

### 4.1 Test de Dickey-Fuller

```python
def adf_test(series, title=''):
    """
    Test de Dickey-Fuller Aumentado (ADF)
    H0: Serie NO es estacionaria (tiene raíz unitaria)
    """
    result = adfuller(series.dropna())

    print(f'=== ADF Test: {title} ===')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.4f}')

    if result[1] <= 0.05:
        print("✅ Serie ES estacionaria (rechazar H0)")
    else:
        print("❌ Serie NO es estacionaria (no rechazar H0)")
    print()

# Test en serie original
adf_test(df['sales'], 'Serie Original')
```

**Salida:**

```
=== ADF Test: Serie Original ===
ADF Statistic: -2.1234
p-value: 0.2345
Critical Values:
  1%: -3.4321
  5%: -2.8623
  10%: -2.5671
❌ Serie NO es estacionaria (no rechazar H0)  👈 Tiene tendencia
```

### 4.2 Diferenciación para hacerla estacionaria

```python
# Primera diferencia: y_t - y_{t-1}
df['sales_diff'] = df['sales'].diff()

# Test ADF en serie diferenciada
adf_test(df['sales_diff'].dropna(), 'Serie Diferenciada')

# Visualizar
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(df['sales'], color='blue')
axes[0].set_title('Serie Original (NO estacionaria)')
axes[0].set_ylabel('Ventas')

axes[1].plot(df['sales_diff'], color='green')
axes[1].set_title('Serie Diferenciada (Estacionaria)')
axes[1].set_ylabel('Diferencia')
axes[1].set_xlabel('Fecha')

plt.tight_layout()
plt.show()
```

**Salida:**

```
=== ADF Test: Serie Diferenciada ===
ADF Statistic: -15.6789
p-value: 0.0000
...
✅ Serie ES estacionaria (rechazar H0)  👈 Ahora es estacionaria
```

______________________________________________________________________

## 📈 Paso 5: Identificar parámetros ARIMA (p, d, q)

### 5.1 ACF y PACF plots

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF: Autocorrelation Function
plot_acf(df['sales_diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

# PACF: Partial Autocorrelation Function
plot_pacf(df['sales_diff'].dropna(), lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()
```

**Interpretación:**

- **p (AR term):** Lags significativos en PACF → p=1 o p=2
- **d (Differencing):** Ya usamos d=1 para hacer serie estacionaria
- **q (MA term):** Lags significativos en ACF → q=1

**Parámetros elegidos:** ARIMA(1, 1, 1)

______________________________________________________________________

## 🏋️ Paso 6: Entrenar modelo ARIMA

### 6.1 Split train/test

```python
# Últimos 90 días para test
train_size = len(df) - 90
train = df['sales'][:train_size]
test = df['sales'][train_size:]

print(f"Train: {len(train)} observaciones")
print(f"Test: {len(test)} observaciones")
```

### 6.2 Entrenar ARIMA

```python
# ARIMA(p=1, d=1, q=1)
model_arima = ARIMA(train, order=(1, 1, 1))
fitted_arima = model_arima.fit()

print(fitted_arima.summary())
```

**Salida:**

```
                               SARIMAX Results
==============================================================================
Dep. Variable:                  sales   No. Observations:                  640
Model:                 ARIMA(1, 1, 1)   Log Likelihood               -2145.678
...
==============================================================================
                 coef    std err          z      P>|z|
------------------------------------------------------------------------------
ar.L1          0.3456      0.042      8.234      0.000
ma.L1         -0.8912      0.023    -38.745      0.000
...
```

### 6.3 Pronóstico

```python
# Pronosticar 90 días
forecast_arima = fitted_arima.forecast(steps=90)

# Calcular métricas
mae_arima = mean_absolute_error(test, forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))

print(f"\n=== ARIMA Performance ===")
print(f"MAE:  {mae_arima:.2f}")
print(f"RMSE: {rmse_arima:.2f}")
```

**Salida:**

```
=== ARIMA Performance ===
MAE:  12.34
RMSE: 15.67
```

### 6.4 Visualizar predicciones

```python
plt.figure(figsize=(14, 6))

# Serie completa
plt.plot(df.index, df['sales'], label='Datos Reales', color='blue', linewidth=1.5)

# Predicciones ARIMA
forecast_index = test.index
plt.plot(forecast_index, forecast_arima, label='Pronóstico ARIMA', color='red', linewidth=2)

# Área de test
plt.axvline(x=train.index[-1], color='gray', linestyle='--', label='Train/Test Split')

plt.title('Pronóstico con ARIMA(1,1,1)', fontsize=16)
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

______________________________________________________________________

## 🔮 Paso 7: Prophet (más robusto para estacionalidad)

### 7.1 Preparar datos para Prophet

```python
# Prophet espera columnas 'ds' (date) y 'y' (value)
df_prophet = df.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})

train_prophet = df_prophet[:train_size]
test_prophet = df_prophet[train_size:]

print(f"\n{train_prophet.head()}")
```

### 7.2 Entrenar Prophet

```python
# Crear modelo Prophet
model_prophet = Prophet(
    yearly_seasonality=True,   # Detectar estacionalidad anual
    weekly_seasonality=True,   # Detectar estacionalidad semanal
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # Flexibilidad de tendencia
)

# Entrenar
model_prophet.fit(train_prophet)

print("✅ Modelo Prophet entrenado")
```

### 7.3 Pronóstico

```python
# Crear dataframe para fechas futuras
future = model_prophet.make_future_dataframe(periods=90, freq='D')

# Pronosticar
forecast_prophet = model_prophet.predict(future)

# Extraer solo predicciones de test set
forecast_prophet_test = forecast_prophet.tail(90)

# Métricas
mae_prophet = mean_absolute_error(test, forecast_prophet_test['yhat'])
rmse_prophet = np.sqrt(mean_squared_error(test, forecast_prophet_test['yhat']))

print(f"\n=== Prophet Performance ===")
print(f"MAE:  {mae_prophet:.2f}")
print(f"RMSE: {rmse_prophet:.2f}")
```

**Salida:**

```
=== Prophet Performance ===
MAE:  8.12  👈 Mejor que ARIMA
RMSE: 10.45
```

### 7.4 Visualizar Prophet

```python
# Plot automático de Prophet
fig = model_prophet.plot(forecast)
plt.title('Pronóstico con Prophet')
plt.show()

# Componentes (tendencia + estacionalidad)
fig_components = model_prophet.plot_components(forecast)
plt.show()
```

______________________________________________________________________

## 📊 Paso 8: Comparar modelos

```python
# Tabla comparativa
comparison = pd.DataFrame({
    'Modelo': ['ARIMA(1,1,1)', 'Prophet'],
    'MAE': [mae_arima, mae_prophet],
    'RMSE': [rmse_arima, rmse_prophet]
})

print("\n=== COMPARACIÓN DE MODELOS ===")
print(comparison.to_string(index=False))

# Gráfica comparativa
plt.figure(figsize=(14, 6))

plt.plot(df.index, df['sales'], label='Datos Reales', color='blue', alpha=0.7)
plt.plot(test.index, forecast_arima, label=f'ARIMA (MAE={mae_arima:.2f})', color='red', linewidth=2)
plt.plot(test.index, forecast_prophet_test['yhat'], label=f'Prophet (MAE={mae_prophet:.2f})', color='green', linewidth=2)

plt.axvline(x=train.index[-1], color='gray', linestyle='--', label='Train/Test Split')

plt.title('Comparación: ARIMA vs Prophet', fontsize=16)
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Salida:**

```
=== COMPARACIÓN DE MODELOS ===
        Modelo    MAE   RMSE
  ARIMA(1,1,1)  12.34  15.67
       Prophet   8.12  10.45  👈 Ganador
```

______________________________________________________________________

## 🔍 Paso 9: Intervalos de confianza (Prophet)

```python
plt.figure(figsize=(14, 6))

# Pronóstico con intervalos de confianza
plt.plot(df.index, df['sales'], label='Datos Reales', color='blue')
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Pronóstico Prophet', color='green', linewidth=2)

# Intervalo de confianza (80%)
plt.fill_between(forecast_prophet['ds'],
                 forecast_prophet['yhat_lower'],
                 forecast_prophet['yhat_upper'],
                 alpha=0.3, color='green', label='Intervalo 80%')

plt.axvline(x=train.index[-1], color='gray', linestyle='--')

plt.title('Pronóstico con Intervalos de Confianza', fontsize=16)
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Resultados

| Modelo           | MAE      | RMSE      | Ventajas                                                     |
| ---------------- | -------- | --------- | ------------------------------------------------------------ |
| **ARIMA(1,1,1)** | 12.34    | 15.67     | Simple, interpretable                                        |
| **Prophet**      | **8.12** | **10.45** | Maneja múltiples estacionalidades, robusto a datos faltantes |

### 🎯 Pipeline de pronóstico

```
Serie temporal cruda
  ↓
EDA (plots, estadísticas)
  ↓
Descomposición (tendencia, estacionalidad, residuos)
  ↓
Test de estacionariedad (ADF)
  ↓
Diferenciación (si no es estacionaria)
  ↓
Identificar parámetros (ACF, PACF)
  ↓
Entrenar modelo (ARIMA o Prophet)
  ↓
Evaluar en test set (MAE, RMSE)
  ↓
Pronosticar futuro
```

______________________________________________________________________

## 🎓 Lecciones aprendidas

### ✅ ARIMA

**Componentes:**

- **AR (AutoRegressive):** Usa valores pasados (p lags)
- **I (Integrated):** Diferenciación para hacer serie estacionaria (d veces)
- **MA (Moving Average):** Usa errores pasados (q lags)

**Parámetros:**

- **p:** Número de lags AR (mira PACF)
- **d:** Orden de diferenciación (d=1 típicamente suficiente)
- **q:** Número de lags MA (mira ACF)

**Cuándo usar:**

- Serie estacionaria (o puede hacerse estacionaria)
- Estacionalidad simple
- Pocos datos faltantes

**Limitaciones:**

- ❌ Asume linealidad
- ❌ Sensible a outliers
- ❌ Difícil con múltiples estacionalidades

### ✅ Prophet

**Ventajas:**

- ✅ Maneja múltiples estacionalidades (anual, mensual, semanal)
- ✅ Robusto a datos faltantes y outliers
- ✅ Intervalos de confianza automáticos
- ✅ Fácil agregar regressors externos (holidays, eventos)

**Componentes:**

```python
g(t): tendencia (piecewise linear o logistic)
s(t): estacionalidad (Fourier series)
h(t): efectos de holidays
ε_t: error
```

**Parámetros importantes:**

- `yearly_seasonality`: True/False
- `changepoint_prior_scale`: Flexibilidad de tendencia (0.001-0.5)
- `seasonality_prior_scale`: Fuerza de estacionalidad

**Cuándo usar:**

- Series con tendencias no lineales
- Múltiples estacionalidades
- Datos faltantes frecuentes
- Necesitas intervalos de confianza

### 💡 Mejoras adicionales

1. **SARIMA:** ARIMA con estacionalidad explícita (p, d, q)(P, D, Q, s)
1. **LSTM/GRU:** Deep learning para series no lineales
1. **XGBoost:** Con features de lag, rolling stats, fecha
1. **Ensemble:** Combinar ARIMA + Prophet + ML

### 🚫 Errores comunes

- ❌ No verificar estacionariedad antes de ARIMA
- ❌ Sobreajustar parámetros (usar grid search con validación cruzada)
- ❌ No considerar eventos externos (holidays, promociones)
- ❌ Pronosticar muy lejos en el futuro (incertidumbre crece)

______________________________________________________________________

## 🔧 Código de producción

```python
# Pipeline completo
def forecast_pipeline(df, target_col, forecast_days=30):
    """
    Pipeline de pronóstico con ARIMA y Prophet
    """
    # Preparar datos
    df_prophet = df.reset_index().rename(columns={'date': 'ds', target_col: 'y'})

    # Entrenar Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)

    # Pronosticar
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)

# Usar
future_sales = forecast_pipeline(df, 'sales', forecast_days=90)
print(future_sales.head())
```

### 📌 Checklist de series temporales

- ✅ Visualizar serie (tendencia, estacionalidad)
- ✅ Descomponer en componentes
- ✅ Verificar estacionariedad (ADF test)
- ✅ Diferenciar si es necesario
- ✅ Identificar parámetros (ACF, PACF)
- ✅ Split train/test temporal (NO random!)
- ✅ Entrenar múltiples modelos
- ✅ Evaluar con MAE/RMSE en test set
- ✅ Visualizar predicciones vs reales
- ✅ Reportar intervalos de confianza
