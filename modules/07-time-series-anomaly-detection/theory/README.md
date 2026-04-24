# Theory — Time Series & Anomaly Detection

## Why this module matters

Series temporales están en todas partes: precios de acciones, demanda de productos, tráfico web, sensores IoT, métricas de salud. Predecir el futuro y detectar anomalías son capacidades críticas para tomar decisiones proactivas y prevenir fallas en sistemas.

______________________________________________________________________

## 1. ¿Qué es una serie temporal?

**Serie temporal (time series):** Secuencia de datos **ordenados por tiempo** donde el orden importa porque valores futuros dependen de valores pasados.

**Ejemplos:**

- Precio de acciones por minuto.
- Temperatura diaria.
- Ventas mensuales.
- Trafico web por hora.
- Sensor de máquina cada segundo.

### Diferencia con datos tabulares

- **Temporal:** orden cronológico es esencial.
- **Tabular:** filas son independientes (generalmente).

📹 **Videos recomendados:**

1. [Time Series Forecasting - StatQuest](https://www.youtube.com/watch?v=w9D6mU4PmXA) - 15 min
1. [Time Series Analysis Crash Course](https://www.youtube.com/watch?v=e8Yw4alG16Q) - 1 hora

______________________________________________________________________

## 2. Componentes de una serie temporal

Descomponer serie temporal en componentes ayuda a entender estructura y elegir modelo.

### Tendencia (Trend)

Dirección general a largo plazo: creciente, decreciente o estable.

**Ejemplo:** Ventas de e-commerce crecen año tras año.

### Estacionalidad (Seasonality)

Patrones repetitivos con período fijo (día, semana, mes, año).

**Ejemplo:** Ventas de helado aumentan en verano cada año.

### Ciclos

Fluctuaciones de largo plazo sin período fijo (ej: ciclos económicos).

### Ruido (Residuales)

Variabilidad aleatoria que no se explica por tendencia, estacionalidad o ciclos.

### Descomposición

```
Y(t) = Tendencia + Estacionalidad + Residuales
```

**Tipos:**

- **Aditiva:** `Y = T + S + R` (cuando amplitud estacional es constante).
- **Multiplicativa:** `Y = T × S × R` (cuando amplitud estacional crece con tendencia).

📹 **Videos recomendados:**

1. [Time Series Decomposition - Ritvik Math](https://www.youtube.com/watch?v=SRhL80phcE8) - 10 min

📚 **Recursos escritos:**

- [Statsmodels Seasonal Decompose](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)

______________________________________________________________________

## 3. Preparación de datos temporales

### Paso 1: Ordenar por timestamp

```python
df = df.sort_values('timestamp')
```

### Paso 2: Verificar frecuencia temporal

- Diaria, horaria, cada 5 minutos?
- Uso: `df.set_index('timestamp').asfreq('D')` (pandas).

### Paso 3: Manejo de valores faltantes

**Opciones:**

- **Forward fill:** usar último valor conocido.
- **Backward fill:** usar siguiente valor.
- **Interpolación lineal:** estimar entre dos puntos.

**Cuidado:** No llenar con promedio global (rompe dinámica temporal).

### Paso 4: Feature engineering temporal

Crear features a partir del timestamp:

- Año, mes, día, hora, día de la semana.
- Es fin de semana (0/1).
- Es festivo (0/1).

### Paso 5: Lags y Rolling windows

- **Lags:** valores en instantes previos (ej: venta de hace 7 días).
- **Rolling statistics:** promedio/std móvil de ventana (ej: promedio 7 días).

**Ejemplo:**

```python
df['lag_7'] = df['sales'].shift(7)
df['rolling_mean_7'] = df['sales'].rolling(window=7).mean()
```

📹 **Videos recomendados:**

1. [Feature Engineering for Time Series - Kaggle](https://www.youtube.com/watch?v=OdaZP1Q_H7k) - 30 min

______________________________________________________________________

## 4. Train/Test split temporal

### 🚫 Error común: Split aleatorio

No mezclar pasado y futuro. Esto causa **data leakage temporal**.

### ✅ Split correcto: temporal

```python
train = df[df['timestamp'] < '2023-01-01']
test = df[df['timestamp'] >= '2023-01-01']
```

### Validación: Time Series Cross-Validation

Ventanas deslizantes que respetan orden temporal.

```
Fold 1: Train [1-100] | Validate [101-120]
Fold 2: Train [1-120] | Validate [121-140]
Fold 3: Train [1-140] | Validate [141-160]
```

📹 **Videos recomendados:**

1. [Time Series Cross-Validation - StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw) - 8 min

______________________________________________________________________

## 5. Modelos de forecasting

### Baselines simples (SIEMPRE empezar con esto)

#### Naive Forecast

Predecir que mañana será igual que hoy.

```python
y_pred = y_train[-1]
```

#### Seasonal Naive

Predecir usando mismo período anterior (ej: ventas de hoy = ventas del mismo día semana pasada).

#### Moving Average

Promedio de últimos `n` valores.

### ARIMA / SARIMA

**ARIMA:** AutoRegressive Integrated Moving Average.

- **AR (p):** autoregresión (depende de valores pasados).
- **I (d):** diferenciación (hacer serie estacionaria).
- **MA (q):** promedio móvil de errores.

**SARIMA:** ARIMA + estacionalidad.

**Uso:** Series univariadas con patrones claros.

📹 **Videos recomendados:**

1. [ARIMA Models - StatQuest](https://www.youtube.com/watch?v=3UmyHed0iYE) - 25 min
1. [SARIMA Explained - Ritvikmath](https://www.youtube.com/watch?v=UzN_lBYXVKI) - 15 min

### Prophet (Facebook)

Modelo aditivo generalizado diseñado para series con estacionalidad fuerte y feriados.

**Ventajas:**

- Fácil de usar.
- Robusto a faltantes.
- Maneja múltiples estacionalidades.

**Limitación:** No captura relaciones complejas.

📹 **Videos recomendados:**

1. [Prophet Tutorial - Data Science Garage](https://www.youtube.com/watch?v=pOYAXv15r3A) - 20 min

📚 **Recursos escritos:**

- [Prophet Docs](https://facebook.github.io/prophet/)

### Machine Learning con features temporales

Usar algoritmos clásicos (Random Forest, XGBoost) con features temporales como lags, rolling stats, features de fecha.

**Ventajas:**

- Maneja múltiples features exógenas.
- Captura no-linealidades.

**Cuidado:** Necesita buen feature engineering.

### Deep Learning: LSTM, GRU

Redes recurrentes para secuencias.

**Uso:** Series muy largas, múltiples series relacionadas.

**Desventaja:** Requiere muchos datos y tuning.

📹 **Videos recomendados:**

1. [LSTM for Time Series - StatQuest](https://www.youtube.com/watch?v=LfnrRPFhkuY) - 15 min
1. [Time Series with XGBoost - Kaggle](https://www.youtube.com/watch?v=vV12dGe_Fho) - 30 min

______________________________________________________________________

## 6. Métricas de forecasting

### MAE (Mean Absolute Error)

```
MAE = (1/n) Σ |y_true - y_pred|
```

- Fácil de interpretar (en mismas unidades que `y`).
- Trata todos los errores igual.

### RMSE (Root Mean Squared Error)

```
RMSE = √[(1/n) Σ (y_true - y_pred)²]
```

- Penaliza errores grandes más fuertemente.

### MAPE (Mean Absolute Percentage Error)

```
MAPE = (100/n) Σ |(y_true - y_pred) / y_true|
```

- Expresa error como %.
- **Cuidado:** no funciona si `y_true` tiene ceros.

### Elección de métrica

- **MAE:** cuando todos los errores importan igual.
- **RMSE:** cuando errores grandes son muy costosos.
- **MAPE:** cuando quieres error relativo (% de error).

**Recomendación:** Usar múltiples métricas + visualización de predicciones.

📹 **Videos recomendados:**

1. [Forecasting Metrics - Krish Naik](https://www.youtube.com/watch?v=SFBjApWPMaE) - 15 min

______________________________________________________________________

## 7. Detección de anomalías

### ¿Qué es una anomalía?

Punto de datos que se desvía significativamente del patrón normal.

**Aplicaciones:**

- Detección de fraudes.
- Fallas en sensores/equipos.
- Intrusiones en redes.
- Picos anormales de tráfico.

### Enfoques

#### 1. Umbrales estáticos

```python
if value > threshold:
    flag_as_anomaly()
```

**Limitación:** No adapta a cambios de patrón.

#### 2. Umbrales dinámicos (basados en estadística)

Z-score sobre residuales:

```python
residuals = y_true - y_pred
z_score = (residuals - mean) / std
if abs(z_score) > 3:
    flag_as_anomaly()
```

#### 3. Métodos no supervisados

**Isolation Forest:**

- Aislamiento de puntos atípicos mediante árboles.
- No requiere etiquetas.

**Autoencoders:**

- Red neuronal que aprende a reconstruir datos normales.
- Error de reconstrucción alto = anomalía.

**DBSCAN Clustering:**

- Puntos que no pertenecen a ningún cluster = anomalías.

📹 **Videos recomendados:**

1. [Anomaly Detection Explained - StatQuest](https://www.youtube.com/watch?v=L0jA7LGSQqI) - 12 min
1. [Isolation Forest - Krish Naik](https://www.youtube.com/watch?v=5p8B2Ikcw-k) - 20 min
1. [Autoencoders for Anomaly Detection](https://www.youtube.com/watch?v=2K3ScZp1dXQ) - 25 min

📚 **Recursos escritos:**

- [Scikit-learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [PyOD Library](https://pyod.readthedocs.io/) - biblioteca especializada

### Métricas para anomalía

- **Precision:** de las que detecté como anomalías, ¿cuántas lo son realmente?
- **Recall:** de las anomalías reales, ¿cuántas detecté?
- **F1:** balance.

**Trade-off:**

- Umbral bajo: más recall, más falsos positivos.
- Umbral alto: más precision, menos recall.

**Contexto de negocio determina balance.**

______________________________________________________________________

## 8. Buenas prácticas

- ✅ Siempre empezar con baseline simple (naive, seasonal naive).
- ✅ Usar split temporal (no aleatorio).
- ✅ Validar con time series cross-validation.
- ✅ Visualizar predicciones junto con datos reales.
- ✅ Monitorear drift: patrones cambian con el tiempo.
- ✅ Re-entrenar modelos periódicamente con datos recientes.
- ✅ Documentar frecuencia, unidades y transformaciones aplicadas.
- ✅ Para anomalías, evaluar en producción (feedback loop).

📚 **Recursos generales:**

- [Forecasting: Principles and Practice (Book - Free)](https://otexts.com/fpp3/)
- [Statsmodels Time Series Guide](https://www.statsmodels.org/stable/tsa.html)
- [Kaggle Time Series Course](https://www.kaggle.com/learn/time-series)

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente módulo, deberías poder:

- ✅ Identificar tendencia, estacionalidad y ruido en un gráfico.
- ✅ Explicar por qué split aleatorio es incorrecto en series temporales.
- ✅ Construir features temporales (lags, rolling stats, features de fecha).
- ✅ Implementar baseline naive y comparar con modelos complejos.
- ✅ Elegir métrica de forecasting apropiada (MAE, RMSE, MAPE).
- ✅ Detectar anomalías usando umbrales dinámicos o Isolation Forest.
- ✅ Justificar trade-off precision/recall según contexto de negocio.
- ✅ Implementar time series cross-validation con scikit-learn.

Si respondiste "sí" a todas, estás listo para aplicaciones avanzadas de series temporales.
