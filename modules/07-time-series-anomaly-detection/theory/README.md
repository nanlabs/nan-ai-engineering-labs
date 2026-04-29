# Theory — Time Series & Anomaly Detection

## Why this module matters

time series are everywhere: stock prices, product demand, web traffic, IoT sensors, health metrics. Predict the future and detect Anomalies are critical capabilities to make proactive decisions and prevent system failures.

______________________________________________________________________

## 1. What is a Time series?

**Time series (time series):** Sequence of Data **ordered by time** where the order matters because future values ​​depend on past values.

**Examples:**

- Stock price per minute.
- Daily temperature.
- Monthly sales.
- Web traffic per hour.
- Machine sensor every second.

### Difference with tabular data

- **Temporal:** Chronological order is essential.
- **Tabular:** rows are independent (usually).

📹 **Videos recommended:**

1. [Time Series Forecasting - StatQuest](https://www.youtube.com/watch?v=w9D6mU4PmXA) - 15 min
1. [Time Series Analysis Crash Course](https://www.youtube.com/watch?v=e8Yw4alG16Q) - 1 hora

______________________________________________________________________

## 2. Components of a Time series

Decomposing Time series into components helps to understand Structure and choose Model.

### Trend (Trend)

General long-term direction: increasing, decreasing or stable.

**Example:** E-commerce sales grow year after year.

### Seasonality (Seasonality)

Repetitive patterns with fixed period (day, week, month, year).

**Example:** Ice cream sales increase in summer every year.

### Cycles

Long-term fluctuations without a fixed period (e.g. economic cycles).

### Noise (Residual)

Random variability that is not explained by Trend, Seasonality or cycles.

### Decomposition

```
Y(t) = Trend + Seasonality + Residuals
```

**Types:**

- **Additive:** `Y = T + S + R` (when seasonal amplitude is constant).
- **Multiplicative:** `Y = T × S × R` (when seasonal amplitude increases with Trend).

📹 **Videos recommended:**

1. [Time Series Decomposition - Ritvik Math](https://www.youtube.com/watch?v=SRhL80phcE8) - 10 min

📚 **Resources written:**

- [Statsmodels Seasonal Decompose](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)

______________________________________________________________________

## 3. Temporary Data Preparation

### Step 1: Sort by timestamp

```python
df = df.sort_values('timestamp')
```

### Step 2: Check time frequency

- Daily, hourly, every 5 minutes?
- Usage: `df.set_index('timestamp').asfreq('D')` (pandas).

### Step 3: Handling missing values

**Options:**

- **Forward fill:** use last known value.
- **Backward fill:** use next valor.
- **Linear interpolation:** estimate between two points.

**Caution:** Do not fill with global average (breaks temporal dynamics).

### Paso 4: Feature engineering temporal

Create features from the timestamp:

- Year, month, day, hour, day of the week.
- It's the end of the week (0/1).
- It's a holiday (0/1).

### Step 5: Lags and Rolling windows

- **Lags:** values ​​in previous moments (e.g. sale from 7 days ago).
- **Rolling statistics:** average/std rolling window (e.g. average 7 days).

**Example:**

```python
df['lag_7'] = df['sales'].shift(7)
df['rolling_mean_7'] = df['sales'].rolling(window=7).mean()
```

📹 **Videos recommended:**

1. [Feature Engineering for Time Series - Kaggle](https://www.youtube.com/watch?v=OdaZP1Q_H7k) - 30 min

______________________________________________________________________

## 4. Train/Test split temporal

### 🚫 common mistake: Random Split

Do not mix past and future. This causes **temporary data leakage**.

### ✅ Correct split: temporary

```python
train = df[df['timestamp'] < '2023-01-01']
test = df[df['timestamp'] >= '2023-01-01']
```

### Validation: Time Series Cross-Validation

Sliding windows that respect temporal order.

```
Fold 1: Train [1-100] | Validate [101-120]
Fold 2: Train [1-120] | Validate [121-140]
Fold 3: Train [1-140] | Validate [141-160]
```

📹 **Videos recommended:**

1. [Time Series Cross-Validation - StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw) - 8 min

______________________________________________________________________

## 5. Forecasting models

### Simple baselines (ALWAYS start with this)

#### Naive Forecast

I predict that tomorrow will be the same as today.

```python
y_pred = y_train[-1]
```

#### Seasonal Naive

Predict using same previous period (e.g. today's sales = same day's sales last week).

#### Moving Average

Average of last `n` values.

### ARIMA / SARIMA

**ARIMA:** AutoRegressive Integrated Moving Average.

- **AR (p):** auto-regression (depends on passed values).
- **I (d):** differentiation (make stationary series).
- **MA (q):** moving average of Errors.

**SARIMA:** ARIMA + Seasonality.

**Usage:** Univariate series with clear patterns.

📹 **Videos recommended:**

1. [ARIMA Models - StatQuest](https://www.youtube.com/watch?v=3UmyHed0iYE) - 25 min
1. [SARIMA Explained - Ritvikmath](https://www.youtube.com/watch?v=UzN_lBYXVKI) - 15 min

### Prophet (Facebook)

Generalized additive model designed for series with strong Seasonality and holidays.

**Advantages:**

- Easy to use.
- Robust to missingness.
- Handles multiple seasonalities.

**Limitation:** Does not capture complex relationships.

📹 **Videos recommended:**

1. [Prophet Tutorial - Data Science Garage](https://www.youtube.com/watch?v=pOYAXv15r3A) - 20 min

📚 **Resources written:**

- [Prophet Docs](https://facebook.github.io/prophet/)

### Machine Learning with temporary features

Use classic algorithms (Random Forest, XGBoost) with temporal features such as lags, rolling stats, date features.

**Advantages:**

- Handles multiple exogenous features.
- Capture non-linearities.

**Caution:** Needs good feature engineering.

### Deep Learning: LSTM, GRU

Recurrent networks for sequences.

**Usage:** Very long series, multiple related series.

**Disadvantage:** Requires a lot of data and tuning.

📹 **Videos recommended:**

1. [LSTM for Time Series - StatQuest](https://www.youtube.com/watch?v=LfnrRPFhkuY) - 15 min
1. [Time Series with XGBoost - Kaggle](https://www.youtube.com/watch?v=vV12dGe_Fho) - 30 min

______________________________________________________________________

## 6. Forecasting metrics

### MAE (Mean Absolute error)

```
MAE = (1/n) Σ |y_true - y_pred|
```

- Easy to interpret (in the same units as `y`).
- Treat all Errors the same.

### RMSE (Root Mean Squared error)

```
RMSE = √[(1/n) Σ (y_true - y_pred)²]
```

- Penalizes large errors more strongly.

### MAP (Mean Absolute Percentage error)

```
MAP = (100/n) Σ |(y_true - y_pred) / y_true|
```

- Expresses error as %.
- **Caution:** does not work if `y_true` has zeroes.

### Metric Choice

- **MAE:** when all errors are important the same.
- **RMSE:** when large errors are very costly.
- **MAP:** when you want relative error (% error).

**Recommendation:** Use multiple Metrics + Visualization of Predictions.

📹 **Videos recommended:**

1. [Forecasting Metrics - Krish Naik](https://www.youtube.com/watch?v=SFBjApWPMaE) - 15 min

______________________________________________________________________

## 7. Anomaly Detection

### What is an Anomaly?

Data point that deviates significantly from the normal pattern.

**Applications:**

- Fraud detection.
- Failures in sensors/equipment.
- Network intrusions.
- Abnormal traffic spikes.

### Approaches

#### 1. Static thresholds

```python
if value > threshold:
    flag_as_anomaly()
```

**Limitation:** Does not adapt to pattern changes.

#### 2. Dynamic thresholds (based on statistics)

Z-score about residuals:

```python
residuals = y_true - y_pred
z_score = (residuals - mean) / std
if abs(z_score) > 3:
    flag_as_anomaly()
```

#### 3. Unsupervised methods

**Isolation Forest:**

- Isolation of atypical points using trees.
- Does not require labels.

**Autoencoders:**

- Neural network that learns to reconstruct normal Data.
- high reconstruction error = Anomaly.

**DBSCAN Clustering:**

- Points that do not belong to any cluster = Anomalies.

📹 **Videos recommended:**

1. [Anomaly Detection Explained - StatQuest](https://www.youtube.com/watch?v=L0jA7LGSQqI) - 12 min
1. [Isolation Forest - Krish Naik](https://www.youtube.com/watch?v=5p8B2Ikcw-k) - 20 min
1. [Autoencoders for Anomaly Detection](https://www.youtube.com/watch?v=2K3ScZp1dXQ) - 25 min

📚 **Resources written:**

- [Scikit-learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [PyOD Library](https://pyod.readthedocs.io/) - specialized library

### Metrics for Anomaly

- **Precision:** Of those that I detected as Anomalies, how many really are?
- **recall:** of the real Anomalies, how many did I detect?
- **f1:** balance.

**Trade-off:**

- Low threshold: more recall, more false positives.
- High threshold: more precision, less recall.

**Business context determines balance.**

______________________________________________________________________

## 8. Buenas Practices

- ✅ Always start with a simple baseline (naive, seasonal naive).
- ✅ Use temporary split (not random).
- ✅ Validate yourself with time series cross-validation.
- ✅ Visualize Predictions along with real Data.
- ✅ Monitor drift: patterns change over time.
- ✅ Re-train Models periodically with recent Data.
- ✅ Document frequency, units and applied transformations.
- ✅ For Anomalies, evaluate in production (feedback loop).

📚 **General resources:**

- [Forecasting: Principles and Practice (Book - Free)](https://otexts.com/fpp3/)
- [Statsmodels Time Series Guide](https://www.statsmodels.org/stable/tsa.html)
- [Kaggle Time Series Course](https://www.kaggle.com/learn/time-series)

______________________________________________________________________

## Final comprehension checklist

Before moving to the next Module, you should be able to:

- ✅ Identify Trend, Seasonality and noise in a graph.
- ✅ Explain why random split is incorrect in time series.
- ✅ Build temporary features (lags, rolling stats, date features).
- ✅ Implement naive baseline and compare with complex Models.
- ✅ Choose appropriate forecasting metrics (MAE, RMSE, MAP).
- ✅ Detect Anomalies using dynamic thresholds or Isolation Forest.
- ✅ Justify trade-off precision/recall according to business context.
- ✅ Implement time series cross-validation with scikit-learn.

If you answered "yes" to all, you are ready for advanced time series applications.
