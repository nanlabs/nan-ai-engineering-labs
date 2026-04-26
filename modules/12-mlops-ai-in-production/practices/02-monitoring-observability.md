# Practice 02 — Monitoring and Observability

## 🎯 Objectives

- Setup Prometheus + Grafana
- Log predictions and Metrics
- Automatic alerting
- Performance monitoring

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Logging Predictions

```python
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def predict_with_logging(model, features, user_id):
    start_time = time.time()

    # Prediction
    prediction = model.predict([features])[0]

    # Latencia
    latency = time.time() - start_time

    # Log
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'features': features,
        'prediction': prediction,
        'latency_ms': latency * 1000
    }

    logging.info(log_data)

    return prediction

# Usage
prediction = predict_with_logging(model, [1.5, 2.3, 0.8], user_id='user123')
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Propuestos

### Exercise 2.1: Prometheus Metrics

**Statement:**
Expose Metrics in `/metrics`:

```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total')
latency_histogram = Histogram('prediction_latency_seconds')

@latency_histogram.time()
def predict():
    prediction_counter.inc()
    # ...
```

### Exercise 2.2: Grafana Dashboard

**Statement:**
Create dashboard with:

- Predictions per minute
- Latency percentiles (p50, p95, p99)
- error rate
- Data drift metrics

### Exercise 2.3: Alerting Rules

**Statement:**
Configura alerts:

- Latency > 500ms for 5 min
- error rate > 5%
- Data drift detected
- Send to Slack/Email

### Exercise 2.4: Model Performance Tracking

**Statement:**
Monitor accuracy in production:

- Colecta ground truth labels (delayed)
- Calculate metrics diarios
- Time evolution plot
- Trigger retraining si baja

### Exercise 2.5: Cost Monitoring

**Statement:**
Track inference costs:

- Requests per second
- Compute resources
- API costs (si LLM)
- Optimize to reduce

______________________________________________________________________

## ✅ Checklist

- [ ] Predictions Logging
- [ ] Prometheus + Grafana
- [ ] Automatic alerting
- [ ] Model performance tracking
- [ ] Cost monitoring

______________________________________________________________________

## 📚 Resources

- [Prometheus Docs](https://prometheus.io/docs/)
- [Evidently AI](https://www.evidentlyai.com/)
