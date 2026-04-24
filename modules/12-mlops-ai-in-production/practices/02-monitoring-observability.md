# Práctica 02 — Monitoring y Observability

## 🎯 Objetivos

- Setup Prometheus + Grafana
- Log predictions y métricas
- Alerting automático
- Performance monitoring

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Logging de Predicciones

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

    # Predicción
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

# Uso
prediction = predict_with_logging(model, [1.5, 2.3, 0.8], user_id='user123')
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Prometheus Metrics

**Enunciado:**
Expón métricas en `/metrics`:

```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total')
latency_histogram = Histogram('prediction_latency_seconds')

@latency_histogram.time()
def predict():
    prediction_counter.inc()
    # ...
```

### Ejercicio 2.2: Grafana Dashboard

**Enunciado:**
Crea dashboard con:

- Predictions per minute
- Latency percentiles (p50, p95, p99)
- Error rate
- Data drift metrics

### Ejercicio 2.3: Alerting Rules

**Enunciado:**
Configura alerts:

- Latency > 500ms por 5 min
- Error rate > 5%
- Data drift detected
- Envía a Slack/Email

### Ejercicio 2.4: Model Performance Tracking

**Enunciado:**
Monitorea accuracy en producción:

- Colecta ground truth labels (delayed)
- Calcula metrics diarios
- Plot evolución temporal
- Trigger retraining si baja

### Ejercicio 2.5: Cost Monitoring

**Enunciado:**
Track costos de inferencia:

- Requests por segundo
- Compute resources
- API costs (si LLM)
- Optimize para reducir

______________________________________________________________________

## ✅ Checklist

- [ ] Logging de predicciones
- [ ] Prometheus + Grafana
- [ ] Alerting automático
- [ ] Model performance tracking
- [ ] Cost monitoring

______________________________________________________________________

## 📚 Recursos

- [Prometheus Docs](https://prometheus.io/docs/)
- [Evidently AI](https://www.evidentlyai.com/)
