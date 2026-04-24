# Theory — MLOps & AI in Production

## Why this module matters

Solo el 13% de los Models de ML llegan a producción (según Gartner). MLOps cierra la brecha entre experimentos en notebooks y sistemas confiables en producción. Dominar MLOps te posiciona como ingeniero completo, no solo data scientist.

______________________________________________________________________

## 1. ¿Qué es MLOps?

**MLOps (Machine Learning Operations):** Conjunto de Practices, herramientas y cultura para **construir, desplegar y operar** Models de ML de forma **confiable, reproducible y escalable**.

### Analogía con DevOps

- **DevOps:** Automatización de desarrollo y operaciones de software.
- **MLOps:** DevOps + desafíos únicos de ML (Data, Models, drift).

### ¿Por qué MLOps es diferente?

- **Código es solo parte:** Data y Models también cambian.
- **Degradación silenciosa:** Model pierde accuracy sin Errors técnicos.
- **Experimentación intensiva:** Muchos experimentos, pocos van a producción.

📹 **Videos recomendados:**

1. [MLOps Explained - Google Cloud](https://www.youtube.com/watch?v=6gdrwFMaEZ0) - 10 min
1. [MLOps: From Model to Production - Andrew Ng](https://www.youtube.com/watch?v=06-AZXmwHjo) - 1 hora

______________________________________________________________________

## 2. Ciclo de vida de ML en producción

### Fase 1: Desarrollo y experimentación

**Actividades:**

- EDA (Exploratory Data Analysis).
- Feature engineering.
- Training de Models.
- Comparación de Algorithms.

**Herramientas:**

- Jupyter Notebooks, Google Colab.
- MLflow, Weights & Biases para tracking.

### Fase 2: Training y Validation

**Actividades:**

- Entrenar Model final con Data completos.
- Validation rigurosa (cross-validation, test set).
- Tuning de Hyperparameters.

**Herramientas:**

- Kubeflow, SageMaker, Vertex AI.
- Optuna, Ray Tune para hyperparameter tuning.

### Fase 3: Registro de Model (Model Registry)

**Actividades:**

- Guardar Model entrenado.
- Versionar Model.
- Documentar metadata (Metrics, features, Hyperparameters).

**Herramientas:**

- MLflow Model Registry.
- SageMaker Model Registry.
- Vertex AI Model Registry.

### Fase 4: Despliegue (Deployment)

**Actividades:**

- Empaquetar Model (Docker, ONNX).
- Exponer API de Prediction.
- Configurar infraestructura (CPU/GPU, scaling).

**Herramientas:**

- FastAPI, Flask para APIs.
- Docker, Kubernetes.
- Cloud services: SageMaker, Vertex AI, Azure ML.

### Fase 5: Monitoreo y retraining

**Actividades:**

- Monitorear performance (latencia, accuracy).
- Detectar drift.
- Reentrenar Model con Data recientes.

**Herramientas:**

- Prometheus, Grafana.
- Evidently AI, Fiddler para drift detection.

📹 **Videos recomendados:**

1. [ML Lifecycle - Full Stack Deep Learning](https://www.youtube.com/watch?v=pvaIi0l1GME) - 1 hora

______________________________________________________________________

## 3. Componentes clave de MLOps

### Versionado

#### Versionado de código

- **Git:** Estándar para código fuente.

#### Versionado de Data

Data cambian, necesitas trackear versiones.

**Herramientas:**

- **DVC (Data Version Control):** Git para Data.
- **LakeFS:** Versionado de data lakes.
- **Delta Lake:** Versionado transaccional.

#### Versionado de Models

Guardar cada Model entrenado con metadata.

**Metadata:**

- Versión de Data usados.
- Hyperparameters.
- Metrics de Validation.
- Fecha de Training.
- Autor.

**Herramientas:**

- MLflow Model Registry.
- Weights & Biases.

### CI/CD para ML

**CI (Continuous Integration):**

- Tests automatizados:
  - Unit tests para código.
  - Data validation (schema, distribuciones).
  - Model validation (Metrics mínimas).

**CD (Continuous Deployment):**

- Despliegue automatizado a staging.
- Validation en staging.
- Despliegue a producción (manual o automático).

**Herramientas:**

- GitHub Actions, GitLab CI.
- Jenkins, CircleCI.

### Feature Store

**Problem:** Features calculadas múltiples veces de forma inconsistente.

**Solución:** Repositorio centralizado de features.

**Beneficios:**

- Reutilización de features.
- Consistencia entre Training y serving.
- Features pre-computadas (latencia baja).

**Herramientas:**

- Feast (open-source).
- Tecton.
- SageMaker Feature Store.

📹 **Videos recomendados:**

1. [Feature Stores Explained - Tecton](https://www.youtube.com/watch?v=2m2LqZfKqKI) - 15 min
1. [CI/CD for ML - Google Cloud](https://www.youtube.com/watch?v=hFhZsDgZFfg) - 20 min

📚 **Resources escritos:**

- [DVC Documentation](https://dvc.org/doc)
- [Feast Documentation](https://docs.feast.dev/)

______________________________________________________________________

## 4. Serving: Cómo servir Predictions

### Batch Prediction

**Usage:** Predictions programadas sobre lotes de Data.

**Example:** Recomendar productos semanalmente a todos los usuarios.

**Ventaja:** Simple, tolerante a latencia alta.

### Online Prediction (Real-time)

**Usage:** Predictions en tiempo real por request.

**Example:** Detectar fraude al momento de transacción.

**Requerimiento:** Latencia baja (\<100ms típicamente).

**Herramientas:**

- FastAPI, Flask.
- TensorFlow Serving, TorchServe.
- Seldon Core, KServe.

### Streaming Prediction

**Usage:** Predictions sobre flujos continuos de Data.

**Example:** Detección de Anomalies en sensores IoT.

**Herramientas:**

- Kafka, Kinesis.
- Spark Streaming, Flink.

### Elección

| Modo          | Latencia         | Costo | Complejidad |
| ------------- | ---------------- | ----- | ----------- |
| **Batch**     | Alta (horas)     | Bajo  | Baja        |
| **Online**    | Baja (ms)        | Alto  | Media       |
| **Streaming** | Media (segundos) | Alto  | Alta        |

📹 **Videos recomendados:**

1. [Model Serving Patterns - Chip Huyen](https://www.youtube.com/watch?v=KdmcFqbMSEM) - 30 min

______________________________________________________________________

## 5. Monitoreo en producción

### Metrics operacionales

#### Latencia

Tiempo de respuesta de Prediction.

**Target:** Depende del caso (p95 \<100ms para real-time).

#### Throughput

Predictions por segundo.

#### Uptime / Availability

% de tiempo que el servicio está disponible.

**Target:** 99.9% ("three nines") o más.

#### Costo

Costo computacional (CPU/GPU, transferencia de Data).

### Metrics de Model

#### accuracy / Performance

¿El Model sigue teniendo buen performance?

**Problem:** En producción no siempre tienes ground truth inmediatamente.

**Solución:**

- Proxy metrics (CTR, conversion rate).
- Etiquetas retrasadas (delayed labels).

#### Data Drift

**Definición:** Distribución de entradas cambia.

**Example:** Model entrenado en 2020, en 2024 comportamiento de usuarios cambió.

**Detección:**

- Kolmogorov-Smirnov test.
- Population Stability Index (PSI).

#### Concept Drift

**Definición:** Relación entre entradas y target cambia.

**Example:** Antes COVID, después COVID (patrones de compra cambiaron radicalmente).

**Detección:**

- Monitorear accuracy en subset reciente.
- Comparar distribuciones de Predictions.

#### Prediction Drift

Distribución de Predictions cambia.

**Example:** Model predice "fraude" para 1% de transacciones, ahora predice 10%.

**Alerta:** Posible cambio en Data o falla en Model.

📹 **Videos recomendados:**

1. [ML Monitoring - Chip Huyen](https://www.youtube.com/watch?v=TYk8wOo48xs) - 40 min
1. [Data Drift Explained - Evidently AI](https://www.youtube.com/watch?v=bKh6CTgHiLU) - 15 min

📚 **Resources escritos:**

- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [Google ML Monitoring Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Alertas

**Configurar alertas cuando:**

- Latencia > umbral.
- error rate > umbral.
- Data drift detectado.
- accuracy cae > X%.

**Herramientas:**

- Prometheus + Alertmanager.
- PagerDuty, Opsgenie.

______________________________________________________________________

## 6. Estrategias de despliegue

### Blue/Green Deployment

**Concept:** Mantener dos ambientes (blue/green). Cambiar tráfico instantáneamente.

**Ventaja:** Rollback instantáneo.
**Desventaja:** Duplica infraestructura (costoso).

### Canary Deployment

**Concept:** Desplegar nuevo Model a pequeño % de tráfico. Incrementar gradualmente.

**Example:**

```
V1: 95% tráfico
V2 (canary): 5% tráfico
→ Monitorear → Si OK, incrementar a 50% → 100%
```

**Ventaja:** Riesgo limitado.

### Shadow Deployment

**Concept:** Nuevo Model recibe tráfico pero Predictions NO se usan (solo logging).

**Ventaja:** Validar en producción sin impacto.

### A/B Testing

**Concept:** Dividir usuarios aleatoriamente entre Models.

**Objective:** Medir impacto en Metric de negocio (ej: revenue, engagement).

**Importante:** Requiere significancia estadística.

📹 **Videos recomendados:**

1. [Deployment Strategies - DevOps Toolkit](https://www.youtube.com/watch?v=AWVTKBUnoIg) - 20 min

______________________________________________________________________

## 7. Retraining: ¿Cuándo y cómo?

### ¿Cuándo reentrenar?

1. **Performance degradation:** accuracy cayó.
1. **Data drift detectado:** Distribución cambió.
1. **Scheduled:** Periódicamente (semanal, mensual).
1. **Nuevos Data disponibles:** Dataset creció significativamente.

### Estrategias de retraining

#### Retraining desde cero

Entrenar completamente con todos los Data.

**Ventaja:** Model fresco.
**Desventaja:** Costoso.

#### Incremental learning

Actualizar Model existente con Data nuevos.

**Ventaja:** Rápido.
**Desventaja:** No todos los Models lo soportan.

### Pipeline automatizado

1. Detectar trigger (drift, schedule).
1. Recolectar Data recientes.
1. Entrenar Model.
1. Validar (Metrics > umbral).
1. Registrar en model registry.
1. Desplegar con estrategia segura (canary).
1. Monitorear.

______________________________________________________________________

## 8. Herramientas y plataformas

### End-to-End Platforms

- **AWS SageMaker:** Completo, integrado con AWS.
- **Google Vertex AI:** Integrado con GCP.
- **Azure ML:** Integrado con Azure.
- **Databricks:** Unificado para Data + ML.

### Open-Source Tools

- **MLflow:** Tracking, registry, serving.
- **Kubeflow:** Pipelines de ML en Kubernetes.
- **Metaflow (Netflix):** Workflow management.
- **DVC:** Versionado de Data.
- **Weights & Biases:** Experiment tracking.

### Orchestration

- **Airflow:** Workflow orchestration.
- **Prefect:** Alternativa moderna a Airflow.
- **Argo Workflows:** Para Kubernetes.

📹 **Videos recomendados:**

1. [MLOps Tools Landscape - Full Stack Deep Learning](https://www.youtube.com/watch?v=d1ZAFxUPC8A) - 30 min

______________________________________________________________________

## 9. Buenas Practices

- ✅ Empezar simple: No construir infraestructura compleja desde día 1.
- ✅ Automatizar gradualmente: A medida que el sistema madura.
- ✅ Definir SLOs (Service Level Objectives) desde el inicio: latencia, accuracy, uptime.
- ✅ Mantener trazabilidad completa: código, Data, Model, Predictions.
- ✅ Implementar CI/CD para ML: tests automatizados, despliegue controlado.
- ✅ Monitorear más que accuracy: drift, latencia, costo.
- ✅ Usar feature store si reutilizas features.
- ✅ Desplegar con estrategia segura: canary, A/B testing.
- ✅ Diseñar runbooks para incidentes: qué hacer si accuracy cae.
- ✅ Documentar decisiones de arquitectura.

📚 **Resources generales:**

- [MLOps Principles - Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Full Stack Deep Learning (Course)](https://fullstackdeeplearning.com/)
- [Made With ML (Course)](https://madewithml.com/)
- [Awesome MLOps (GitHub)](https://github.com/visenger/awesome-mlops)

______________________________________________________________________

## Final comprehension checklist

Antes de completar tu formación, deberías poder:

- ✅ Describir flujo completo desde experimento a producción.
- ✅ Versionar Data, código y Models con DVC/MLflow.
- ✅ Implementar CI/CD pipeline para Model.
- ✅ Servir Model con API (FastAPI/Flask).
- ✅ Elegir estrategia de serving (batch, online, streaming) según caso.
- ✅ Monitorear latencia, accuracy, y drift en producción.
- ✅ Detectar data drift y concept drift.
- ✅ Implementar canary deployment para reducir riesgo.
- ✅ Diseñar pipeline automatizado de retraining.
- ✅ Definir SLOs y alertas para Model en producción.

Si respondiste "sí" a todas, ¡estás listo para llevar Models de ML a producción de forma profesional! 🎉
