# Theory — MLOps & AI in Production

## Why this module matters

Solo el 13% de los modelos de ML llegan a producción (según Gartner). MLOps cierra la brecha entre experimentos en notebooks y sistemas confiables en producción. Dominar MLOps te posiciona como ingeniero completo, no solo data scientist.

______________________________________________________________________

## 1. ¿Qué es MLOps?

**MLOps (Machine Learning Operations):** Conjunto de prácticas, herramientas y cultura para **construir, desplegar y operar** modelos de ML de forma **confiable, reproducible y escalable**.

### Analogía con DevOps

- **DevOps:** Automatización de desarrollo y operaciones de software.
- **MLOps:** DevOps + desafíos únicos de ML (datos, modelos, drift).

### ¿Por qué MLOps es diferente?

- **Código es solo parte:** Datos y modelos también cambian.
- **Degradación silenciosa:** Modelo pierde accuracy sin errores técnicos.
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
- Entrenamiento de modelos.
- Comparación de algoritmos.

**Herramientas:**

- Jupyter Notebooks, Google Colab.
- MLflow, Weights & Biases para tracking.

### Fase 2: Entrenamiento y validación

**Actividades:**

- Entrenar modelo final con datos completos.
- Validación rigurosa (cross-validation, test set).
- Tuning de hiperparámetros.

**Herramientas:**

- Kubeflow, SageMaker, Vertex AI.
- Optuna, Ray Tune para hyperparameter tuning.

### Fase 3: Registro de modelo (Model Registry)

**Actividades:**

- Guardar modelo entrenado.
- Versionar modelo.
- Documentar metadata (métricas, features, hiperparámetros).

**Herramientas:**

- MLflow Model Registry.
- SageMaker Model Registry.
- Vertex AI Model Registry.

### Fase 4: Despliegue (Deployment)

**Actividades:**

- Empaquetar modelo (Docker, ONNX).
- Exponer API de predicción.
- Configurar infraestructura (CPU/GPU, scaling).

**Herramientas:**

- FastAPI, Flask para APIs.
- Docker, Kubernetes.
- Cloud services: SageMaker, Vertex AI, Azure ML.

### Fase 5: Monitoreo y retraining

**Actividades:**

- Monitorear performance (latencia, accuracy).
- Detectar drift.
- Reentrenar modelo con datos recientes.

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

#### Versionado de datos

Datos cambian, necesitas trackear versiones.

**Herramientas:**

- **DVC (Data Version Control):** Git para datos.
- **LakeFS:** Versionado de data lakes.
- **Delta Lake:** Versionado transaccional.

#### Versionado de modelos

Guardar cada modelo entrenado con metadata.

**Metadata:**

- Versión de datos usados.
- Hiperparámetros.
- Métricas de validación.
- Fecha de entrenamiento.
- Autor.

**Herramientas:**

- MLflow Model Registry.
- Weights & Biases.

### CI/CD para ML

**CI (Continuous Integration):**

- Tests automatizados:
  - Unit tests para código.
  - Data validation (schema, distribuciones).
  - Model validation (métricas mínimas).

**CD (Continuous Deployment):**

- Despliegue automatizado a staging.
- Validación en staging.
- Despliegue a producción (manual o automático).

**Herramientas:**

- GitHub Actions, GitLab CI.
- Jenkins, CircleCI.

### Feature Store

**Problema:** Features calculadas múltiples veces de forma inconsistente.

**Solución:** Repositorio centralizado de features.

**Beneficios:**

- Reutilización de features.
- Consistencia entre entrenamiento y serving.
- Features pre-computadas (latencia baja).

**Herramientas:**

- Feast (open-source).
- Tecton.
- SageMaker Feature Store.

📹 **Videos recomendados:**

1. [Feature Stores Explained - Tecton](https://www.youtube.com/watch?v=2m2LqZfKqKI) - 15 min
1. [CI/CD for ML - Google Cloud](https://www.youtube.com/watch?v=hFhZsDgZFfg) - 20 min

📚 **Recursos escritos:**

- [DVC Documentation](https://dvc.org/doc)
- [Feast Documentation](https://docs.feast.dev/)

______________________________________________________________________

## 4. Serving: Cómo servir predicciones

### Batch Prediction

**Uso:** Predicciones programadas sobre lotes de datos.

**Ejemplo:** Recomendar productos semanalmente a todos los usuarios.

**Ventaja:** Simple, tolerante a latencia alta.

### Online Prediction (Real-time)

**Uso:** Predicciones en tiempo real por request.

**Ejemplo:** Detectar fraude al momento de transacción.

**Requerimiento:** Latencia baja (\<100ms típicamente).

**Herramientas:**

- FastAPI, Flask.
- TensorFlow Serving, TorchServe.
- Seldon Core, KServe.

### Streaming Prediction

**Uso:** Predicciones sobre flujos continuos de datos.

**Ejemplo:** Detección de anomalías en sensores IoT.

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

### Métricas operacionales

#### Latencia

Tiempo de respuesta de predicción.

**Target:** Depende del caso (p95 \<100ms para real-time).

#### Throughput

Predicciones por segundo.

#### Uptime / Availability

% de tiempo que el servicio está disponible.

**Target:** 99.9% ("three nines") o más.

#### Costo

Costo computacional (CPU/GPU, transferencia de datos).

### Métricas de modelo

#### Accuracy / Performance

¿El modelo sigue teniendo buen performance?

**Problema:** En producción no siempre tienes ground truth inmediatamente.

**Solución:**

- Proxy metrics (CTR, conversion rate).
- Etiquetas retrasadas (delayed labels).

#### Data Drift

**Definición:** Distribución de entradas cambia.

**Ejemplo:** Modelo entrenado en 2020, en 2024 comportamiento de usuarios cambió.

**Detección:**

- Kolmogorov-Smirnov test.
- Population Stability Index (PSI).

#### Concept Drift

**Definición:** Relación entre entradas y target cambia.

**Ejemplo:** Antes COVID, después COVID (patrones de compra cambiaron radicalmente).

**Detección:**

- Monitorear accuracy en subset reciente.
- Comparar distribuciones de predicciones.

#### Prediction Drift

Distribución de predicciones cambia.

**Ejemplo:** Modelo predice "fraude" para 1% de transacciones, ahora predice 10%.

**Alerta:** Posible cambio en datos o falla en modelo.

📹 **Videos recomendados:**

1. [ML Monitoring - Chip Huyen](https://www.youtube.com/watch?v=TYk8wOo48xs) - 40 min
1. [Data Drift Explained - Evidently AI](https://www.youtube.com/watch?v=bKh6CTgHiLU) - 15 min

📚 **Recursos escritos:**

- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [Google ML Monitoring Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Alertas

**Configurar alertas cuando:**

- Latencia > umbral.
- Error rate > umbral.
- Data drift detectado.
- Accuracy cae > X%.

**Herramientas:**

- Prometheus + Alertmanager.
- PagerDuty, Opsgenie.

______________________________________________________________________

## 6. Estrategias de despliegue

### Blue/Green Deployment

**Concepto:** Mantener dos ambientes (blue/green). Cambiar tráfico instantáneamente.

**Ventaja:** Rollback instantáneo.
**Desventaja:** Duplica infraestructura (costoso).

### Canary Deployment

**Concepto:** Desplegar nuevo modelo a pequeño % de tráfico. Incrementar gradualmente.

**Ejemplo:**

```
V1: 95% tráfico
V2 (canary): 5% tráfico
→ Monitorear → Si OK, incrementar a 50% → 100%
```

**Ventaja:** Riesgo limitado.

### Shadow Deployment

**Concepto:** Nuevo modelo recibe tráfico pero predicciones NO se usan (solo logging).

**Ventaja:** Validar en producción sin impacto.

### A/B Testing

**Concepto:** Dividir usuarios aleatoriamente entre modelos.

**Objetivo:** Medir impacto en métrica de negocio (ej: revenue, engagement).

**Importante:** Requiere significancia estadística.

📹 **Videos recomendados:**

1. [Deployment Strategies - DevOps Toolkit](https://www.youtube.com/watch?v=AWVTKBUnoIg) - 20 min

______________________________________________________________________

## 7. Retraining: ¿Cuándo y cómo?

### ¿Cuándo reentrenar?

1. **Performance degradation:** Accuracy cayó.
1. **Data drift detectado:** Distribución cambió.
1. **Scheduled:** Periódicamente (semanal, mensual).
1. **Nuevos datos disponibles:** Dataset creció significativamente.

### Estrategias de retraining

#### Retraining desde cero

Entrenar completamente con todos los datos.

**Ventaja:** Modelo fresco.
**Desventaja:** Costoso.

#### Incremental learning

Actualizar modelo existente con datos nuevos.

**Ventaja:** Rápido.
**Desventaja:** No todos los modelos lo soportan.

### Pipeline automatizado

1. Detectar trigger (drift, schedule).
1. Recolectar datos recientes.
1. Entrenar modelo.
1. Validar (métricas > umbral).
1. Registrar en model registry.
1. Desplegar con estrategia segura (canary).
1. Monitorear.

______________________________________________________________________

## 8. Herramientas y plataformas

### End-to-End Platforms

- **AWS SageMaker:** Completo, integrado con AWS.
- **Google Vertex AI:** Integrado con GCP.
- **Azure ML:** Integrado con Azure.
- **Databricks:** Unificado para datos + ML.

### Open-Source Tools

- **MLflow:** Tracking, registry, serving.
- **Kubeflow:** Pipelines de ML en Kubernetes.
- **Metaflow (Netflix):** Workflow management.
- **DVC:** Versionado de datos.
- **Weights & Biases:** Experiment tracking.

### Orchestration

- **Airflow:** Workflow orchestration.
- **Prefect:** Alternativa moderna a Airflow.
- **Argo Workflows:** Para Kubernetes.

📹 **Videos recomendados:**

1. [MLOps Tools Landscape - Full Stack Deep Learning](https://www.youtube.com/watch?v=d1ZAFxUPC8A) - 30 min

______________________________________________________________________

## 9. Buenas prácticas

- ✅ Empezar simple: No construir infraestructura compleja desde día 1.
- ✅ Automatizar gradualmente: A medida que el sistema madura.
- ✅ Definir SLOs (Service Level Objectives) desde el inicio: latencia, accuracy, uptime.
- ✅ Mantener trazabilidad completa: código, datos, modelo, predicciones.
- ✅ Implementar CI/CD para ML: tests automatizados, despliegue controlado.
- ✅ Monitorear más que accuracy: drift, latencia, costo.
- ✅ Usar feature store si reutilizas features.
- ✅ Desplegar con estrategia segura: canary, A/B testing.
- ✅ Diseñar runbooks para incidentes: qué hacer si accuracy cae.
- ✅ Documentar decisiones de arquitectura.

📚 **Recursos generales:**

- [MLOps Principles - Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Full Stack Deep Learning (Course)](https://fullstackdeeplearning.com/)
- [Made With ML (Course)](https://madewithml.com/)
- [Awesome MLOps (GitHub)](https://github.com/visenger/awesome-mlops)

______________________________________________________________________

## Final comprehension checklist

Antes de completar tu formación, deberías poder:

- ✅ Describir flujo completo desde experimento a producción.
- ✅ Versionar datos, código y modelos con DVC/MLflow.
- ✅ Implementar CI/CD pipeline para modelo.
- ✅ Servir modelo con API (FastAPI/Flask).
- ✅ Elegir estrategia de serving (batch, online, streaming) según caso.
- ✅ Monitorear latencia, accuracy, y drift en producción.
- ✅ Detectar data drift y concept drift.
- ✅ Implementar canary deployment para reducir riesgo.
- ✅ Diseñar pipeline automatizado de retraining.
- ✅ Definir SLOs y alertas para modelo en producción.

Si respondiste "sí" a todas, ¡estás listo para llevar modelos de ML a producción de forma profesional! 🎉
