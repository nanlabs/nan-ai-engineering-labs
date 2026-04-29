# Theory — MLOps & AI in Production

## Why this module matters

Only 13% of ML Models reach production (according to Gartner). MLOps bridges the gap between experiments on notebooks and reliable systems in production. Mastering MLOps positions you as a complete engineer, not just a data scientist.

______________________________________________________________________

## 1. What is MLOps?

**MLOps (Machine Learning Operations):** Set of Practices, tools and culture to **build, deploy and operate** ML Models in a **reliable, reproducible and scalable** way.

### Analogy with DevOps

- **DevOps:** Automation of software development and operations.
- **MLOps:** DevOps + unique ML challenges (Data, Models, drift).

### Why is MLOps different?

- **Code is only part:** Data and Models also change.
- **Silent degradation:** Model loses accuracy without technical errors.
- **Intensive experimentation:** Many experiments, few go to production.

📹 **Videos recommended:**

1. [MLOps Explained - Google Cloud](https://www.youtube.com/watch?v=6gdrwFMaEZ0) - 10 min
1. [MLOps: From Model to Production - Andrew Ng](https://www.youtube.com/watch?v=06-AZXmwHjo) - 1 hora

______________________________________________________________________

## 2. ML lifecycle in production

### Phase 1: Development and experimentation

**Activities:**

- EDA (Exploratory Data Analysis).
- Feature engineering.
- Model Training.
- Comparison of Algorithms.

**Tools:**

- Jupyter Notebooks, Google Colab.
- MLflow, Weights & Biases for tracking.

### Phase 2: Training and Validation

**Activities:**

- Final Train Model with complete Data.
- Rigorous validation (cross-validation, test set).
- Hyperparameters tuning.

**Tools:**

- Kubeflow, SageMaker, Vertex AI.
- Optuna, Ray Tune for hyperparameter tuning.

### Phase 3: Model Registry

**Activities:**

- Save Trained Model.
- Version Model.
- Document metadata (Metrics, features, Hyperparameters).

**Tools:**

- MLflow Model Registry.
- SageMaker Model Registry.
- Vertex AI Model Registry.

### Phase 4: Deployment

**Activities:**

- Package Model (Docker, ONNX).
- Expose Prediction API.
- Configure infrastructure (CPU/GPU, scaling).

**Tools:**

- FastAPI, Flask for APIs.
- Docker, Kubernetes.
- Cloud services: SageMaker, Vertex AI, Azure ML.

### Phase 5: Monitoring and retraining

**Activities:**

- Monitor performance (latency, accuracy).
- Detect drift.
- Retrain Model with recent Data.

**Tools:**

- Prometheus, Grafana.
- Evidently AI, Fiddler for drift detection.

📹 **Videos recommended:**

1. [ML Lifecycle - Full Stack Deep Learning](https://www.youtube.com/watch?v=pvaIi0l1GME) - 1 hora

______________________________________________________________________

## 3. Key Components of MLOps

### Versioned

#### Code versioning

- **Git:** Standard for source code.

#### Data Versioning

Data changes, you need to track versions.

**Tools:**

- **DVC (Data Version Control):** Git for Data.
- **LakeFS:** Versioning of data lakes.
- **Delta Lake:** Transactional versioning.

#### Versioning of Models

Save each Model trained with metadata.

**Metadata:**

- Data version used.
- Hyperparameters.
- Validation Metrics.
- Training Date.
- Author.

**Tools:**

- MLflow Model Registry.
- Weights & Biases.

### CI/CD for ML

**CI (Continuous Integration):**

- Automated tests:
  - Unit tests for code.
  - Data validation (schema, distributions).
- Model validation (Minimum Metrics).

**CD (Continuous Deployment):**

- Automated deployment to staging.
- Validation in staging.
- Deployment to production (manual or automatic).

**Tools:**

- GitHub Actions, GitLab CI.
- Jenkins, CircleCI.

### Feature Store

**Problem:** Features calculated multiple times inconsistently.

**Solution:** Centralized features repository.

**Benefits:**

- Reuse of features.
- Consistency between Training and serving.
- Pre-computed features (low latency).

**Tools:**

- Feast (open-source).
- Tecton.
- SageMaker Feature Store.

📹 **Videos recommended:**

1. [Feature Stores Explained - Tecton](https://www.youtube.com/watch?v=2m2LqZfKqKI) - 15 min
1. [CI/CD for ML - Google Cloud](https://www.youtube.com/watch?v=hFhZsDgZFfg) - 20 min

📚 **Resources written:**

- [DVC Documentation](https://dvc.org/doc)
- [Feast Documentation](https://docs.feast.dev/)

______________________________________________________________________

## 4. Serving: How to serve Predictions

### Batch Prediction

**Usage:** Scheduled predictions about batches of data.

**Example:** Recommend products weekly to all users.

**Advantage:** Simple, tolerant of high latency.

### Online Prediction (Real-time)

**Usage:** Real-time predictions per request.

**Example:** Detect fraud at the time of transaction.

**Requirement:** Low latency (\<100ms typically).

**Tools:**

- FastAPI, Flask.
- TensorFlow Serving, TorchServe.
- Seldon Core, KServe.

### Streaming Prediction

**Usage:** Predictions about continuous data flows.

**Example:** Detection of Anomalies in IoT sensors.

**Tools:**

- Kafka, Kinesis.
- Spark Streaming, Flink.

### Choice

| Mode | Latency | Cost | Complexity |
| ------------- | ---------------- | ----- | ----------- |
| **Batch** | High (hours) | Low | Low |
| **Online** | Low (ms) | High | Medium |
| **Streaming** | Average (seconds) | High | High |

📹 **Videos recommended:**

1. [Model Serving Patterns - Chip Huyen](https://www.youtube.com/watch?v=KdmcFqbMSEM) - 30 min

______________________________________________________________________

## 5. Production monitoring

### Operational metrics

#### Latency

Prediction response time.

**Target:** Depends on the case (p95 \<100ms for real-time).

#### Throughput

Predictions per second.

#### Uptime / Availability

% of time that the service is available.

**Target:** 99.9% ("three nines") or more.

#### Cost

Computational cost (CPU/GPU, data transfer).

### Model Metrics

#### accuracy / Performance

Does the Model still have good performance?

**Problem:** In production you don't always have ground truth immediately.

**Solution:**

- Proxy metrics (CTR, conversion rate).
- Delayed labels (delayed labels).

#### Data Drift

**Definition:** Input distribution changes.

**Example:** Model trained in 2020, in 2024 user behavior changed.

**Detection:**

- Kolmogorov-Smirnov test.
- Population Stability Index (PSI).

#### Concept Drift

**Definition:** Relationship between inputs and target changes.

**Example:** Before COVID, after COVID (purchase patterns changed radically).

**Detection:**

- Monitor accuracy in recent subset.
- Compare distributions of Predictions.

#### Prediction Drift

Predictions distribution changes.

**Example:** Model predicts "fraud" for 1% of transactions, now predicts 10%.

**Alert:** Possible change in Data or failure in Model.

📹 **Videos recommended:**

1. [ML Monitoring - Chip Huyen](https://www.youtube.com/watch?v=TYk8wOo48xs) - 40 min
1. [Data Drift Explained - Evidently AI](https://www.youtube.com/watch?v=bKh6CTgHiLU) - 15 min

📚 **Resources written:**

- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [Google ML Monitoring Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Alerts

**Configure alerts when:**

- Latency > threshold.
- error rate > umbral.
- Data drift detected.
- accuracy cae > X%.

**Tools:**

- Prometheus + Alertmanager.
- PagerDuty, Opsgenie.

______________________________________________________________________

## 6. Deployment strategies

### Blue/Green Deployment

**Concept:** Maintain two environments (blue/green). Change traffic instantly.

**Advantage:** Instant rollback.
**Disadvantage:** Duplicates infrastructure (expensive).

### Canary Deployment

**Concept:** Deploy new Model to small % of traffic. Increase gradually.

**Example:**

```
V1: 95% traffic
V2 (canary): 5% traffic
→ Monitor → If OK, increase to 50% → 100%
```

**Advantage:** Limited risk.

### Shadow Deployment

**Concept:** New Model receives traffic but Predictions are NOT used (logging only).

**Advantage:** Validate yourself in production without impact.

### A/B Testing

**Concept:** Split users randomly between Models.

**Objective:** Measure impact on business metrics (e.g. revenue, engagement).

**Important:** Requires statistical significance.

📹 **Videos recommended:**

1. [Deployment Strategies - DevOps Toolkit](https://www.youtube.com/watch?v=AWVTKBUnoIg) - 20 min

______________________________________________________________________

## 7. Retraining: When and how?

### When to retrain?

1. **Performance degradation:** accuracy dropped.
1. **Data drift detected:** Distribution changed.
1. **Scheduled:** Periodically (weekly, monthly).
1. **New Data available:** Dataset grew significantly.

### Retraining strategies

#### Retraining from cero

Train completely with all Data.

**Advantage:** Model fresco.
**Disadvantage:** Expensive.

#### Incremental learning

Update existing Model with new Data.

**Advantage:** Fast.
**Disadvantage:** Not all Models support it.

### Automated pipeline

1. Detect trigger (drift, schedule).
1. Collect recent data.
1. Train Model.
1. Validate (Metrics > umbral).
1. Register in model registry.
1. Deploy with a safe strategy (canary).
1. Monitor.

______________________________________________________________________

## 8. Tools and platforms

### End-to-End Platforms

- **AWS SageMaker:** Complete, integrated with AWS.
- **Google Vertex AI:** Integrated with GCP.
- **Azure ML:** Integrated with Azure.
- **Databricks:** Unified for Data + ML.

### Open-Source Tools

- **MLflow:** Tracking, registry, serving.
- **Kubeflow:** ML pipelines in Kubernetes.
- **Metaflow (Netflix):** Workflow management.
- **DVC:** Data versioning.
- **Weights & Biases:** Experiment tracking.

### Orchestration

- **Airflow:** Workflow orchestration.
- **Prefect:** Modern alternative to Airflow.
- **Argo Workflows:** For Kubernetes.

📹 **Videos recommended:**

1. [MLOps Tools Landscape - Full Stack Deep Learning](https://www.youtube.com/watch?v=d1ZAFxUPC8A) - 30 min

______________________________________________________________________

## 9. Buenas Practices

- ✅ Start simple: Don't build complex infrastructure from day 1.
- ✅ Automate gradually: As the system matures.
- ✅ Define SLOs (Service Level Objectives) from the beginning: latency, accuracy, uptime.
- ✅ Maintain complete traceability: code, Data, Model, Predictions.
- ✅ Implement CI/CD for ML: automated tests, controlled deployment.
- ✅ Monitor more than accuracy: drift, latency, cost.
- ✅ Use feature store if you reuse features.
- ✅ Deploy with a secure strategy: canary, A/B testing.
- ✅ Design runbooks for incidents: what to do if accuracy drops.
- ✅ Document architectural decisions.

📚 **General resources:**

- [MLOps Principles - Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Full Stack Deep Learning (Course)](https://fullstackdeeplearning.com/)
- [Made With ML (Course)](https://madewithml.com/)
- [Awesome MLOps (GitHub)](https://github.com/visenger/awesome-mlops)

______________________________________________________________________

## Final comprehension checklist

Before completing your training, you should be able to:

- ✅ Describe flow complete from experiment to production.
- ✅ Version Data, code and Models with DVC/MLflow.
- ✅ Implement CI/CD pipeline for Model.
- ✅ Serve Model with API (FastAPI/Flask).
- ✅ Choose serving strategy (batch, online, streaming) according to case.
- ✅ Monitor latency, accuracy, and drift in production.
- ✅ Detect data drift and concept drift.
- ✅ Implement canary deployment to reduce risk.
- ✅ Design automated retraining pipeline.
- ✅ Define SLOs and alerts for Model in production.

If you answered "yes" to all, you're ready to take ML Models into production professionally! 🎉
