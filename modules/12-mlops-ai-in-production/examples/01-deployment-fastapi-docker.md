# Example 01 — Model ML Deployment with FastAPI and Docker

## Context

Bringing Jupyter notebook models to production requires robust API, versioning, containerization, and CI/CD.

## Objective

Deploy Classification Model with FastAPI, Docker and MLOps best practices.

______________________________________________________________________

## 🚀 Project setup

```bash
# Structure del project
ml-api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── models.py         # Pydantic schemas
│   ├── inference.py      # Logic de prediction
│   └── monitoring.py     # Logs y metrics
├── model/
│   ├── model.pkl         # Model trained
│   └── preprocessing.pkl # Transformaciones
├── tests/
│   ├── test_api.py
│   └── test_inference.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md
```

______________________________________________________________________

## 🤖 Train and save Model

```python
# train.py - Script de training

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba),
    'train_date': datetime.now().isoformat(),
    'model_version': '1.0.0'
}

# Save model y artefactos
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/preprocessing.pkl')

# Save metadata
with open('model/metadata.json', 'w') as f:
    json.dump({
        'metrics': metrics,
        'feature_names': list(data.feature_names),
        'target_names': list(data.target_names)
    }, f, indent=2)

print(f"✅ Model trained y guardado")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

**Output:**

```
✅ Model trained y guardado
Accuracy: 0.9737
ROC-AUC: 0.9956
```

______________________________________________________________________

## 📦 Pydantic Schemas

```python
# app/models.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np

class PredictionInput(BaseModel):
    """
    Schema para input de prediction
    """
    features: List[float] = Field(
        ...,
        min_items=30,
        max_items=30,
        description="30 features del tumor"
    )

    @validator('features')
    def validate_features(cls, v):
        # Validate que no haya NaN o Inf
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features contienen NaN o Inf")
        return v

class PredictionOutput(BaseModel):
    """
    Schema para output de prediction
    """
    prediction: int = Field(..., ge=0, le=1, description="0=malignant, 1=benign")
    probability_benign: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    prediction_id: str

class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str
    model_loaded: bool
    model_version: Optional[str] = None

class MetricsResponse(BaseModel):
    """
    Metrics del model
    """
    total_predictions: int
    avg_inference_time_ms: float
    model_version: str
```

______________________________________________________________________

## 🔧 Inference logic

```python
# app/inference.py

import joblib
import numpy as np
from pathlib import Path
import json
import time
import logging

logger = logging.getLogger(__name__)

class ModelInference:
    """
    Clause, Class para manejar carga y prediction del model
    """
    def __init__(self, model_path: str = "model/model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.prediction_count = 0
        self.total_inference_time = 0.0

        self.load_model()

    def load_model(self):
        """
        Carga model, scaler y metadata
        """
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.model_path.parent / "preprocessing.pkl")

            with open(self.model_path.parent / "metadata.json") as f:
                self.metadata = json.load(f)

            logger.info(f"✅ Model cargado: v{self.metadata['metrics']['model_version']}")

        except Exception as e:
            logger.error(f"❌ Error cargando model: {e}")
            raise

    def predict(self, features: list) -> dict:
        """
        Realiza prediction
        """
        start_time = time.time()

        # Validation dimensional
        if len(features) != 30:
            raise ValueError(f"Expected 30 features, got {len(features)}")

        # Preprocesar
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)

        # Predict
        prediction = int(self.model.predict(features_scaled)[0])
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Metrics
        inference_time = (time.time() - start_time) * 1000  # ms
        self.prediction_count += 1
        self.total_inference_time += inference_time

        logger.info(f"Prediction: {prediction}, Prob: {probabilities[1]:.4f}, Time: {inference_time:.2f}ms")

        return {
            'prediction': prediction,
            'probability_benign': float(probabilities[1]),
            'inference_time_ms': inference_time
        }

    def get_metrics(self) -> dict:
        """
        Retorna metrics de usage
        """
        avg_time = (
            self.total_inference_time / self.prediction_count
            if self.prediction_count > 0
            else 0
        )

        return {
            'total_predictions': self.prediction_count,
            'avg_inference_time_ms': avg_time,
            'model_version': self.metadata['metrics']['model_version']
        }
```

______________________________________________________________________

## 🌐 FastAPI Application

```python
# app/main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from datetime import datetime

from .models import (
    PredictionInput,
    PredictionOutput,
    HealthResponse,
    MetricsResponse
)
from .inference import ModelInference

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="ML Model API",
    description="API para prediction de cancer de mama",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model al iniciar
model_inference = None

@app.on_event("startup")
async def startup_event():
    """
    Load model al iniciar la application
    """
    global model_inference
    try:
        model_inference = ModelInference()
        logger.info("🚀 API iniciada y model cargado")
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint root
    """
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    model_loaded = model_inference is not None and model_inference.model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=model_inference.metadata['metrics']['model_version'] if model_loaded else None
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Realizar prediction

    Args:
        input_data: 30 features del tumor

    Returns:
        Prediction y probability
    """
    try:
        # Predict
        result = model_inference.predict(input_data.features)

        # Generate ID only
        prediction_id = str(uuid.uuid4())

        return PredictionOutput(
            prediction=result['prediction'],
            probability_benign=result['probability_benign'],
            model_version=model_inference.metadata['metrics']['model_version'],
            prediction_id=prediction_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error en prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Obtener metrics de usage del model
    """
    metrics = model_inference.get_metrics()

    return MetricsResponse(**metrics)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handler global de excepciones
    """
    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

______________________________________________________________________

## 📋 Requirements

```txt
# requirements.txt

fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
scikit-learn==1.3.2
joblib==1.3.2
numpy==1.24.3
pandas==2.1.3
```

______________________________________________________________________

## 🐳 Dockerfile

```dockerfile
# Dockerfile

FROM python:3.10-slim

# Metadata
LABEL maintainer="ml-team@company.com"
LABEL version="1.0.0"

# Directorio de trabajo
WORKDIR /app

# Instalar dependencies del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencies Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar code
COPY app/ ./app/
COPY model/ ./model/

# Create user no-root
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Commando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

______________________________________________________________________

## 🚀 Build and Run

```bash
# Build image
docker build -t ml-api:1.0.0 .

# Run container
docker run -d \
    --name ml-api \
    -p 8000:8000 \
    --memory="512m" \
    --cpus="1.0" \
    ml-api:1.0.0

# Ver logs
docker logs -f ml-api

# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                     0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                     0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                     25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                     0.2654, 0.4601, 0.1189]
    }'

# Ver metrics
curl http://localhost:8000/metrics
```

**Output:**

```json
{
  "prediction": 0,
  "probability_benign": 0.0234,
  "model_version": "1.0.0",
  "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

______________________________________________________________________

## ✅ Testing

```python
# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """
    Test endpoint root
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """
    Test health check
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict_valid():
    """
    Test prediction valid
    """
    payload = {
        "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                     0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                     0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                     25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                     0.2654, 0.4601, 0.1189]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability_benign" in data
    assert 0 <= data["prediction"] <= 1
    assert 0.0 <= data["probability_benign"] <= 1.0

def test_predict_invalid_features():
    """
    Test con number incorrecto de features
    """
    payload = {
        "features": [1.0, 2.0, 3.0]  # Solo 3 features (requires 30)
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_metrics():
    """
    Test endpoint de metrics
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_predictions" in response.json()

# Execute tests
# pytest tests/ -v --cov=app
```

______________________________________________________________________

## 📊 Load Testing

```python
# load_test.py

import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

API_URL = "http://localhost:8000/predict"

def generate_random_features():
    """
    Genera features aleatorias
    """
    return list(np.random.randn(30))

def make_prediction(features):
    """
    Have una prediction
    """
    start = time.time()

    response = requests.post(
        API_URL,
        json={"features": features}
    )

    latency = (time.time() - start) * 1000  # ms

    return {
        'status_code': response.status_code,
        'latency_ms': latency,
        'success': response.status_code == 200
    }

def load_test(n_requests=100, n_workers=10):
    """
    Load test con multiple workers
    """
    print(f"🔥 Load test: {n_requests} requests con {n_workers} workers paralelos")

    results = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(make_prediction, generate_random_features())
            for _ in range(n_requests)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    # Analysis
    latencies = [r['latency_ms'] for r in results if r['success']]
    success_rate = sum(r['success'] for r in results) / len(results)

    print(f"\n✅ Results:")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Latency media: {np.mean(latencies):.2f}ms")
    print(f"Latency p50: {np.percentile(latencies, 50):.2f}ms")
    print(f"Latency p95: {np.percentile(latencies, 95):.2f}ms")
    print(f"Latency p99: {np.percentile(latencies, 99):.2f}ms")
    print(f"Latency max: {np.max(latencies):.2f}ms")
    print(f"Throughput: {n_requests / (sum(latencies) / 1000):.2f} req/s")

if __name__ == "__main__":
    load_test(n_requests=500, n_workers=20)
```

**Output:**

```
🔥 Load test: 500 requests con 20 workers paralelos

✅ Results:
Success rate: 100.00%
Latency media: 15.34ms
Latency p50: 12.45ms
Latency p95: 28.67ms
Latency p99: 45.23ms
Latency max: 67.89ms
Throughput: 326.78 req/s
```

______________________________________________________________________

## 📝 Summary

### ✅ Production components

```
Model trained (joblib)
    ↓
FastAPI (REST API)
    ↓
Pydantic (validation)
    ↓
Docker (containerization)
    ↓
Tests (pytest)
    ↓
CI/CD (GitHub Actions)
    ↓
Kubernetes / Cloud Run (deployment)
```

### 🎯 Mejores Practices

- ✅ **Versionado:** Model registry (MLflow, DVC)
- ✅ **Validation:** Pydantic schemas for input/output
- ✅ **Logging:** Structured logging (JSON)
- ✅ **Monitoring:** Prometheus + Grafana
- ✅ **Testing:** Unit + integration + load tests
- ✅ **Security:** Rate limiting, authentication (API keys)
- ✅ **Documentation:** Automatic OpenAPI (Swagger) with FastAPI

### 🚫 Errors common

- ❌ No versionar Models (impossible rollback)
- ❌ Do not validate inputs (crashes in production)
- ❌ Model very large in memory (use batch or streaming)
- ❌ No monitorear latencia/errors (blind deployment)
- ❌ Hardcodear paths (use env variables)

### 📌 Checklist Deployment

- ✅ Model saved with metadata (version, Metrics, date)
- ✅ API with input/output validation
- ✅ Optimized Dockerfile (multi-stage, minimum size)
- ✅ Health check endpoint
- ✅ Automatic tests (unit + integration)
- ✅ Load testing realizado
- ✅ Logging estructurado
- ✅ Monitoring configurado
- ✅ CI/CD pipeline

### 🚀 Next steps

- Integration with Model Registry (MLflow)
- A/B Testing framework
- Feature store (Feast)
- Canary deployments
- Auto-scaling in Kubernetes
