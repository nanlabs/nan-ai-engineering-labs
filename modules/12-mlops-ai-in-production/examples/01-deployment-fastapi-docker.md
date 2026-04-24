# Ejemplo 01 — Deployment de Modelo ML con FastAPI y Docker

## Contexto

Llevar modelos de Jupyter notebooks a producción requiere API robusta, versionado, containerización y CI/CD.

## Objective

Desplegar modelo de clasificación con FastAPI, Docker y mejores prácticas de MLOps.

______________________________________________________________________

## 🚀 Setup del proyecto

```bash
# Estructura del proyecto
ml-api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── models.py         # Pydantic schemas
│   ├── inference.py      # Lógica de predicción
│   └── monitoring.py     # Logs y métricas
├── model/
│   ├── model.pkl         # Modelo entrenado
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

## 🤖 Entrenar y guardar modelo

```python
# train.py - Script de entrenamiento

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

# Cargar datos
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

# Entrenar
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluar
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba),
    'train_date': datetime.now().isoformat(),
    'model_version': '1.0.0'
}

# Guardar modelo y artefactos
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/preprocessing.pkl')

# Guardar metadata
with open('model/metadata.json', 'w') as f:
    json.dump({
        'metrics': metrics,
        'feature_names': list(data.feature_names),
        'target_names': list(data.target_names)
    }, f, indent=2)

print(f"✅ Modelo entrenado y guardado")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

**Salida:**

```
✅ Modelo entrenado y guardado
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
    Schema para input de predicción
    """
    features: List[float] = Field(
        ...,
        min_items=30,
        max_items=30,
        description="30 features del tumor"
    )

    @validator('features')
    def validate_features(cls, v):
        # Validar que no haya NaN o Inf
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features contienen NaN o Inf")
        return v

class PredictionOutput(BaseModel):
    """
    Schema para output de predicción
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
    Métricas del modelo
    """
    total_predictions: int
    avg_inference_time_ms: float
    model_version: str
```

______________________________________________________________________

## 🔧 Lógica de inferencia

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
    Clase para manejar carga y predicción del modelo
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
        Carga modelo, scaler y metadata
        """
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.model_path.parent / "preprocessing.pkl")

            with open(self.model_path.parent / "metadata.json") as f:
                self.metadata = json.load(f)

            logger.info(f"✅ Modelo cargado: v{self.metadata['metrics']['model_version']}")

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise

    def predict(self, features: list) -> dict:
        """
        Realiza predicción
        """
        start_time = time.time()

        # Validación dimensional
        if len(features) != 30:
            raise ValueError(f"Expected 30 features, got {len(features)}")

        # Preprocesar
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)

        # Predecir
        prediction = int(self.model.predict(features_scaled)[0])
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Métricas
        inference_time = (time.time() - start_time) * 1000  # ms
        self.prediction_count += 1
        self.total_inference_time += inference_time

        logger.info(f"Predicción: {prediction}, Prob: {probabilities[1]:.4f}, Time: {inference_time:.2f}ms")

        return {
            'prediction': prediction,
            'probability_benign': float(probabilities[1]),
            'inference_time_ms': inference_time
        }

    def get_metrics(self) -> dict:
        """
        Retorna métricas de uso
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

# Crear app
app = FastAPI(
    title="ML Model API",
    description="API para predicción de cáncer de mama",
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

# Cargar modelo al iniciar
model_inference = None

@app.on_event("startup")
async def startup_event():
    """
    Cargar modelo al iniciar la aplicación
    """
    global model_inference
    try:
        model_inference = ModelInference()
        logger.info("🚀 API iniciada y modelo cargado")
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint raíz
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
    Realizar predicción

    Args:
        input_data: 30 features del tumor

    Returns:
        Predicción y probabilidad
    """
    try:
        # Predecir
        result = model_inference.predict(input_data.features)

        # Generar ID único
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
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Obtener métricas de uso del modelo
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

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app/ ./app/
COPY model/ ./model/

# Crear usuario no-root
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

______________________________________________________________________

## 🚀 Build y Run

```bash
# Build imagen
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

# Ver métricas
curl http://localhost:8000/metrics
```

**Salida:**

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
    Test endpoint raíz
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
    Test predicción válida
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
    Test con número incorrecto de features
    """
    payload = {
        "features": [1.0, 2.0, 3.0]  # Solo 3 features (requiere 30)
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_metrics():
    """
    Test endpoint de métricas
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_predictions" in response.json()

# Ejecutar tests
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
    Hace una predicción
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
    Load test con múltiples workers
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

    # Análisis
    latencies = [r['latency_ms'] for r in results if r['success']]
    success_rate = sum(r['success'] for r in results) / len(results)

    print(f"\n✅ Resultados:")
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

**Salida:**

```
🔥 Load test: 500 requests con 20 workers paralelos

✅ Resultados:
Success rate: 100.00%
Latency media: 15.34ms
Latency p50: 12.45ms
Latency p95: 28.67ms
Latency p99: 45.23ms
Latency max: 67.89ms
Throughput: 326.78 req/s
```

______________________________________________________________________

## 📝 Resumen

### ✅ Componentes de producción

```
Modelo entrenado (joblib)
    ↓
FastAPI (REST API)
    ↓
Pydantic (validación)
    ↓
Docker (containerización)
    ↓
Tests (pytest)
    ↓
CI/CD (GitHub Actions)
    ↓
Kubernetes / Cloud Run (deployment)
```

### 🎯 Mejores prácticas

- ✅ **Versionado:** Model registry (MLflow, DVC)
- ✅ **Validación:** Pydantic schemas para input/output
- ✅ **Logging:** Structured logging (JSON)
- ✅ **Monitoring:** Prometheus + Grafana
- ✅ **Testing:** Unit + integration + load tests
- ✅ **Security:** Rate limiting, authentication (API keys)
- ✅ **Documentation:** OpenAPI (Swagger) automático con FastAPI

### 🚫 Errores comunes

- ❌ No versionar modelos (imposible rollback)
- ❌ No validar inputs (crashes en producción)
- ❌ Modelo muy grande en memoria (usar batch o streaming)
- ❌ No monitorear latencia/errors (blind deployment)
- ❌ Hardcodear paths (usar env variables)

### 📌 Checklist Deployment

- ✅ Modelo guardado con metadata (versión, métricas, fecha)
- ✅ API con validación de input/output
- ✅ Dockerfile optimizado (multi-stage, tamaño mínimo)
- ✅ Health check endpoint
- ✅ Tests automáticos (unit + integration)
- ✅ Load testing realizado
- ✅ Logging estructurado
- ✅ Monitoring configurado
- ✅ CI/CD pipeline

### 🚀 Next steps

- Integración con Model Registry (MLflow)
- A/B Testing framework
- Feature store (Feast)
- Canary deployments
- Auto-scaling en Kubernetes
