# Práctica 01 — CI/CD para ML

## 🎯 Objetivos

- Setup GitHub Actions para ML
- Dockerize modelos
- Automated testing y retraining
- Version control para modelos

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Dockerfile para ML

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
# app.py
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(features: list[float]):
    X = np.array([features])
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: GitHub Actions Workflow

**Enunciado:**
Crea `.github/workflows/ml-pipeline.yml`:

```yaml
on: [push]
jobs:
  test:
    - Install dependencies
    - Run unit tests
    - Train model
    - Evaluate metrics
    - Upload artifacts
```

### Ejercicio 2.2: Model Registry

**Enunciado:**
Implementa model versioning:

- MLflow Model Registry
- Track experiments
- Compare metrics
- Promote best model

### Ejercicio 2.3: Automated Retraining

**Enunciado:**
Trigger retraining cuando:

- Nuevos datos disponibles
- Performance drift detectado
- Scheduled (weekly)

### Ejercicio 2.4: Data Drift Detection

**Enunciado:**
Monitor data distribution:

- Compare train vs production
- KS test, PSI metric
- Alert si drift significativo

### Ejercicio 2.5: A/B Testing Infrastructure

**Enunciado:**
Deploy dos versiones:

- Model A (current)
- Model B (candidate)
- Route traffic 50/50
- Compare metrics

______________________________________________________________________

## ✅ Checklist

- [ ] Dockerize ML models
- [ ] CI/CD con GitHub Actions
- [ ] Model versioning
- [ ] Automated retraining
- [ ] Data drift detection

______________________________________________________________________

## 📚 Recursos

- [MLflow](https://mlflow.org/)
- [DVC (Data Version Control)](https://dvc.org/)
