# Practice 01 — CI/CD for ML

## 🎯 Objectives

- Setup GitHub Actions for ML
- Dockerize Models
- Automated testing and retraining
- Version control for Models

______________________________________________________________________

## 📚 Parte 1: Exercises Guiados

### Exercise 1.1: Dockerfile for ML

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

## 🚀 Parte 2: Exercises Propuestos

### Exercise 2.1: GitHub Actions Workflow

**Statement:**
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

### Exercise 2.2: Model Registry

**Statement:**
Implementa model versioning:

- MLflow Model Registry
- Track experiments
- Compare metrics
- Promote best model

### Exercise 2.3: Automated Retraining

**Statement:**
Trigger retraining when:

- Nuevos Data disponibles
- Performance drift detectado
- Scheduled (weekly)

### Exercise 2.4: Data Drift Detection

**Statement:**
Monitor data distribution:

- Compare train vs production
- KS test, PSI metric
- Alert si drift significativo

### Exercise 2.5: A/B Testing Infrastructure

**Statement:**
Deploy dos versions:

- Model A (current)
- Model B (candidate)
- Route traffic 50/50
- Compare metrics

______________________________________________________________________

## ✅ Checklist

- [ ] Dockerize ML models
- [ ] CI/CD with GitHub Actions
- [ ] Model versioning
- [ ] Automated retraining
- [ ] Data drift detection

______________________________________________________________________

## 📚 Resources

- [MLflow](https://mlflow.org/)
- [DVC (Data Version Control)](https://dvc.org/)
