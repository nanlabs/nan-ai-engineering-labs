# Práctica 01 — Detección y Mitigación de Bias

## 🎯 Objetivos

- Detectar bias en datasets
- Medir disparate impact
- Implementar debiasing techniques
- Evaluar fairness metrics

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Análisis de Bias

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Dataset simulado (loan approval)
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, n),
    'credit_score': np.random.normal(650, 100, n),
    'age': np.random.randint(18, 70, n),
    'gender': np.random.choice(['M', 'F'], n)
})

# Bias: mujeres tienen menor approval rate
data['approved'] = (
    (data['income'] > 40000) &
    (data['credit_score'] > 600) &
    ((data['gender'] == 'M') | (np.random.random(n) > 0.3))
).astype(int)

# Analysis
print("Approval rates:")
print(data.groupby('gender')['approved'].mean())

# Disparate Impact
female_approval = data[data['gender'] == 'F']['approved'].mean()
male_approval = data[data['gender'] == 'M']['approved'].mean()
di = female_approval / male_approval
print(f"\\nDisparate Impact: {di:.3f}")
print(f"80% rule violated: {di < 0.8}")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Reweighting

**Enunciado:**
Implementa reweighting en entrenamiento:

- Calcula sample weights por grupo
- Entrena con `class_weight='balanced'`
- Compara fairness antes/después

### Ejercicio 2.2: Threshold Optimization

**Enunciado:**
Ajusta thresholds por grupo:

- Busca threshold que equaliza TPR
- O equaliza FPR
- Analiza trade-off con accuracy

### Ejercicio 2.3: Adversarial Debiasing

**Enunciado:**
Entrena modelo que:

- Predice target correctamente
- Adversary no puede predecir sensitive attribute
- Loss combinado

### Ejercicio 2.4: Counterfactual Fairness

**Enunciado:**
Genera counterfactuals:

- Cambia género manteniendo features
- Compara predictions
- Mide cambio promedio

### Ejercicio 2.5: Fairness Metrics Dashboard

**Enunciado:**
Visualiza múltiples métricas:

- Demographic parity
- Equalized odds
- Equal opportunity
- Disparate impact

______________________________________________________________________

## ✅ Checklist

- [ ] Detectar bias en datos
- [ ] Calcular disparate impact
- [ ] Reweighting y resampling
- [ ] Threshold optimization
- [ ] Adversarial debiasing

______________________________________________________________________

## 📚 Recursos

- [Fairlearn](https://fairlearn.org/)
- [AI Fairness 360](https://aif360.mybluemix.net/)
