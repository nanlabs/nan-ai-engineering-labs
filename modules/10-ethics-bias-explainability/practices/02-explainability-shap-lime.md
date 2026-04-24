# Práctica 02 — Explainability con SHAP y LIME

## 🎯 Objetivos

- Implementar SHAP values
- Usar LIME para black-box models
- Visualizar feature importance
- Generar explanations para usuarios

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: SHAP con Tree Models

```python
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Dataset
X, y = load_iris(return_X_y=True, as_frame=True)
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X.columns = feature_names

# Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualización
shap.summary_plot(shap_values, X, feature_names=feature_names)
plt.savefig('shap_summary.png')

# Individual prediction
instance = X.iloc[0]
shap.force_plot(explainer.expected_value[0], shap_values[0][0], instance)
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: LIME para Neural Networks

**Enunciado:**
Explica predicciones de NN:

- Usa `lime.lime_tabular.LimeTabularExplainer`
- Genera explanation para instancia
- Visualiza top features

### Ejercicio 2.2: Counterfactual Explanations

**Enunciado:**
Genera counterfactuals:
"Si feature X fuera Y, predicción cambiaría a Z"

- Encuentra mínimos cambios necesarios
- Mantén feasibilidad

### Ejercicio 2.3: Partial Dependence Plots

**Enunciado:**
Visualiza efecto de features:

- `sklearn.inspection.partial_dependence`
- PDPs para top 4 features
- ICE plots para heterogeneidad

### Ejercicio 2.4: Anchors

**Enunciado:**
Implementa Anchors algorithm:

- Reglas suficientes para predicción
- "SI edad > 50 AND income > 60k THEN approve"
- Coverage y precision

### Ejercicio 2.5: Explanation Dashboard

**Enunciado:**
Interface interactiva:

- Usuario selecciona instancia
- Muestra SHAP + LIME + Counterfactuals
- Permite "what-if" analysis

______________________________________________________________________

## ✅ Checklist

- [ ] SHAP values y visualizaciones
- [ ] LIME para modelos black-box
- [ ] Partial dependence plots
- [ ] Counterfactual explanations
- [ ] Anchors rules

______________________________________________________________________

## 📚 Recursos

- [SHAP Docs](https://shap.readthedocs.io/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
