# Practice 02 — Explainability with SHAP and LIME

## 🎯 Objectives

- Implement SHAP values
- Wear LIME for black-box models
- Visualize feature importance
- Generate explanations for users

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: SHAP with Tree Models

```python
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Dataset
X, y = load_iris(return_X_y=True, as_frame=True)
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X.columns = feature_names

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualization
shap.summary_plot(shap_values, X, feature_names=feature_names)
plt.savefig('shap_summary.png')

# Individual prediction
instance = X.iloc[0]
shap.force_plot(explainer.expected_value[0], shap_values[0][0], instance)
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: LIME for Neural Networks

**Statement:**
Explains NN Predictions:

- Usa `lime.lime_tabular.LimeTabularExplainer`
- Generate explanation for instance
- Visualize top features

### Exercise 2.2: Counterfactual Explanations

**Statement:**
Genera counterfactuals:
"If feature X were Y, Prediction would change to Z"

- Find minimum necessary changes
- Maintain feasibility

### Exercise 2.3: Partial Dependence Plots

**Statement:**
Visualize effect of features:

- `sklearn.inspection.partial_dependence`
- PDPs for top 4 features
- ICE plots for heterogeneity

### Exercise 2.4: Anchors

**Statement:**
Implement Anchors algorithm:

- Sufficient rules for Prediction
- "SI edad > 50 AND income > 60k THEN approve"
- Coverage and precision

### Exercise 2.5: Explanation Dashboard

**Statement:**
Interface interactiva:

- User selecciona instancia
- Muestra SHAP + LIME + Counterfactuals
- Permite "what-if" analysis

______________________________________________________________________

## ✅ Checklist

- [ ] SHAP values ​​and visualizations
- [ ] LIME for black-box Models
- [ ] Partial dependence plots
- [ ] Counterfactual explanations
- [ ] Anchors rules

______________________________________________________________________

## 📚 Resources

- [SHAP Docs](https://shap.readthedocs.io/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
