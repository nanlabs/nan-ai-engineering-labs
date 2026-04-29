# Practice 01 — Bias Detection and Mitigation

## 🎯 Objectives

- Detect bias in datasets
- Measure disparate impact
- Implement debiasing techniques
- Evaluate fairness metrics

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Bias Analysis

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

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Reweighting

**Statement:**
Implement reweighting in Training:

- Calculate sample weights per group
- Train with `class_weight='balanced'`
- Compare fairness antes/after

### Exercise 2.2: Threshold Optimization

**Statement:**
Adjust thresholds per group:

- Look for threshold that equalizes TPR
- O equaliza FPR
- Analyze trade-off with accuracy

### Exercise 2.3: Adversarial Debiasing

**Statement:**
Train Model that:

- Predice target correctly
- Adversary no can predecir sensitive attribute
- Loss combinado

### Exercise 2.4: Counterfactual Fairness

**Statement:**
Genera counterfactuals:

- Change gender while maintaining features
- Compare predictions
- Measure cambio average

### Exercise 2.5: Fairness Metrics Dashboard

**Statement:**
Visualize multiple Metrics:

- Demographic parity
- Equalized odds
- Equal opportunity
- Disparate impact

______________________________________________________________________

## ✅ Checklist

- [ ] Detect bias in Data
- [ ] Calculate disparate impact
- [ ] Reweighting and resampling
- [ ] Threshold optimization
- [ ] Adversarial debiasing

______________________________________________________________________

## 📚 Resources

- [Fairlearn](https://fairlearn.org/)
- [AI Fairness 360](https://aif360.mybluemix.net/)
