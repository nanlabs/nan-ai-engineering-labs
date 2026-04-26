# Practice 02 — Differential Privacy and Federated Learning

## 🎯 Objectives

- Implement differential privacy
- Privacy budget management
- Federated learning simulation
- Trade-off privacy/accuracy

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Laplace Mechanism

```python
import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    Duck noise Laplaciano para differential privacy.

    Args:
        true_value: Valor real
        sensitivity: Sensibilidad de la query
        epsilon: Privacy budget
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise

# Example: Private mean
data = np.array([10, 20, 30, 40, 50])
true_mean = data.mean()

# Sensitivity = (max - min) / n
sensitivity = (data.max() - data.min()) / len(data)

# different epsilons
epsilons = [0.1, 1.0, 10.0]

print(f"True mean: {true_mean:.2f}")
for eps in epsilons:
    private_mean = laplace_mechanism(true_mean, sensitivity, eps)
    error = abs(private_mean - true_mean)
    print(f"ε={eps:4.1f}: {private_mean:.2f} (error: {error:.2f})")
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Private gradient Descent

**Statement:**
Implement DP-SGD:

- Clip gradients per sample
- Add Gaussian noise to batch gradient
- Track privacy budget (ε, δ)

### Exercise 2.2: Privacy Budget Tracking

**Statement:**
Accounting system:

- Queries composition
- Calculate cumulative ε
- Stop when budget agotado

### Exercise 2.3: Federated Learning

**Statement:**
Simulate FL with 5 clients:

- Cada client, clientele entrena localmente
- Servidor agrega pesos
- FedAvg algorithm
- Measures communication overhead

### Exercise 2.4: Secure Aggregation

**Statement:**
Agrega gradients sin revelar:

- Secret sharing scheme
- Servidor solo ve agregado
- Clients do not see each other's gradients

### Exercise 2.5: Privacy-Utility Trade-off

**Statement:**
Experiment with ε:

- Vary epsilon: [0.1, 0.5, 1.0, 5.0, 10.0]
- Plot accuracy vs epsilon
- Find acceptable trade-off

______________________________________________________________________

## ✅ Checklist

- [ ] Laplace mechanism
- [ ] Differential privacy in ML
- [ ] Privacy budget tracking
- [ ] Federated learning simulation
- [ ] Privacy-utility trade-off analysis

______________________________________________________________________

## 📚 Resources

- [OpenDP](https://github.com/opendp/opendp)
- [PySyft (Federated Learning)](https://github.com/OpenMined/PySyft)
