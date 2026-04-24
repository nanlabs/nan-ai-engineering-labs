# Práctica 02 — Differential Privacy y Federated Learning

## 🎯 Objetivos

- Implementar differential privacy
- Privacy budget management
- Federated learning simulation
- Trade-off privacy/accuracy

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Laplace Mechanism

```python
import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    Añade ruido Laplaciano para differential privacy.

    Args:
        true_value: Valor real
        sensitivity: Sensibilidad de la query
        epsilon: Privacy budget
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise

# Ejemplo: Private mean
data = np.array([10, 20, 30, 40, 50])
true_mean = data.mean()

# Sensitivity = (max - min) / n
sensitivity = (data.max() - data.min()) / len(data)

# diferentes epsilons
epsilons = [0.1, 1.0, 10.0]

print(f"True mean: {true_mean:.2f}")
for eps in epsilons:
    private_mean = laplace_mechanism(true_mean, sensitivity, eps)
    error = abs(private_mean - true_mean)
    print(f"ε={eps:4.1f}: {private_mean:.2f} (error: {error:.2f})")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Private Gradient Descent

**Enunciado:**
Implementa DP-SGD:

- Clip gradients por sample
- Añade ruido gaussiano a batch gradient
- Track privacy budget (ε, δ)

### Ejercicio 2.2: Privacy Budget Tracking

**Enunciado:**
Sistema de accounting:

- Composición de queries
- Calculate cumulative ε
- Stop cuando budget agotado

### Ejercicio 2.3: Federated Learning

**Enunciado:**
Simula FL con 5 clientes:

- Cada cliente entrena localmente
- Servidor agrega pesos
- FedAvg algorithm
- Mide comunicación overhead

### Ejercicio 2.4: Secure Aggregation

**Enunciado:**
Agrega gradients sin revelar:

- Secret sharing scheme
- Servidor solo ve agregado
- Clientes no ven gradients de otros

### Ejercicio 2.5: Privacy-Utility Trade-off

**Enunciado:**
Experimenta con ε:

- Vary epsilon: [0.1, 0.5, 1.0, 5.0, 10.0]
- Plot accuracy vs epsilon
- Find acceptable trade-off

______________________________________________________________________

## ✅ Checklist

- [ ] Laplace mechanism
- [ ] Differential privacy en ML
- [ ] Privacy budget tracking
- [ ] Federated learning simulation
- [ ] Privacy-utility trade-off analysis

______________________________________________________________________

## 📚 Recursos

- [OpenDP](https://github.com/opendp/opendp)
- [PySyft (Federated Learning)](https://github.com/OpenMined/PySyft)
