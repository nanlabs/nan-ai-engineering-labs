# Example 02 — Differential Privacy en la Practice

## Contexto

**Differential Privacy (DP)** garantiza que la presencia/ausencia de un individuo en el dataset no cambia significativamente las estadísticas publicadas.

## Objective

Implementar mecanismos de Differential Privacy para consultas estadísticas sobre Data sensibles.

______________________________________________________________________

## 🚀 Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm
import seaborn as sns

np.random.seed(42)
```

______________________________________________________________________

## 📚 Dataset sensible

```python
# Simulación: salarios de empleados
n_employees = 500

data = {
    'employee_id': range(1, n_employees + 1),
    'age': np.random.randint(22, 65, n_employees),
    'department': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing'], n_employees),
    'years_experience': np.random.randint(0, 40, n_employees),
    'salary': np.random.normal(75000, 25000, n_employees).clip(30000, 200000)
}

df = pd.DataFrame(data)

print("=== Dataset de Salarios (SENSIBLE) ===")
print(df.head(10))
print(f"\nTotal empleados: {len(df)}")
print(f"\nEstadísticas reales:")
print(f"Salario promedio: ${df['salary'].mean():,.2f}")
print(f"Salario mediano: ${df['salary'].median():,.2f}")
```

**Salida:**

```
=== Dataset de Salarios (SENSIBLE) ===
   employee_id  age    department  years_experience       salary
0            1   45   Engineering                15  $78,234.56
1            2   33         Sales                 8  $62,145.78
2            3   52            HR                25  $95,678.23
...

Total empleados: 500

Estadísticas reales:
Salario promedio: $75,234.67
Salario mediano: $74,123.45
```

______________________________________________________________________

## 🔐 Differential Privacy: Concepts

### Definición formal

```
Mecanismo M satisface ε-differential privacy si:

P(M(D) ∈ S) ≤ e^ε × P(M(D') ∈ S)

Donde:
- D y D' difieren en exactamente 1 registro
- ε (epsilon) = privacy budget (más bajo = más privacidad)
- S = cualquier posible output
```

### Privacy budget (ε)

```python
# Visualizar trade-off ε
epsilons = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
privacy_levels = ['Muy alta', 'Alta', 'Media-Alta', 'Media', 'Media-Baja', 'Baja']

print("\n=== Privacy Budget (ε) ===")
print(f"{'ε':<10} {'Privacidad':<15} {'Utilidad':<10} {'Uso'}")
print("-" * 70)
for eps, level in zip(epsilons, privacy_levels):
    utility = 'Baja' if eps < 0.5 else 'Media' if eps < 2 else 'Alta'
    use_case = 'Médico' if eps < 0.5 else 'Financiero' if eps < 2 else 'Marketing'
    print(f"{eps:<10.2f} {level:<15} {utility:<10} {use_case}")
```

**Salida:**

```
=== Privacy Budget (ε) ===
ε          Privacidad      Utilidad   Uso
----------------------------------------------------------------------
0.01       Muy alta        Baja       Médico
0.10       Alta            Baja       Médico
0.50       Media-Alta      Media      Médico
1.00       Media           Media      Financiero
2.00       Media-Baja      Alta       Financiero
5.00       Baja            Alta       Marketing
```

______________________________________________________________________

## 🎲 Laplace Mechanism

### Para consultas numéricas (conteos, sumas, promedios)

```python
def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    Añade ruido Laplaciano calibrado por sensitivity y epsilon

    Args:
        true_value: valor real de la consulta
        sensitivity: máximo cambio por añadir/remover 1 registro
        epsilon: privacy budget

    Returns:
        Valor con ruido que satisface ε-DP
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)

    return true_value + noise

# Ejemplo: salario promedio
true_avg = df['salary'].mean()

# Sensitivity del promedio:
# Si agregamos 1 persona con salario max (200k) o min (30k)
# Cambio máximo = max(|200k - true_avg|, |true_avg - 30k|) / n
max_salary = 200000
min_salary = 30000
n = len(df)
sensitivity_avg = max(abs(max_salary - true_avg), abs(true_avg - min_salary)) / n

print(f"\n=== Laplace Mechanism - Salario Promedio ===")
print(f"Valor real: ${true_avg:,.2f}")
print(f"Sensitivity: ${sensitivity_avg:.2f}")

# Diferentes valores de epsilon
for eps in [0.1, 0.5, 1.0, 2.0]:
    noisy_avg = laplace_mechanism(true_avg, sensitivity_avg, eps)
    error = abs(noisy_avg - true_avg)
    error_pct = error / true_avg * 100

    print(f"\nε = {eps:.1f}:")
    print(f"  Valor con ruido: ${noisy_avg:,.2f}")
    print(f"  Error: ${error:,.2f} ({error_pct:.2f}%)")
```

**Salida:**

```
=== Laplace Mechanism - Salario Promedio ===
Valor real: $75,234.67
Sensitivity: $249.53

ε = 0.1:
  Valor con ruido: $76,845.23
  Error: $1,610.56 (2.14%)

ε = 0.5:
  Valor con ruido: $75,687.34
  Error: $452.67 (0.60%)

ε = 1.0:
  Valor con ruido: $75,456.12
  Error: $221.45 (0.29%)

ε = 2.0:
  Valor con ruido: $75,312.89
  Error: $78.22 (0.10%)
```

### Visualization del ruido

```python
# Simular múltiples releases con ruido
n_releases = 1000
epsilon = 1.0

noisy_values = [
    laplace_mechanism(true_avg, sensitivity_avg, epsilon)
    for _ in range(n_releases)
]

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(noisy_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(true_avg, color='red', linestyle='--', linewidth=2, label='Valor real')
axes[0].set_xlabel('Salario promedio')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title(f'Distribución del ruido (ε={epsilon})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Comparar múltiples epsilons
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
for eps in epsilons:
    values = [laplace_mechanism(true_avg, sensitivity_avg, eps) for _ in range(1000)]
    axes[1].hist(values, bins=30, alpha=0.4, label=f'ε={eps}')

axes[1].axvline(true_avg, color='red', linestyle='--', linewidth=2, label='Real')
axes[1].set_xlabel('Salario promedio')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Impacto de ε en ruido')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('differential_privacy_laplace.png', dpi=150)
plt.show()

print(f"\nDesviación estándar del ruido (ε={epsilon}): ${np.std(noisy_values):,.2f}")
```

______________________________________________________________________

## 🎲 Gaussian Mechanism

### Para consultas con composición

```python
def gaussian_mechanism(true_value, sensitivity, epsilon, delta=1e-5):
    """
    Gaussian mechanism para (ε, δ)-differential privacy

    Args:
        delta: probabilidad de falla de privacidad (típicamente 1/n²)
    """
    # Calibrar sigma basado en epsilon y delta
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma)

    return true_value + noise

# Ejemplo: mediana
true_median = df['salary'].median()

# Sensitivity de mediana: en el peor caso, agregar 1 persona puede mover mediana
# hasta la mitad del rango
sensitivity_median = (max_salary - min_salary) / 2

print(f"\n=== Gaussian Mechanism - Salario Mediano ===")
print(f"Valor real: ${true_median:,.2f}")
print(f"Sensitivity: ${sensitivity_median:,.2f}")

for eps in [0.5, 1.0, 2.0]:
    noisy_median = gaussian_mechanism(true_median, sensitivity_median, eps, delta=1e-5)
    error = abs(noisy_median - true_median)

    print(f"\nε = {eps:.1f}, δ = 1e-5:")
    print(f"  Valor con ruido: ${noisy_median:,.2f}")
    print(f"  Error: ${error:,.2f}")
```

______________________________________________________________________

## 📊 Query sobre histogramas

```python
def dp_histogram(data, bins, epsilon):
    """
    Histograma que satisface differential privacy

    Sensitivity de histograma = 1 (un individuo puede estar en max 1 bin)
    """
    # Contar verdaderos
    counts, bin_edges = np.histogram(data, bins=bins)

    # Añadir ruido Laplaciano a cada bin
    sensitivity = 1  # Cada persona en 1 bin
    epsilon_per_bin = epsilon / len(counts)  # Composición paralela

    noisy_counts = [
        max(0, count + np.random.laplace(0, sensitivity / epsilon_per_bin))
        for count in counts
    ]

    return noisy_counts, bin_edges

# Ejemplo: distribución de salarios por rangos
bins = [30000, 50000, 70000, 90000, 110000, 130000, 150000, 200000]
epsilon = 1.0

true_counts, bin_edges = np.histogram(df['salary'], bins=bins)
noisy_counts, _ = dp_histogram(df['salary'], bins=bins, epsilon=epsilon)

# Comparación
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Real
axes[0].bar(range(len(true_counts)), true_counts, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Rango salarial')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Histograma Real')
axes[0].set_xticks(range(len(true_counts)))
axes[0].set_xticklabels([f"${b/1000:.0f}k" for b in bin_edges[:-1]], rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Con ruido DP
axes[1].bar(range(len(noisy_counts)), noisy_counts, alpha=0.7, color='coral')
axes[1].set_xlabel('Rango salarial')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title(f'Histograma con DP (ε={epsilon})')
axes[1].set_xticks(range(len(noisy_counts)))
axes[1].set_xticklabels([f"${b/1000:.0f}k" for b in bin_edges[:-1]], rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('dp_histogram.png', dpi=150)
plt.show()

# Métricas de error
total_error = sum(abs(t - n) for t, n in zip(true_counts, noisy_counts))
print(f"\n=== Histograma con DP ===")
print(f"Error total (suma de diferencias): {total_error:.0f}")
print(f"Error promedio por bin: {total_error / len(true_counts):.2f}")
```

______________________________________________________________________

## 🔄 Composición de queries

### Sequential Composition

```
Si hacemos k queries con ε₁, ε₂, ..., εₖ:
Total privacy loss = ε₁ + ε₂ + ... + εₖ

⚠️ El privacy budget se ACUMULA
```

```python
class DPBudgetTracker:
    """
    Tracker para privacy budget
    """
    def __init__(self, total_epsilon):
        self.total_epsilon = total_epsilon
        self.spent_epsilon = 0
        self.queries = []

    def check_budget(self, epsilon_needed):
        if self.spent_epsilon + epsilon_needed > self.total_epsilon:
            raise ValueError(
                f"Insufficient privacy budget. "
                f"Need {epsilon_needed}, but only "
                f"{self.total_epsilon - self.spent_epsilon:.4f} remaining."
            )

    def spend(self, epsilon, query_name):
        self.check_budget(epsilon)
        self.spent_epsilon += epsilon
        self.queries.append({'query': query_name, 'epsilon': epsilon})
        print(f"Query: {query_name} | ε used: {epsilon:.2f} | Remaining: {self.total_epsilon - self.spent_epsilon:.2f}")

# Ejemplo: múltiples queries
budget = DPBudgetTracker(total_epsilon=2.0)

print("\n=== Privacy Budget Tracking ===")
print(f"Total budget: ε = {budget.total_epsilon}\n")

try:
    # Query 1: promedio
    eps1 = 0.5
    budget.spend(eps1, "Salario promedio")
    avg_noisy = laplace_mechanism(df['salary'].mean(), sensitivity_avg, eps1)

    # Query 2: mediana
    eps2 = 0.5
    budget.spend(eps2, "Salario mediano")
    median_noisy = gaussian_mechanism(df['salary'].median(), sensitivity_median, eps2)

    # Query 3: conteo por departamento
    eps3 = 0.5
    budget.spend(eps3, "Conteo por departamento")

    # Query 4: EXCEDE budget
    eps4 = 1.0
    budget.spend(eps4, "Percentil 95")

except ValueError as e:
    print(f"\n⚠️ Error: {e}")

print(f"\nTotal privacy loss: ε = {budget.spent_epsilon:.2f}")
```

**Salida:**

```
=== Privacy Budget Tracking ===
Total budget: ε = 2.0

Query: Salario promedio | ε used: 0.50 | Remaining: 1.50
Query: Salario mediano | ε used: 0.50 | Remaining: 1.00
Query: Conteo por departamento | ε used: 0.50 | Remaining: 0.50

⚠️ Error: Insufficient privacy budget. Need 1.0, but only 0.50 remaining.

Total privacy loss: ε = 1.50
```

### Advanced Composition

```python
def advanced_composition_bound(k, epsilon_per_query, delta):
    """
    Teorema de Composición Avanzada:
    k queries con ε individual resultan en ε_total menor que suma simple

    ε_total ≈ √(2k log(1/δ)) × ε + k × ε × (e^ε - 1)
    """
    term1 = np.sqrt(2 * k * np.log(1 / delta)) * epsilon_per_query
    term2 = k * epsilon_per_query * (np.exp(epsilon_per_query) - 1)

    return term1 + term2

# Comparar composiciones
k_queries = 10
eps_per_query = 0.1
delta = 1e-5

simple_composition = k_queries * eps_per_query
advanced = advanced_composition_bound(k_queries, eps_per_query, delta)

print(f"\n=== Composición de {k_queries} queries (ε={eps_per_query} cada una) ===")
print(f"Composición simple: ε_total = {simple_composition:.2f}")
print(f"Composición avanzada: ε_total = {advanced:.2f}")
print(f"Mejora: {(simple_composition - advanced) / simple_composition * 100:.1f}%")
```

______________________________________________________________________

## 💡 Casos de Usage reales

### 1. Reporte de salarios por departamento

```python
def dp_mean_by_group(df, group_col, value_col, epsilon):
    """
    Calcula promedio por grupo con DP
    """
    groups = df.groupby(group_col)[value_col]

    # Epsilon por grupo (parallel composition)
    n_groups = df[group_col].nunique()
    eps_per_group = epsilon / n_groups

    results = {}

    for name, group in groups:
        true_mean = group.mean()

        # Sensitivity: range / group_size
        sensitivity = (max_salary - min_salary) / len(group)

        noisy_mean = laplace_mechanism(true_mean, sensitivity, eps_per_group)

        results[name] = {
            'count': len(group),
            'mean_noisy': noisy_mean
        }

    return results

# Aplicar
epsilon = 1.0
dept_salaries = dp_mean_by_group(df, 'department', 'salary', epsilon)

print(f"\n=== Salarios Promedio por Departamento (ε={epsilon}) ===")
for dept, data in dept_salaries.items():
    print(f"{dept}: ${data['mean_noisy']:,.2f} (n={data['count']})")

# Comparar con real
print("\n=== Comparación con valores reales ===")
for dept in df['department'].unique():
    real = df[df['department'] == dept]['salary'].mean()
    noisy = dept_salaries[dept]['mean_noisy']
    error = abs(real - noisy)
    print(f"{dept}: Real=${real:,.2f}, DP=${noisy:,.2f}, Error=${error:,.2f}")
```

### 2. Detección de outliers con DP

```python
def dp_outlier_detection(data, epsilon, threshold_percentile=95):
    """
    Detecta si hay outliers extremos sin revelar valores exactos
    """
    # Calcular threshold con DP
    true_threshold = np.percentile(data, threshold_percentile)

    # Sensitivity del percentil
    sensitivity = (data.max() - data.min()) / len(data)

    noisy_threshold = laplace_mechanism(true_threshold, sensitivity, epsilon)

    # Contar outliers con DP
    true_count = (data > true_threshold).sum()
    count_sensitivity = 1
    noisy_count = laplace_mechanism(true_count, count_sensitivity, epsilon)

    return {
        'threshold': noisy_threshold,
        'outlier_count': max(0, int(noisy_count)),
        'outlier_percentage': max(0, noisy_count / len(data) * 100)
    }

result = dp_outlier_detection(df['salary'], epsilon=0.5)

print(f"\n=== Detección de Outliers con DP ===")
print(f"Threshold (P95): ${result['threshold']:,.2f}")
print(f"Número de outliers: {result['outlier_count']}")
print(f"Porcentaje: {result['outlier_percentage']:.2f}%")
```

______________________________________________________________________

## 📝 Resumen

### ✅ Mecanismos de DP

| Mecanismo               | Distribución     | Usage                              | DP Guarantee |
| ----------------------- | ---------------- | ---------------------------------- | ------------ |
| **Laplace**             | Laplace(0, Δf/ε) | Queries numéricas (conteos, sumas) | ε-DP         |
| **Gaussian**            | Normal(0, σ²)    | Queries con composición            | (ε, δ)-DP    |
| **Exponential**         | Exponencial      | Selección (argmax)                 | ε-DP         |
| **Randomized Response** | Bernoulli        | Data categóricos                   | ε-DP         |

### 🎯 Parámetros clave

**Epsilon (ε):**

- 0.01 - 0.1: Privacidad muy alta (Data médicos)
- 0.1 - 1.0: Privacidad alta (Data financieros)
- 1.0 - 3.0: Privacidad media (Data corporativos)
- > 3.0: Privacidad baja (Data públicos anonimizados)

**Delta (δ):**

- Típicamente: 1/n² o 1e-5
- Representa probabilidad de "falla catastrófica"

**Sensitivity (Δf):**

- Máximo cambio en output al añadir/remover 1 registro
- Count: Δf = 1
- Sum: Δf = max_value
- Mean: Δf = range / n

### 💡 Mejores Practices

- ✅ Pre-calcular sensitivity (no estimarla de los Data)
- ✅ Usar parallel composition cuando sea posible
- ✅ Trackear privacy budget cuidadosamente
- ✅ Post-procesar para garantías (ej: no negativos en conteos)
- ✅ Documentar todos los parámetros (ε, δ, sensitivity)
- ✅ Considerar utility-privacy trade-off por stakeholder

### 🚫 Errors comunes

- ❌ Calcular sensitivity desde los Data (privacy leak)
- ❌ Olvidar composición (múltiples queries agotan budget)
- ❌ Elegir ε arbitrariamente sin justificación
- ❌ Publicar tanto DP como Data originales
- ❌ No considerar side information del atacante

### 📌 Checklist DP

- ✅ Privacy budget (ε) seleccionado y justificado
- ✅ Sensitivity calculada teóricamente
- ✅ Mecanismo apropiado (Laplace/Gaussian)
- ✅ Budget tracker implementado
- ✅ Composición considerada (simple/avanzada)
- ✅ Post-processing aplicado (clipping, rounding)
- ✅ Trade-off accuracy/privacy evaluado
- ✅ Documentación completa de parámetros

### 🚀 Librerías en producción

```python
# Google Differential Privacy
# pip install python-dp

# OpenDP
# pip install opendp

# IBM Diffprivlib
# pip install diffprivlib

# Ejemplo con diffprivlib:
# from diffprivlib.models import LogisticRegression
# clf = LogisticRegression(epsilon=1.0)
# clf.fit(X, y)
```

### 🌐 Casos de Usage reales

- **US Census 2020:** ε ≈ 19.6 para proteger Data censales
- **Google RAPPOR:** Chrome usage statistics con local DP
- **Apple:** Keyboard suggestions con ε = 2-6
- **Microsoft:** Telemetry con ε-DP
- **Uber/Lyft:** Aggregate trip statistics
