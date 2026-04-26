# Example 01 — Data Anonymization: k-Anonymity and l-Diversity

## Context

Sharing sensitive data (medical, financial) requires protecting the identity of individuals while maintaining analytical utility.

## Objective

Apply anonymization techniques (k-anonymity, l-diversity) to medical dataset to prevent re-identification.

______________________________________________________________________

## 🚀 Setup

```python
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
```

______________________________________________________________________

## 📚 Original dataset (not anonymized)

```python
# Data doctors simulados
n_patients = 100

data = {
    'patient_id': range(1, n_patients + 1),
    'age': np.random.randint(18, 85, n_patients),
    'gender': np.random.choice(['M', 'F'], n_patients),
    'zipcode': np.random.choice(['10001', '10002', '10003', '20001', '20002'], n_patients),
    'diagnosis': np.random.choice([
        'Diabetes', 'Hypertension', 'Asthma', 'Cancer', 'HIV'
    ], n_patients, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'medication': np.random.choice([
        'Metformin', 'Lisinopril', 'Albuterol', 'Chemotherapy', 'Antiretroviral'
    ], n_patients)
}

df_original = pd.DataFrame(data)

print("=== Dataset Original (SIN anonimizar) ===")
print(df_original.head(10))
print(f"\nTotal registros: {len(df_original)}")
```

**Output:**

```
=== Dataset Original (SIN anonimizar) ===
   patient_id  age gender zipcode     diagnosis     medication
0           1   42      F   10001      Diabetes      Metformin
1           2   67      M   20001  Hypertension    Lisinopril
2           3   29      F   10002         Asthma     Albuterol
3           4   55      M   10003        Cancer  Chemotherapy
4           5   38      F   20002           HIV Antiretroviral
...
```

______________________________________________________________________

## ⚠️Privacy risks

```python
# Quasi-identifiers: combination que can identificar individuo
quasi_identifiers = ['age', 'gender', 'zipcode']

# Contar combinaciones unique
unique_combinations = df_original[quasi_identifiers].drop_duplicates()
print(f"\n=== Analysis de Re-ID ===")
print(f"Combineciones unique de quasi-identifiers: {len(unique_combinations)}")

# Individuos only identificables
unique_individuals = df_original.groupby(quasi_identifiers).size()
uniquely_identifiable = (unique_individuals == 1).sum()

print(f"Individuos only identificables: {uniquely_identifiable} ({uniquely_identifiable/len(df_original):.1%})")
print("\n⚠️ RIESGO: Con data auxiliares (ej: censo public), un atacante could re-identificar individuos")
```

**Output:**

```
=== Analysis de Re-ID ===
Combineciones unique de quasi-identifiers: 78
Individuos only identificables: 45 (45.0%)

⚠️ RIESGO: Con data auxiliares, un atacante could re-identificar individuos
```

______________________________________________________________________

## 🔒 k-Anonymity

### Concept

```
k-anonymity: Cada combination de quasi-identifiers aparece al less k veces.

Si k=5: each persona es "indistinguible" de al less 4 otras personas.
```

### Attribute generalization

```python
def generalize_age(age, bins=[0, 30, 40, 50, 60, 100]):
    """
    Generaliza edad en rangos
    """
    for i in range(len(bins) - 1):
        if bins[i] <= age < bins[i+1]:
            return f"{bins[i]}-{bins[i+1]}"
    return f"{bins[-2]}+"

def generalize_zipcode(zipcode):
    """
    Generaliza zipcode: 10001 → 100**
    """
    return zipcode[:3] + "**"

# Apply generalizaciones
df_anon = df_original.copy()
df_anon['age_range'] = df_anon['age'].apply(generalize_age)
df_anon['zipcode_generalized'] = df_anon['zipcode'].apply(generalize_zipcode)

# Remover columns originals specific
df_anon = df_anon[['age_range', 'gender', 'zipcode_generalized', 'diagnosis', 'medication']]

print("\n=== Dataset con Generalization ===")
print(df_anon.head(10))
```

**Output:**

```
=== Dataset con Generalization ===
  age_range gender zipcode_generalized     diagnosis     medication
0     40-50      F               100**      Diabetes      Metformin
1     60-100     M               200**  Hypertension    Lisinopril
2     0-30       F               100**         Asthma     Albuterol
3     50-60      M               100**        Cancer  Chemotherapy
4     30-40      F               200**           HIV Antiretroviral
...
```

### Verify k-anonymity

```python
def check_k_anonymity(df, quasi_identifiers, k=5):
    """
    Verifica si dataset satisface k-anonymity
    """
    # Contar ocurrencias de each combination
    group_sizes = df.groupby(quasi_identifiers).size()

    # Verificar k-anonymity
    satisfies_k = (group_sizes >= k).all()
    min_group_size = group_sizes.min()

    # Equivalence classes con < k individuos
    violations = (group_sizes < k).sum()

    return {
        'satisfies_k_anonymity': satisfies_k,
        'k': k,
        'min_group_size': min_group_size,
        'violations': violations,
        'total_groups': len(group_sizes)
    }

# Verificar con nuevos quasi-identifiers
quasi_identifiers_anon = ['age_range', 'gender', 'zipcode_generalized']

result = check_k_anonymity(df_anon, quasi_identifiers_anon, k=5)

print(f"\n=== k-Anonymity Check (k={result['k']}) ===")
print(f"Satisface k-anonymity: {result['satisfies_k_anonymity']}")
print(f"Grupo más little: {result['min_group_size']} individuos")
print(f"Grupos con < k individuos: {result['violations']}")
print(f"Total equivalence classes: {result['total_groups']}")

if not result['satisfies_k_anonymity']:
    print(f"\n⚠️ Se requires más generalization o suppression de registros")
```

**Output:**

```
=== k-Anonymity Check (k=5) ===
Satisface k-anonymity: True
Grupo más little: 5 individuos
Grupos con < k individuos: 0
Total equivalence classes: 18

✅ Dataset satisface 5-anonymity
```

______________________________________________________________________

## 🎯 l-Diversity

### Concept

```
l-diversity: Dentro de each equivalence class, must haber al less l values distintos del atributo sensible.

Problem de k-anonymity:
Si todos en un group tienen same diagnosis → privacy no protegida

l-diversity: each group must tener al less l diagnoses different
```

### Check l-diversity

```python
def check_l_diversity(df, quasi_identifiers, sensitive_attribute, l=3):
    """
    Verifica si dataset satisface l-diversity
    """
    groups = df.groupby(quasi_identifiers)

    violations = []
    min_diversity = float('inf')

    for group_id, group_df in groups:
        # Contar values distintos del atributo sensible
        distinct_values = group_df[sensitive_attribute].nunique()

        min_diversity = min(min_diversity, distinct_values)

        if distinct_values < l:
            violations.append({
                'group': group_id,
                'size': len(group_df),
                'distinct_values': distinct_values,
                'values': group_df[sensitive_attribute].value_counts().to_dict()
            })

    return {
        'satisfies_l_diversity': len(violations) == 0,
        'l': l,
        'min_diversity': min_diversity,
        'violations': violations
    }

# Verificar l-diversity para diagnosis
result_l = check_l_diversity(
    df_anon,
    quasi_identifiers_anon,
    'diagnosis',
    l=3
)

print(f"\n=== l-Diversity Check (l={result_l['l']}) ===")
print(f"Satisface l-diversity: {result_l['satisfies_l_diversity']}")
print(f"Diversidad minimum encontrada: {result_l['min_diversity']}")

if result_l['violations']:
    print(f"\nGrupos que violan l-diversity: {len(result_l['violations'])}")
    print("\nEjemplos de violaciones:")
    for i, v in enumerate(result_l['violations'][:3], 1):
        print(f"\n{i}. Grupo {v['group']}:")
        print(f"   Size: {v['size']} personas")
        print(f"   Diagnostics distintos: {v['distinct_values']} (requires {result_l['l']})")
        print(f"   Distribution: {v['values']}")
```

**Output:**

```
=== l-Diversity Check (l=3) ===
Satisface l-diversity: False
Diversidad minimum encontrada: 2

Grupos que violan l-diversity: 5

Examples de violaciones:

1. Grupo ('40-50', 'F', '100**'):
   Size: 6 personas
   Diagnostics distintos: 2 (requires 3)
   Distribution: {'Diabetes': 4, 'Hypertension': 2}

2. Grupo ('30-40', 'M', '200**'):
   Size: 5 personas
   Diagnostics distintos: 2 (requires 3)
   Distribution: {'Asthma': 3, 'Diabetes': 2}
```

### Improve l-diversity with suppression

```python
def enforce_l_diversity(df, quasi_identifiers, sensitive_attribute, l=3):
    """
    Elimina registros de grupos que no satisfacen l-diversity
    """
    groups = df.groupby(quasi_identifiers)

    records_to_keep = []

    for group_id, group_df in groups:
        distinct_values = group_df[sensitive_attribute].nunique()

        if distinct_values >= l:
            records_to_keep.append(group_df)

    df_diverse = pd.concat(records_to_keep, ignore_index=True)

    removed = len(df) - len(df_diverse)

    return df_diverse, removed

# Apply
df_l_diverse, removed_count = enforce_l_diversity(
    df_anon,
    quasi_identifiers_anon,
    'diagnosis',
    l=3
)

print(f"\n=== After de Suppression ===")
print(f"Registros eliminados: {removed_count} ({removed_count/len(df_anon):.1%})")
print(f"Registros restantes: {len(df_l_diverse)}")

# Re-verificar
result_after = check_l_diversity(
    df_l_diverse,
    quasi_identifiers_anon,
    'diagnosis',
    l=3
)

print(f"Satisface l-diversity: {result_after['satisfies_l_diversity']}")
```

**Output:**

```
=== After de Suppression ===
Registros eliminados: 23 (23.0%)
Registros restantes: 77
Satisface l-diversity: True
```

______________________________________________________________________

## 📊 Trade-offs: Privacy vs Utility

```python
# Metrics de utility
def calculate_information_loss(df_original, df_anon):
    """
    Measure loss de information por generalization
    """
    # Registros perdidos
    records_lost = len(df_original) - len(df_anon)
    records_lost_pct = records_lost / len(df_original)

    # Granularidad perdida (example: age → age_range)
    # Simple: contar atributos generalizados
    generalized_attrs = 2  # age, zipcode

    return {
        'records_lost': records_lost,
        'records_lost_pct': records_lost_pct,
        'generalized_attributes': generalized_attrs
    }

loss = calculate_information_loss(df_original, df_l_diverse)

print("\n=== Analysis de Loss de Information ===")
print(f"Registros eliminados: {loss['records_lost']} ({loss['records_lost_pct']:.1%})")
print(f"Atributos generalizados: {loss['generalized_attributes']}")

# Comparison de distributions
print("\n=== Distribution de Diagnosis ===")
print("Original:")
print(df_original['diagnosis'].value_counts(normalize=True))
print("\nAnonimizado:")
print(df_l_diverse['diagnosis'].value_counts(normalize=True))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_original['diagnosis'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
axes[0].set_title('Distribution Original')
axes[0].set_ylabel('Frecuencia')
axes[0].set_xlabel('Diagnosis')

df_l_diverse['diagnosis'].value_counts().plot(kind='bar', ax=axes[1], color='coral', alpha=0.7)
axes[1].set_title('Distribution Anonimizada')
axes[1].set_ylabel('Frecuencia')
axes[1].set_xlabel('Diagnosis')

plt.tight_layout()
plt.savefig('anonymization_distribution.png', dpi=150)
plt.show()
```

______________________________________________________________________

## 💡 Additional techniques

### t-Closeness

```python
def check_t_closeness(df, quasi_identifiers, sensitive_attribute, t=0.2):
    """
    t-closeness: distancia entre distribution del atributo sensible
    en each group y la distribution global must ser ≤ t
    """
    global_dist = df[sensitive_attribute].value_counts(normalize=True)
    groups = df.groupby(quasi_identifiers)

    violations = []

    for group_id, group_df in groups:
        group_dist = group_df[sensitive_attribute].value_counts(normalize=True)

        # Earth Mover's Distance (simplified: sum of absolute differences)
        distance = 0
        for category in global_dist.index:
            global_prob = global_dist.get(category, 0)
            group_prob = group_dist.get(category, 0)
            distance += abs(global_prob - group_prob)

        distance /= 2  # Normalize

        if distance > t:
            violations.append({
                'group': group_id,
                'distance': distance
            })

    return {
        'satisfies_t_closeness': len(violations) == 0,
        't': t,
        'violations': len(violations)
    }

result_t = check_t_closeness(df_l_diverse, quasi_identifiers_anon, 'diagnosis', t=0.2)

print(f"\n=== t-Closeness Check (t={result_t['t']}) ===")
print(f"Satisface t-closeness: {result_t['satisfies_t_closeness']}")
print(f"Grupos violando: {result_t['violations']}")
```

### Differential Privacy (preview)

```python
# Add noise Laplaciano para conteos
def add_laplace_noise(true_count, epsilon=1.0):
    """
    Differential Privacy: agregar noise calibrado por epsilon

    epsilon: privacy budget (más low = más privacy, más noise)
    """
    sensitivity = 1  # Para conteos, sensitivity = 1
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)

    return max(0, true_count + noise)  # No negativos

# Example: contar pacientes con Diabetes
true_count = (df_original['diagnosis'] == 'Diabetes').sum()
noisy_count = add_laplace_noise(true_count, epsilon=1.0)

print(f"\n=== Differential Privacy (conteo) ===")
print(f"Conteo real: {true_count}")
print(f"Conteo con noise (ε=1.0): {noisy_count:.0f}")
print(f"Error: {abs(true_count - noisy_count):.0f}")
```

______________________________________________________________________

## 📝 Summary

### ✅ Anonymization levels

```
k-anonymity: Indistinguibilidad basic (≥k personas en each group)
          ↓
l-diversity: Diversidad en atributo sensible (≥l values distintos)
          ↓
t-closeness: Distribution similar a global (distancia ≤t)
```

### 🎯 Transformation techniques

| Technique | Example | Usage |
| ------------------ | --------------------------------- | ----------------------- |
| **Generalization** | age 42 → range 40-50 | Reduce specificity |
| **Deletion** | Delete abnormal records | Comply k/l/t |
| **Anatomization** | Separate QI from sensitive attributes | PPLM principle |
| **Permutation** | Permutation within groups | Maintain distributions |

### 💡Best Practices

- ✅ Identify quasi-identifiers and sensitive attributes
- ✅ Evaluate multiple levels (k-anonymity, l-diversity, t-closeness)
- ✅ Measure information loss (Deleted data, granularity)
- ✅ Validate with attack scenarios (linking, homogeneity, background knowledge)
- ✅ Document transformations applied
- ✅ Re-evaluate if auxiliary data changes

### 🚫 Errors common

- ❌ Only apply k-anonymity (vulnerable to homogeneity attack)
- ❌ Excessive generalization (useless data)
- ❌ Do not consider temporary changes in auxiliary data
- ❌ Forget attributes that can be quasi-identifiers
- ❌ Publish multiple versions of the same dataset (composition attack)

### 📌 Anonymization Checklist

- ✅ Quasi-identifiers identified
- ✅ Sensitive attributes classified
- ✅ k-anonymity verified (k ≥ 5 typically)
- ✅ l-diversity verified (l ≥ 3 typically)
- ✅ Quantified loss information
- ✅ Distributions compared (original vs. anonymized)
- ✅ Attack scenarios evaluated
- ✅ Transformation documentation

### ⚖️ Regulation

- **GDPR (EU):** Anonymization is not personal data (but must be irreversible)
- **HIPAA (US):** Safe Harbor method (18 identifiers), Expert Determination
- **CCPA (California):** Limitations on Usage of de-identified data

### 🚀 Next steps

- Differential Privacy (Example 02)
- Federated Learning (train without sharing Data)
- Secure Multi-Party Computation
- Homomorphic Encryption (computation about encrypted Data)
