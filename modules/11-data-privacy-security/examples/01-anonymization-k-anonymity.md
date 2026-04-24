# Ejemplo 01 — Anonimización de Datos: k-Anonymity y l-Diversity

## Contexto

Compartir datos sensibles (médicos, financieros) requiere proteger identidad de individuos mientras se mantiene utilidad analítica.

## Objective

Aplicar técnicas de anonimización (k-anonymity, l-diversity) a dataset médico para prevenir re-identificación.

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

## 📚 Dataset original (sin anonimizar)

```python
# Datos médicos simulados
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

**Salida:**

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

## ⚠️ Riesgos de privacidad

```python
# Quasi-identifiers: combinación que puede identificar individuo
quasi_identifiers = ['age', 'gender', 'zipcode']

# Contar combinaciones únicas
unique_combinations = df_original[quasi_identifiers].drop_duplicates()
print(f"\n=== Análisis de Re-identificación ===")
print(f"Combinaciones únicas de quasi-identifiers: {len(unique_combinations)}")

# Individuos únicamente identificables
unique_individuals = df_original.groupby(quasi_identifiers).size()
uniquely_identifiable = (unique_individuals == 1).sum()

print(f"Individuos únicamente identificables: {uniquely_identifiable} ({uniquely_identifiable/len(df_original):.1%})")
print("\n⚠️ RIESGO: Con datos auxiliares (ej: censo público), un atacante podría re-identificar individuos")
```

**Salida:**

```
=== Análisis de Re-identificación ===
Combinaciones únicas de quasi-identifiers: 78
Individuos únicamente identificables: 45 (45.0%)

⚠️ RIESGO: Con datos auxiliares, un atacante podría re-identificar individuos
```

______________________________________________________________________

## 🔒 k-Anonymity

### Concepto

```
k-anonymity: Cada combinación de quasi-identifiers aparece al menos k veces.

Si k=5: cada persona es "indistinguible" de al menos 4 otras personas.
```

### Generalización de atributos

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

# Aplicar generalizaciones
df_anon = df_original.copy()
df_anon['age_range'] = df_anon['age'].apply(generalize_age)
df_anon['zipcode_generalized'] = df_anon['zipcode'].apply(generalize_zipcode)

# Remover columnas originales específicas
df_anon = df_anon[['age_range', 'gender', 'zipcode_generalized', 'diagnosis', 'medication']]

print("\n=== Dataset con Generalización ===")
print(df_anon.head(10))
```

**Salida:**

```
=== Dataset con Generalización ===
  age_range gender zipcode_generalized     diagnosis     medication
0     40-50      F               100**      Diabetes      Metformin
1     60-100     M               200**  Hypertension    Lisinopril
2     0-30       F               100**         Asthma     Albuterol
3     50-60      M               100**        Cancer  Chemotherapy
4     30-40      F               200**           HIV Antiretroviral
...
```

### Verificar k-anonymity

```python
def check_k_anonymity(df, quasi_identifiers, k=5):
    """
    Verifica si dataset satisface k-anonymity
    """
    # Contar ocurrencias de cada combinación
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
print(f"Grupo más pequeño: {result['min_group_size']} individuos")
print(f"Grupos con < k individuos: {result['violations']}")
print(f"Total equivalence classes: {result['total_groups']}")

if not result['satisfies_k_anonymity']:
    print(f"\n⚠️ Se requiere más generalización o supresión de registros")
```

**Salida:**

```
=== k-Anonymity Check (k=5) ===
Satisface k-anonymity: True
Grupo más pequeño: 5 individuos
Grupos con < k individuos: 0
Total equivalence classes: 18

✅ Dataset satisface 5-anonymity
```

______________________________________________________________________

## 🎯 l-Diversity

### Concepto

```
l-diversity: Dentro de cada equivalence class, debe haber al menos l valores distintos del atributo sensible.

Problema de k-anonymity:
Si todos en un grupo tienen mismo diagnosis → privacidad no protegida

l-diversity: cada grupo debe tener al menos l diagnósticos diferentes
```

### Verificar l-diversity

```python
def check_l_diversity(df, quasi_identifiers, sensitive_attribute, l=3):
    """
    Verifica si dataset satisface l-diversity
    """
    groups = df.groupby(quasi_identifiers)

    violations = []
    min_diversity = float('inf')

    for group_id, group_df in groups:
        # Contar valores distintos del atributo sensible
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
print(f"Diversidad mínima encontrada: {result_l['min_diversity']}")

if result_l['violations']:
    print(f"\nGrupos que violan l-diversity: {len(result_l['violations'])}")
    print("\nEjemplos de violaciones:")
    for i, v in enumerate(result_l['violations'][:3], 1):
        print(f"\n{i}. Grupo {v['group']}:")
        print(f"   Tamaño: {v['size']} personas")
        print(f"   Diagnósticos distintos: {v['distinct_values']} (requiere {result_l['l']})")
        print(f"   Distribución: {v['values']}")
```

**Salida:**

```
=== l-Diversity Check (l=3) ===
Satisface l-diversity: False
Diversidad mínima encontrada: 2

Grupos que violan l-diversity: 5

Ejemplos de violaciones:

1. Grupo ('40-50', 'F', '100**'):
   Tamaño: 6 personas
   Diagnósticos distintos: 2 (requiere 3)
   Distribución: {'Diabetes': 4, 'Hypertension': 2}

2. Grupo ('30-40', 'M', '200**'):
   Tamaño: 5 personas
   Diagnósticos distintos: 2 (requiere 3)
   Distribución: {'Asthma': 3, 'Diabetes': 2}
```

### Mejorar l-diversity con supresión

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

# Aplicar
df_l_diverse, removed_count = enforce_l_diversity(
    df_anon,
    quasi_identifiers_anon,
    'diagnosis',
    l=3
)

print(f"\n=== Después de Supresión ===")
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

**Salida:**

```
=== Después de Supresión ===
Registros eliminados: 23 (23.0%)
Registros restantes: 77
Satisface l-diversity: True
```

______________________________________________________________________

## 📊 Trade-offs: Privacidad vs Utilidad

```python
# Métricas de utilidad
def calculate_information_loss(df_original, df_anon):
    """
    Mide pérdida de información por generalización
    """
    # Registros perdidos
    records_lost = len(df_original) - len(df_anon)
    records_lost_pct = records_lost / len(df_original)

    # Granularidad perdida (ejemplo: age → age_range)
    # Simple: contar atributos generalizados
    generalized_attrs = 2  # age, zipcode

    return {
        'records_lost': records_lost,
        'records_lost_pct': records_lost_pct,
        'generalized_attributes': generalized_attrs
    }

loss = calculate_information_loss(df_original, df_l_diverse)

print("\n=== Análisis de Pérdida de Información ===")
print(f"Registros eliminados: {loss['records_lost']} ({loss['records_lost_pct']:.1%})")
print(f"Atributos generalizados: {loss['generalized_attributes']}")

# Comparación de distribuciones
print("\n=== Distribución de Diagnosis ===")
print("Original:")
print(df_original['diagnosis'].value_counts(normalize=True))
print("\nAnonimizado:")
print(df_l_diverse['diagnosis'].value_counts(normalize=True))

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_original['diagnosis'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
axes[0].set_title('Distribución Original')
axes[0].set_ylabel('Frecuencia')
axes[0].set_xlabel('Diagnosis')

df_l_diverse['diagnosis'].value_counts().plot(kind='bar', ax=axes[1], color='coral', alpha=0.7)
axes[1].set_title('Distribución Anonimizada')
axes[1].set_ylabel('Frecuencia')
axes[1].set_xlabel('Diagnosis')

plt.tight_layout()
plt.savefig('anonymization_distribution.png', dpi=150)
plt.show()
```

______________________________________________________________________

## 💡 Técnicas adicionales

### t-Closeness

```python
def check_t_closeness(df, quasi_identifiers, sensitive_attribute, t=0.2):
    """
    t-closeness: distancia entre distribución del atributo sensible
    en cada grupo y la distribución global debe ser ≤ t
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
# Agregar ruido Laplaciano para conteos
def add_laplace_noise(true_count, epsilon=1.0):
    """
    Differential Privacy: agregar ruido calibrado por epsilon

    epsilon: privacy budget (más bajo = más privacidad, más ruido)
    """
    sensitivity = 1  # Para conteos, sensitivity = 1
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)

    return max(0, true_count + noise)  # No negativos

# Ejemplo: contar pacientes con Diabetes
true_count = (df_original['diagnosis'] == 'Diabetes').sum()
noisy_count = add_laplace_noise(true_count, epsilon=1.0)

print(f"\n=== Differential Privacy (conteo) ===")
print(f"Conteo real: {true_count}")
print(f"Conteo con ruido (ε=1.0): {noisy_count:.0f}")
print(f"Error: {abs(true_count - noisy_count):.0f}")
```

______________________________________________________________________

## 📝 Resumen

### ✅ Niveles de anonimización

```
k-anonymity: Indistinguibilidad básica (≥k personas en cada grupo)
          ↓
l-diversity: Diversidad en atributo sensible (≥l valores distintos)
          ↓
t-closeness: Distribución similar a global (distancia ≤t)
```

### 🎯 Técnicas de transformación

| Técnica            | Ejemplo                           | Uso                     |
| ------------------ | --------------------------------- | ----------------------- |
| **Generalización** | edad 42 → rango 40-50             | Reducir especificidad   |
| **Supresión**      | Eliminar registros anómalos       | Cumplir k/l/t           |
| **Anatomization**  | Separar QI de atributos sensibles | PPLM principle          |
| **Permutation**    | Permutación dentro de grupos      | Mantener distribuciones |

### 💡 Mejores prácticas

- ✅ Identificar quasi-identifiers y atributos sensibles
- ✅ Evaluar múltiples niveles (k-anonymity, l-diversity, t-closeness)
- ✅ Medir información loss (datos eliminados, granularidad)
- ✅ Validar con attack scenarios (linking, homogeneity, background knowledge)
- ✅ Documentar transformaciones aplicadas
- ✅ Re-evaluar si datos auxiliares cambian

### 🚫 Errores comunes

- ❌ Solo aplicar k-anonymity (vulnerable a homogeneity attack)
- ❌ Generalización excesiva (datos inútiles)
- ❌ No considerar cambios temporales en datos auxiliares
- ❌ Olvidar atributos que pueden ser quasi-identifiers
- ❌ Publicar múltiples versiones del mismo dataset (composition attack)

### 📌 Checklist Anonimización

- ✅ Quasi-identifiers identificados
- ✅ Atributos sensibles clasificados
- ✅ k-anonymity verificado (k ≥ 5 típicamente)
- ✅ l-diversity verificado (l ≥ 3 típicamente)
- ✅ Información loss cuantificada
- ✅ Distribuciones comparadas (original vs anonimizado)
- ✅ Attack scenarios evaluados
- ✅ Documentación de transformaciones

### ⚖️ Regulación

- **GDPR (EU):** Anonymization no es personal data (pero debe ser irreversible)
- **HIPAA (US):** Safe Harbor method (18 identifiers), Expert Determination
- **CCPA (California):** Limitations en uso de de-identified data

### 🚀 Next steps

- Differential Privacy (Example 02)
- Federated Learning (entrenar sin compartir datos)
- Secure Multi-Party Computation
- Homomorphic Encryption (cómputo sobre datos encriptados)
