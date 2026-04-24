# Example 02 — Explainability con SHAP y LIME

## Contexto

Models complejos (Random Forest, XGBoost, Neural Nets) son "black boxes". SHAP y LIME proporcionan explicaciones de por qué el Model predice X.

## Objective

Explicar Predictions de Model de Classification usando SHAP (global + local) y LIME (local).

______________________________________________________________________

## 🚀 Setup

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

# Deshabilitar warnings
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
```

______________________________________________________________________

## 📚 Dataset

```python
# Simulación: Predicción de churn en telecomunicaciones
n_samples = 1000

data = {
    'tenure_months': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'total_charges': np.random.uniform(100, 8000, n_samples),
    'contract_type': np.random.choice([0, 1, 2], n_samples),  # 0=month, 1=year, 2=2years
    'internet_service': np.random.choice([0, 1, 2], n_samples),  # 0=no, 1=DSL, 2=fiber
    'tech_support': np.random.choice([0, 1], n_samples),
    'num_services': np.random.randint(0, 6, n_samples),
    'payment_auto': np.random.choice([0, 1], n_samples),
}

df = pd.DataFrame(data)

# Target: churn (cliente abandona servicio)
def generate_churn(row):
    """
    Lógica: churn más probable si:
    - tenure bajo
    - monthly_charges alto
    - contract month-to-month
    - no tech support
    """
    score = 0

    score += (72 - row['tenure_months']) / 72 * 3  # tenure bajo = +3
    score += row['monthly_charges'] / 120 * 2      # charges altos = +2
    score += 1.5 if row['contract_type'] == 0 else 0  # month-to-month = +1.5
    score += 1 if row['tech_support'] == 0 else 0  # no support = +1
    score -= row['num_services'] * 0.3             # más servicios = menos churn

    # Ruido
    score += np.random.randn() * 0.5

    return 1 if score > 3 else 0

df['churn'] = df.apply(generate_churn, axis=1)

print(f"Dataset: {len(df)} clientes")
print(f"Tasa de churn: {df['churn'].mean():.2%}")
```

**Salida:**

```
Dataset: 1000 clientes
Tasa de churn: 34.20%
```

______________________________________________________________________

## 🤖 Entrenar Model

```python
# Features
X = df.drop('churn', axis=1)
y = df['churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modelo (Random Forest = black box)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")

# Feature importance básico
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (global):")
print(feature_importance)
```

**Salida:**

```
Accuracy: 0.8850

Feature Importance (global):
           feature  importance
0   tenure_months    0.285432
1  monthly_charges   0.234123
4  internet_service  0.156789
5    contract_type   0.134567
2   total_charges    0.089234
...
```

______________________________________________________________________

## 🔍 SHAP (SHapley Additive exPlanations)

### Explicación global

```python
# Crear explainer
explainer = shap.TreeExplainer(model)

# Calcular SHAP values (muestra del test set)
X_sample = X_test.sample(100, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Para clasificación binaria: shap_values[1] = clase positiva (churn)
shap_values_churn = shap_values[1]

print("SHAP values calculados")
print(f"Shape: {shap_values_churn.shape}")  # (100 samples, 8 features)
```

### Summary plot

```python
# Visualización global: importancia + impacto
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_churn, X_sample, plot_type="dot", show=False)
plt.title("SHAP Summary Plot - Global Feature Importance")
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Interpretación:
# - Eje X: SHAP value (impacto en predicción)
# - Color: valor de feature (rojo=alto, azul=bajo)
# - Posición Y: feature ordenada por importancia
```

### Bar plot (importancia promedio)

```python
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_churn, X_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - Mean |SHAP value|")
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Explicación local (un cliente específico)

```python
# Cliente de alto riesgo
idx = 0  # Primera muestra
client = X_sample.iloc[idx]

print(f"\n=== Cliente #{idx} ===")
print(client)
print(f"\nPredicción: {'CHURN' if model.predict([client])[0] == 1 else 'NO CHURN'}")
print(f"Probabilidad churn: {model.predict_proba([client])[0][1]:.2%}")

# SHAP values para este cliente
shap_value_client = shap_values_churn[idx]

# Waterfall plot: contribución de cada feature
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_value_client,
        base_values=explainer.expected_value[1],
        data=client.values,
        feature_names=X.columns.tolist()
    ),
    show=False
)
plt.title(f"SHAP Waterfall - Cliente #{idx}")
plt.tight_layout()
plt.savefig(f'shap_waterfall_client_{idx}.png', dpi=150, bbox_inches='tight')
plt.show()

# Force plot (alternativa visual)
shap.force_plot(
    explainer.expected_value[1],
    shap_value_client,
    client,
    matplotlib=True,
    show=False
)
plt.savefig(f'shap_force_client_{idx}.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Interpretación:**

```
Cliente #0:
tenure_months: 6 meses → +0.15 SHAP (bajo tenure aumenta churn)
monthly_charges: $95 → +0.08 SHAP (alto cargo aumenta churn)
contract_type: 0 (month-to-month) → +0.12 SHAP (contrato flexible aumenta churn)
tech_support: 0 (no tiene) → +0.06 SHAP (sin soporte aumenta churn)

Base value (media): 0.342
Predicción final: 0.342 + 0.15 + 0.08 + ... = 0.72 → CHURN
```

### Dependence plot (interacciones)

```python
# Cómo tenure_months afecta predicción
plt.figure(figsize=(10, 6))
shap.dependence_plot(
    "tenure_months",
    shap_values_churn,
    X_sample,
    interaction_index="contract_type",
    show=False
)
plt.title("SHAP Dependence: tenure_months (coloreado por contract_type)")
plt.tight_layout()
plt.savefig('shap_dependence_tenure.png', dpi=150, bbox_inches='tight')
plt.show()

# Interpretación:
# - tenure bajo + contract month-to-month = SHAP muy positivo (alto riesgo)
# - tenure alto = SHAP negativo (bajo riesgo)
```

______________________________________________________________________

## 🔬 LIME (Local Interpretable Model-agnostic Explanations)

```python
# Crear explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['No Churn', 'Churn'],
    mode='classification',
    random_state=42
)

# Explicar mismo cliente
idx = 0
client_values = X_sample.iloc[idx].values

# Generar explicación
lime_exp = lime_explainer.explain_instance(
    data_row=client_values,
    predict_fn=model.predict_proba,
    num_features=8
)

# Mostrar explicación
print(f"\n=== LIME Explanation - Cliente #{idx} ===")
print(f"Predicción: {lime_exp.predict_proba}")

# Visualización
fig = lime_exp.as_pyplot_figure()
plt.title(f"LIME Explanation - Cliente #{idx}")
plt.tight_layout()
plt.savefig(f'lime_client_{idx}.png', dpi=150, bbox_inches='tight')
plt.show()

# Lista de contribuciones
print("\nContribuciones LIME:")
for feature, weight in lime_exp.as_list():
    print(f"{feature}: {weight:+.4f}")
```

**Salida:**

```
=== LIME Explanation - Cliente #0 ===
Predicción: [0.28, 0.72]  # 72% probabilidad de churn

Contribuciones LIME:
tenure_months <= 12: +0.18
monthly_charges > 80: +0.12
contract_type = 0: +0.15
tech_support = 0: +0.08
...
```

______________________________________________________________________

## 📊 Comparación SHAP vs LIME

```python
# Extraer top features de ambos métodos
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'shap_value': np.abs(shap_value_client)
}).sort_values('shap_value', ascending=False)

lime_importance = pd.DataFrame(
    lime_exp.as_list(),
    columns=['feature', 'lime_weight']
)
lime_importance['lime_weight'] = lime_importance['lime_weight'].abs()

print("\n=== Top Features Comparison ===")
print("\nSHAP (absoluto):")
print(shap_importance.head())

print("\nLIME (absoluto):")
print(lime_importance.head())

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# SHAP
shap_importance.head(5).plot(
    x='feature',
    y='shap_value',
    kind='barh',
    ax=axes[0],
    legend=False,
    color='steelblue'
)
axes[0].set_title('SHAP - Top 5 Features')
axes[0].set_xlabel('|SHAP value|')

# LIME (extraer solo nombre de feature)
lime_features = [f.split('<=')[0].split('>')[0].split('=')[0].strip()
                 for f in lime_importance['feature']]
lime_importance['feature_name'] = lime_features

lime_top = lime_importance.groupby('feature_name')['lime_weight'].sum().nlargest(5)
lime_top.plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('LIME - Top 5 Features')
axes[1].set_xlabel('|LIME weight|')

plt.tight_layout()
plt.savefig('shap_vs_lime_comparison.png', dpi=150)
plt.show()
```

______________________________________________________________________

## 💡 Casos de Usage

### 1. Auditoría de decisiones

```python
def audit_prediction(model, explainer, client_data, threshold=0.05):
    """
    Identifica features con alto impacto para revisión humana
    """
    # SHAP values
    shap_vals = explainer.shap_values(client_data)[1]

    # Features con alto impacto
    high_impact = []
    for i, (feat, val) in enumerate(zip(X.columns, shap_vals)):
        if abs(val) > threshold:
            high_impact.append({
                'feature': feat,
                'shap_value': val,
                'feature_value': client_data[i]
            })

    return high_impact

# Auditar cliente
client_to_audit = X_sample.iloc[0:1]
audit_results = audit_prediction(model, explainer, client_to_audit, threshold=0.05)

print("\n=== Auditoría de Decisión ===")
for item in audit_results:
    print(f"{item['feature']}: {item['feature_value']:.2f} "
          f"(impacto: {item['shap_value']:+.4f})")
```

### 2. Recomendaciones accionables

```python
def generate_recommendations(client, shap_values, top_n=3):
    """
    Genera acciones para reducir riesgo de churn
    """
    # Features que aumentan churn más (SHAP positivos)
    feature_shap = list(zip(X.columns, shap_values))
    sorted_features = sorted(feature_shap, key=lambda x: x[1], reverse=True)

    recommendations = []

    for feature, shap_val in sorted_features[:top_n]:
        if shap_val > 0:
            if feature == 'tenure_months':
                recommendations.append(
                    f"Cliente nuevo ({client[feature]:.0f} meses). "
                    f"Ofrecer descuento de fidelización."
                )
            elif feature == 'monthly_charges':
                recommendations.append(
                    f"Cargos altos (${client[feature]:.2f}). "
                    f"Revisar plan, ofrecer opciones más económicas."
                )
            elif feature == 'contract_type':
                if client[feature] == 0:
                    recommendations.append(
                        "Contrato month-to-month. Incentivar upgrade a contrato anual."
                    )
            elif feature == 'tech_support':
                if client[feature] == 0:
                    recommendations.append(
                        "Sin tech support. Ofrecer prueba gratuita de soporte técnico."
                    )

    return recommendations

# Generar para cliente de alto riesgo
recs = generate_recommendations(
    X_sample.iloc[0],
    shap_values_churn[0],
    top_n=3
)

print("\n=== Recomendaciones para Retención ===")
for i, rec in enumerate(recs, 1):
    print(f"{i}. {rec}")
```

**Salida:**

```
=== Recomendaciones para Retención ===
1. Cliente nuevo (6 meses). Ofrecer descuento de fidelización.
2. Cargos altos ($95.34). Revisar plan, ofrecer opciones más económicas.
3. Contrato month-to-month. Incentivar upgrade a contrato anual.
```

______________________________________________________________________

## 📝 Resumen

### ✅ SHAP vs LIME

| Aspecto             | SHAP                                       | LIME                       |
| ------------------- | ------------------------------------------ | -------------------------- |
| **Base teórica**    | Shapley values (Theory de juegos)          | Local linear approximation |
| **Consistencia**    | Garantizada (propiedades Shapley)          | No garantizada             |
| **Velocidad**       | Lento (especialmente KernelExplainer)      | Rápido                     |
| **Global vs Local** | Ambos                                      | Solo local                 |
| **Model-agnostic**  | Sí (KernelExplainer), optimized para trees | Sí                         |
| **Interpretación**  | Contribución aditiva                       | Peso en Model local        |

### 🎯 Cuándo usar cada uno

**SHAP:**

- ✅ Necesitas explicaciones globales + locales
- ✅ Model basado en árboles (TreeExplainer es rápido)
- ✅ Consistencia matemática crítica (regulación, legal)
- ✅ Analysis de interacciones entre features

**LIME:**

- ✅ Explicaciones rápidas en producción
- ✅ Models arbitrarios (incluso APIs externas)
- ✅ Interpretabilidad sencilla para stakeholders
- ✅ Prototipado rápido

### 💡 Mejores Practices

- ✅ Usar ambos métodos para Validation cruzada
- ✅ Explicar decisiones críticas (rechazos, diagnósticos)
- ✅ Integrar explicaciones en UI para usuarios finales
- ✅ Documentar assumptions (ej: independence en LIME)
- ✅ Validar con expertos del dominio
- ✅ Monitorear cambios en explicaciones (drift)

### 🚫 Errors comunes

- ❌ Confiar ciegamente en una sola explicación
- ❌ Ignorar correlaciones entre features
- ❌ Explicaciones solo para el equipo técnico (no para usuarios)
- ❌ No validar explicaciones con ground truth
- ❌ Olvidar overhead computacional en producción

### 📌 Checklist Explainability

- ✅ Método de explicación seleccionado (SHAP/LIME/ambos)
- ✅ Explicaciones globales generadas
- ✅ Explicaciones locales para casos críticos
- ✅ Visualizaciones comprensibles para stakeholders
- ✅ Recomendaciones accionables derivadas
- ✅ Validation con expertos del dominio
- ✅ Documentación de limitaciones
- ✅ Integración en workflow de decisión

### 🚀 Extensiones

- **Conterfactual Explanations:** "Si tenure fuera 24 meses (en vez de 6), Prediction sería No Churn"
- **Anchors:** Reglas simples que "anclan" Prediction (ej: "Si tenure > 24 → No Churn con 95% confianza")
- **Integrated Gradients:** Para neural networks
- **SHAP Interaction Values:** Capturar interacciones de segundo orden
