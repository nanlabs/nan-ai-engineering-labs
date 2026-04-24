# Synthetic Data — Generación de Datos Sintéticos

## 🎯 Objetivo

Generar datos sintéticos de alta calidad para ML: desde tabular (GANs) hasta texto (augmentation/LLMs) manteniendo privacidad y balance.

## 💡 Qué aprenderás

- GANs para generar datos tabulares (CTGAN, TabGAN)
- Text augmentation (back-translation, paraphrasing, EDA)
- Privacidad con Differential Privacy
- Validar calidad de datos sintéticos (statistical tests, ML utility)
- Detectar bias en datos generados
- Synthetic data para edge cases y testing

## 📂 Contenido

### Examples

- **ex_01_ctgan_tabular.py**: CTGAN para generar clientes sintéticos
- **ex_02_text_augmentation.py**: Aumentar dataset de texto con NLP
- **ex_03_validate_quality.py**: Comparar distribuciones real vs sintético
- **ex_04_differential_privacy.py**: Añadir ruido con garantías DP

## 🔑 Conceptos Clave

### Why Synthetic Data?

**1. Privacy**: Compartir datos sin exponer información real
**2. Data Scarcity**: Aumentar dataset pequeño para training
**3. Balancing**: Oversample clases minoritarias
**4. Testing**: Generar edge cases para QA
**5. Simulation**: Crear escenarios "what-if"

### Synthetic Data Pipeline

```
┌──────────────────────────────────────────┐
│  1. Real Data (private)                  │
│     ├─ Customer demographics             │
│     └─ Purchase history                  │
├──────────────────────────────────────────┤
│  2. Train Generative Model               │
│     ├─ GAN (CTGAN, TabGAN)               │
│     ├─ VAE                                │
│     └─ Diffusion Models                  │
├──────────────────────────────────────────┤
│  3. Generate Synthetic Data              │
│     ├─ Sample from latent space          │
│     └─ Add differential privacy noise    │
├──────────────────────────────────────────┤
│  4. Validate Quality                     │
│     ├─ Statistical similarity (KS test)  │
│     ├─ ML utility (train model)          │
│     └─ Privacy (membership inference)    │
└──────────────────────────────────────────┘
```

## 🧬 CTGAN for Tabular Data

**CTGAN (Conditional Tabular GAN)**: GAN especializado en datos tabulares con categorical + continuous features.

```python
from sdv.tabular import CTGAN

# Real customer data
real_data = pd.DataFrame({
    "age": [25, 67, 42, 31],
    "income": [45000, 120000, 67000, 52000],
    "gender": ["M", "F", "M", "F"],
    "purchased": [0, 1, 1, 0]
})

# Train CTGAN
model = CTGAN(epochs=500)
model.fit(real_data)

# Generate synthetic customers
synthetic_data = model.sample(num_rows=1000)

# synthetic_data tiene mismas distribuciones pero no son clientes reales
```

**Ventajas**:

- Maneja categorical + continuous
- Preserva correlaciones entre features
- No requiere feature engineering

**Limitaciones**:

- Entrenamiento lento (horas para datasets grandes)
- Puede generar outliers imposibles (edad = 150 años)
- No garantiza privacidad (puede memorizar datos)

## 📝 Text Augmentation Techniques

### 1. Back-Translation

```python
# Original
text = "The movie was amazing and emotional"

# Translate to French
french = translator.translate(text, target="fr")
# "Le film était incroyable et émouvant"

# Translate back to English
augmented = translator.translate(french, target="en")
# "The film was incredible and moving"
```

### 2. Synonym Replacement (EDA)

```python
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
augmented = aug.augment("The product is good")
# "The merchandise is good"
```

### 3. Paraphrasing with LLMs

```python
prompt = """
Parafrasea el siguiente texto manteniendo el significado:

Original: "El servicio al cliente fue excelente y rápido."

Paráfrasis:
"""

# LLM genera: "La atención al cliente fue muy buena y eficiente."
```

## 🔒 Differential Privacy

**Problema**: Synthetic data puede leak información real.

**Solución**: Añadir ruido calibrado con garantías matemáticas.

```python
import numpy as np

def add_laplace_noise(value, sensitivity, epsilon):
    """
    epsilon: privacy budget (más bajo = más privacidad)
    sensitivity: máximo cambio posible en el agregado
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

# Ejemplo: promedio de salarios
real_avg_salary = 75000
epsilon = 1.0  # Privacy budget
private_avg_salary = add_laplace_noise(real_avg_salary, 10000, epsilon)

# private_avg_salary ≈ 75000 pero con ruido
```

## 📊 Validating Quality

### 1. Statistical Similarity

```python
from scipy.stats import ks_2samp

# Kolmogorov-Smirnov test
statistic, pvalue = ks_2samp(real_data["age"], synthetic_data["age"])

if pvalue > 0.05:
    print("✅ Distributions are similar")
else:
    print("❌ Distributions differ significantly")
```

### 2. ML Utility

```python
# Train on synthetic, test on real
model_syn = RandomForestClassifier()
model_syn.fit(X_synthetic, y_synthetic)
score_syn = model_syn.score(X_test_real, y_test_real)

# Train on real, test on real
model_real = RandomForestClassifier()
model_real.fit(X_train_real, y_train_real)
score_real = model_real.score(X_test_real, y_test_real)

# Utility ratio (should be > 0.9)
utility = score_syn / score_real
```

### 3. Privacy Audit (Membership Inference)

```python
# ¿Puede un atacante determinar si un registro estaba en training data?
# Si el atacante acierta > 50%, hay privacy leak
```

## 🛠️ Tools & Libraries

| Tool           | Type   | Use Case                        | Ease       | Privacy           |
| -------------- | ------ | ------------------------------- | ---------- | ----------------- |
| **SDV**        | Python | Tabular GANs (CTGAN)            | ⭐⭐⭐     | ⚠️ No DP          |
| **Gretel**     | API    | Tabular + text + time series    | ⭐⭐⭐⭐   | ✅ DP opcional    |
| **NLPaug**     | Python | Text augmentation               | ⭐⭐⭐⭐⭐ | N/A               |
| **SmartNoise** | Python | Differential Privacy            | ⭐⭐       | ✅ DP garantizado |
| **Faker**      | Python | Fake data simple (not ML-based) | ⭐⭐⭐⭐⭐ | ✅ No real data   |

## 🧪 Ejercicio Rápido

1. **Setup**: `pip install sdv pandas scikit-learn`
1. **Load dataset**: Usa Titanic o Iris
1. **Train CTGAN**: `model.fit(real_data)`
1. **Generate**: 1000 registros sintéticos
1. **Validate**: KS test + ML utility test
1. **Privacy**: Añade Differential Privacy noise
1. **Compare**: Visualiza distribuciones real vs sintético

## 📚 Recursos Curados

**Libraries:**

- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) - CTGAN, TabGAN
- [NLPaug](https://github.com/makcedward/nlpaug) - Text augmentation
- [Gretel.ai](https://gretel.ai/) - Commercial tool con DP
- [SmartNoise](https://github.com/opendp/smartnoise-sdk) - Microsoft DP library

**Papers:**

- [CTGAN (2019)](https://arxiv.org/abs/1907.00503) - Modelando datos tabulares con GANs
- [Differential Privacy (2006)](https://www.microsoft.com/en-us/research/publication/calibrating-noise-to-sensitivity-in-private-data-analysis/) - Dwork et al.
- [TabGAN](https://arxiv.org/abs/2102.08468) - Otra arquitectura para tabular

**Tutorials:**

- [SDV Quick Start](https://sdv.dev/SDV/getting_started/index.html)
- [Differential Privacy Explained](https://desfontain.es/privacy/)

## ✅ Checklist de Aprendizaje

- [ ] Entrenar CTGAN en dataset tabular
- [ ] Generar 1000+ registros sintéticos
- [ ] Validar con KS test (p > 0.05)
- [ ] ML utility test (score_syn/score_real > 0.9)
- [ ] Text augmentation con back-translation
- [ ] Añadir Differential Privacy noise
- [ ] Detectar bias en datos sintéticos (protected attributes)
- [ ] Privacy audit con membership inference

## 🎯 Impacto Real

- **Healthcare**: Compartir datos médicos sintéticos para research sin violar HIPAA
- **Finance**: Generar transacciones sintéticas para fraud detection training
- **Testing**: Crear edge cases (edades extremas, ingresos negativos) para QA
- **Data Marketplaces**: Vender datasets sintéticos sin privacy concerns
- **Balancing**: Oversample clases minoritarias (fraude, enfermedades raras)

## 🚨 Common Pitfalls

**Over-fitting**: GAN memoriza datos reales

- **Solución**: Early stopping, validation set, privacy audits

**Unrealistic outliers**: Generar valores imposibles

- **Solución**: Post-processing constraints (edad > 0)

**Bias amplification**: Sesgo en real data se amplifica en synthetic

- **Solución**: Fairness constraints durante training

**Mode collapse**: GAN genera poca diversidad

- **Solución**: Usar CTGAN (menos propenso), increase epochs

## 🚀 Próximos Pasos

Combina con:

- **data-privacy-security** para DP avanzado
- **ethics-bias-explainability** para detectar bias en synthetic data
- **machine-learning-fundamentals** para usar synthetic data en training

## Module objective

Pendiente de completar este apartado.

## What you will achieve

Pendiente de completar este apartado.

## Internal structure

Pendiente de completar este apartado.

## Level path (L1-L4)

Pendiente de completar este apartado.

## Recommended plan (by progress, not by weeks)

Pendiente de completar este apartado.

## Module completion criteria

Pendiente de completar este apartado.
