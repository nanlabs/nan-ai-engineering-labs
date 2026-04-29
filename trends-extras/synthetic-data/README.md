# Synthetic Data — Generation of Synthetic Data

## 🎯 Objective

Generate high quality synthetic data for ML: from tabular (GANs) to text (augmentation/LLMs) maintaining privacy and balance.

## 💡 What will you learn

- GANs to generate tabular data (CTGAN, TabGAN)
- Text augmentation (back-translation, paraphrasing, EDA)
- Privacy with Differential Privacy
- Validate quality of synthetic data (statistical tests, ML utility)
- Detect bias in generated data
- Synthetic data for edge cases and testing

## 📂 Content

### Examples

- **ex_01_ctgan_tabular.py**: CTGAN to generate synthetic clients
- **ex_02_text_augmentation.py**: Augment text dataset with NLP
- **ex_03_validate_quality.py**: Compare real vs synthetic distributions
- **ex_04_differential_privacy.py**: Add noise with DP guarantees

## 🔑 Concepts Clave

### Why Synthetic Data?

**1. Privacy**: Share data without exposing real information
**2. Data Scarcity**: Increase small dataset for training
**3. Balancing**: Over-sample minority classes
**4. Testing**: Generate edge cases for QA
**5. Simulation**: Create "what-if" scenarios

### Synthetic Data Pipeline

```
┌──────────────────────────────────────────┐
│  1. Real Data (private)                  │
│     ├─ Customer demographics             │
│     └─ Purchase history                  │
├──────────────────────────────────────────┤
│  2. Train Generative Model               │
│     ├─ GAN (CTGAN, TabGAN)               │
│     ├─ VAE                               │
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

**CTGAN (Conditional Tabular GAN)**: GAN specialized in tabular data with categorical + continuous features.

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

# Synthetic data shares the same distributions but does not contain real customers.
```

**Advantages**:

- Handles categorical + continuous
- Preserve correlations between features
- No requires feature engineering

**Limitations**:

- Slow training (hours for large datasets)
- Can generate impossible outliers (age = 150 years)
- Does not guarantee privacy (can memorize Data)

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
Paraphrase the following text maintaining its meaning:

Original: "Customer service was excellent and fast."

Paraphrase:
"""

# LLM generate: "The customer support was outstanding and efficient."

```

## 🔒 Differential Privacy

**Problem**: Synthetic data can leak information real.

**Solution**: Add noise calibrated with mathematical guarantees.

```python
import numpy as np

def add_laplace_noise(value, sensitivity, epsilon):
    """
    epsilon: privacy budget (lower = more privacy)
    sensitivity: maximum possible change in the aggregate
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

# Example: average salary
real_avg_salary = 75000
epsilon = 1.0  # Privacy budget
private_avg_salary = add_laplace_noise(real_avg_salary, 10000, epsilon)

# private_avg_salary ≈ 75000 but con noise
```

## 📊 Validating Quality

### 1. Statistical Similarity

```python
from scipy.stats import ks_2samp

# Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(real_data["age"], synthetic_data["age"])

if p_value > 0.05:
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
# # Can an attacker determine if a record was in training data?
# If the attacker is correct > 50%, there is a privacy leak
```

## 🛠️ Tools & Libraries

| Tools          | Type   | Use Case                        | Ease       | Privacy          |
| -------------- | ------ | --------------------------------| -----------| -----------------|
| **SDV**        | Python | Tabular GANs (CTGAN)            | ⭐⭐⭐     | ⚠️No DP          |
| **Gretel**     | API    | Tabular + text + time series    | ⭐⭐⭐⭐   | ✅ DP optional   |
| **NLPaug**     | Python | Text augmentation               | ⭐⭐⭐⭐⭐ | N/A              |
| **SmartNoise** | Python | Differential Privacy            | ⭐⭐       | ✅ DP guaranteed |
| **Faker**      | Python | Simple fake data (not ML-based) | ⭐⭐⭐⭐⭐ | ✅ Not real data |

## 🧪 Quick Exercise

1. **Setup**: `pip install sdv pandas scikit-learn`
1. **Load dataset**: Usa Titanic o Iris
1. **Train CTGAN**: `model.fit(real_data)`
1. **Generate**: 1000 synthetic records
1. **Validate**: KS test + ML utility test
1. **Privacy**: Add Differential Privacy noise
1. **Compare**: Visualize real vs synthetic distributions

## 📚 Curated Resources

**Libraries:**

- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) - CTGAN, TabGAN
- [NLPaug](https://github.com/makcedward/nlpaug) - Text augmentation
- [Gretel.ai](https://gretel.ai/) - Commercial tool with DP
- [SmartNoise](https://github.com/opendp/smartnoise-sdk) - Microsoft DP library

**Papers:**

- [CTGAN (2019)](https://arxiv.org/abs/1907.00503) - Modeling tabular data with GANs
- [Differential Privacy (2006)](https://www.microsoft.com/en-us/research/publication/calibrating-noise-to-sensitivity-in-private-data-analysis/) - Dwork et al.
- [TabGAN](https://arxiv.org/abs/2102.08468) - Another architecture to tabulate

**Tutorials:**

- [SDV Quick Start](https://sdv.dev/SDV/getting_started/index.html)
- [Differential Privacy Explained](https://desfontain.es/privacy/)

## ✅ Learning Checklist

- [ ] Train CTGAN in tabular dataset
- [ ] Generate 1000+ synthetic records
- [ ] Validate with KS test (p > 0.05)
- [ ] ML utility test (score_syn/score_real > 0.9)
- [ ] Text augmentation with back-translation
- [ ] Add Differential Privacy noise
- [ ] Detect bias in synthetic data (protected attributes)
- [ ] Privacy audit with membership inference

## 🎯 Real Impact

- **Healthcare**: Share synthetic medical data for research without violating HIPAA
- **Finance**: Generate synthetic transactions for fraud detection training
- **Testing**: Create edge cases (extreme ages, negative income) for QA
- **Data Marketplaces**: Sell synthetic datasets without privacy concerns
- **Balancing**: Over-sample minority classes (fraud, rare diseases)

## 🚨 Common Pitfalls

**Over-fitting**: GAN memorizes real data

- **Solution**: Early stopping, validation set, privacy audits

**Unrealistic outliers**: Generate impossible values

- **Solution**: Post-processing constraints (age > 0)

**Bias amplification**: Bias in real data is amplified in synthetic

- **Solution**: Fairness constraints durante training

**Mode collapse**: GAN generates little diversity

- **Solution**: Use CTGAN (less prone), increase epochs

## 🚀 Next Steps

Combines with:

- **data-privacy-security** for advanced DP
- **ethics-bias-explainability** to detect bias in synthetic data
- **machine-learning-fundamentals** to use synthetic data in training

## Module objective

Learn to generate, validate, and use synthetic datasets responsibly to improve model development while preserving privacy and data utility.

## What you will achieve

- Generate tabular and text synthetic data with modern approaches.
- Measure utility and distribution similarity against source data.
- Evaluate privacy risk and leakage exposure.
- Integrate synthetic data into ML experimentation workflows.

## Internal structure

- `README.md`: generation methods, risks, and quality criteria.
- `examples/`: CTGAN, text augmentation, and privacy-preserving synthesis.
- `practices/`: utility/privacy experiments and dataset comparisons.

## Level path (L1-L4)

- L1: Generate baseline synthetic samples.
- L2: Tune generation parameters and compare quality.
- L3: Evaluate privacy and bias side effects.
- L4: Build a reproducible synthetic-data experiment pipeline.

## Recommended plan (by progress, not by weeks)

Begin with one baseline generator and utility metrics, then add privacy checks and bias analysis. Scale to multiple generators only after baseline reproducibility is achieved.

## Module completion criteria

- You can generate synthetic data with reproducible settings.
- You can present utility and privacy metrics in a concise report.
- You can justify when synthetic data improves or harms model quality.
