"""
CTGAN: Synthetic Tabular Data Generation
=========================================
Generate synthetic tabular data with CTGAN preserving distributions and correlations.

Requirements:
    pip install sdv pandas matplotlib seaborn scikit-learn
"""

from sdv.tabular import CTGAN
from sdv.evaluation import evaluate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATOS DE EJEMPLO
# ============================================================================

def create_sample_dataset() -> pd.DataFrame:
    """
    Create sample dataset (simulation of customer data).
    """
    np.random.seed(42)
    n = 1000

    # Generate correlated features
    age = np.random.normal(45, 15, n).astype(int)
    age = np.clip(age, 18, 90)

    # Income correlacionado con age
    income = 30000 + age * 800 + np.random.normal(0, 10000, n)
    income = np.clip(income, 15000, 200000)

    # Credit score correlacionado con income
    credit_score = 300 + (income / 500) + np.random.normal(0, 50, n)
    credit_score = np.clip(credit_score, 300, 850).astype(int)

    # Categorical features
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n,
                                  p=[0.3, 0.4, 0.2, 0.1])

    employment = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n,
                                   p=[0.7, 0.2, 0.1])

    # Target (loan approval) - correlacionado con features
    loan_approved = ((credit_score > 650) & (income > 50000)).astype(int)
    # Add some noise
    noise_mask = np.random.random(n) < 0.1
    loan_approved[noise_mask] = 1 - loan_approved[noise_mask]

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'education': education,
        'employment': employment,
        'loan_approved': loan_approved
    })

    return df


# ============================================================================
# CTGAN TRAINING
# ============================================================================

def train_ctgan(real_data: pd.DataFrame) -> CTGAN:
    """
    Entrena CTGAN en datos reales.

    CTGAN uses GANs to generate synthetic data:
    - Generator: Creates synthetic data
    - Discriminator: Distinguishes real vs synthetic
    - Conditional: Preserva distribuciones condicionales
    """
    print("🧠 Training CTGAN...")
    print(f"   Real data shape: {real_data.shape}")

    model = CTGAN(
        epochs=100,  # More epochs = better quality (but slower)
        batch_size=500,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256)
    )

    model.fit(real_data)
    print("   ✅ Training complete\n")

    return model


def generate_synthetic_data(model: CTGAN, n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic data.
    """
    print(f"🎲 Generating {n_samples} synthetic samples...")
    synthetic_data = model.sample(n_samples)
    print(f"   ✅ Synthetic data shape: {synthetic_data.shape}\n")

    return synthetic_data


# ============================================================================
# VALIDATION
# ============================================================================

def validate_distributions(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    Compare distributions of numerical features.
    """
    print("="*70)
    print("VALIDATION 1: Distribution Comparison")
    print("="*70 + "\n")

    numerical_cols = ['age', 'income', 'credit_score']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, col in enumerate(numerical_cols):
        axes[i].hist(real_data[col], bins=30, alpha=0.5, label='Real', density=True)
        axes[i].hist(synthetic_data[col], bins=30, alpha=0.5, label='Synthetic', density=True)
        axes[i].set_title(f'{col.title()} Distribution')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: distribution_comparison.png\n")


def validate_correlations(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    Compare correlation matrices.
    """
    print("="*70)
    print("VALIDATION 2: Correlation Matrices")
    print("="*70 + "\n")

    numerical_cols = ['age', 'income', 'credit_score', 'loan_approved']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Real data correlations
    real_corr = real_data[numerical_cols].corr()
    sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Real Data Correlations')

    # Synthetic data correlations
    synthetic_corr = synthetic_data[numerical_cols].corr()
    sns.heatmap(synthetic_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('Synthetic Data Correlations')

    plt.tight_layout()
    plt.savefig('correlation_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: correlation_comparison.png")

    # Print correlation differences
    corr_diff = np.abs(real_corr - synthetic_corr).mean().mean()
    print(f"   Mean correlation difference: {corr_diff:.3f}")
    print(f"   {'✅ Good' if corr_diff < 0.1 else '⚠️  Needs improvement'}\n")


def validate_ml_utility(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    ML utility test:
    Train on synthetic, test on real.
    Accuracy drop should be < 5%.
    """
    print("="*70)
    print("VALIDATION 3: ML Utility Test")
    print("="*70 + "\n")

    # Prepare data
    feature_cols = ['age', 'income', 'credit_score']
    target_col = 'loan_approved'

    # Encode categorical features (simplified)
    for df in [real_data, synthetic_data]:
        df['education_encoded'] = df['education'].astype('category').cat.codes
        df['employment_encoded'] = df['employment'].astype('category').cat.codes

    feature_cols_encoded = feature_cols + ['education_encoded', 'employment_encoded']

    # Split real data
    X_real = real_data[feature_cols_encoded]
    y_real = real_data[target_col]
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42
    )

    # Synthetic data
    X_synthetic = synthetic_data[feature_cols_encoded]
    y_synthetic = synthetic_data[target_col]

    # Model 1: Train on real, test on real (baseline)
    clf_real = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_real.fit(X_train_real, y_train_real)
    acc_real = clf_real.score(X_test_real, y_test_real)

    # Model 2: Train on synthetic, test on real
    clf_synthetic = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_synthetic.fit(X_synthetic, y_synthetic)
    acc_synthetic = clf_synthetic.score(X_test_real, y_test_real)

    # Compare
    acc_drop = acc_real - acc_synthetic
    acc_drop_pct = (acc_drop / acc_real) * 100

    print(f"📈 Trained on Real:      {acc_real:.3f} accuracy")
    print(f"📈 Trained on Synthetic: {acc_synthetic:.3f} accuracy")
    print(f"📉 Accuracy drop:        {acc_drop:.3f} ({acc_drop_pct:.1f}%)")

    if acc_drop_pct < 5:
        print("   ✅ Synthetic data has good ML utility (< 5% drop)")
    elif acc_drop_pct < 10:
        print("   ⚠️  Acceptable ML utility (5-10% drop)")
    else:
        print("   ❌ Poor ML utility (> 10% drop)")

    print()


def validate_categorical_distributions(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    Compare distributions of categorical features.
    """
    print("="*70)
    print("VALIDATION 4: Categorical Feature Distributions")
    print("="*70 + "\n")

    categorical_cols = ['education', 'employment']

    for col in categorical_cols:
        print(f"\n{col.title()}:")
        print("-" * 40)

        real_dist = real_data[col].value_counts(normalize=True).sort_index()
        synthetic_dist = synthetic_data[col].value_counts(normalize=True).sort_index()

        comparison = pd.DataFrame({
            'Real': real_dist,
            'Synthetic': synthetic_dist,
            'Difference': (real_dist - synthetic_dist).abs()
        })

        print(comparison)

        max_diff = comparison['Difference'].max()
        print(f"\nMax difference: {max_diff:.3f}")
        print(f"{'✅ Good' if max_diff < 0.05 else '⚠️  Needs improvement'}")


# ============================================================================
# SDV QUALITY REPORT
# ============================================================================

def generate_quality_report(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    Genera reporte automatic de calidad con SDV.
    """
    print("\n" + "="*70)
    print("QUALITY REPORT (SDV Evaluation)")
    print("="*70 + "\n")

    # SDV evaluation
    quality_report = evaluate(synthetic_data, real_data, aggregate=True)

    print(f"📊 Overall Quality Score: {quality_report:.3f}")
    print(f"   (1.0 = perfect, 0.0 = poor)\n")

    if quality_report > 0.9:
        print("   ✅ Excellent synthetic data quality")
    elif quality_report > 0.7:
        print("   ✅ Good synthetic data quality")
    elif quality_report > 0.5:
        print("   ⚠️  Acceptable quality (consider more training)")
    else:
        print("   ❌ Poor quality (retrain with more epochs)")


# ============================================================================
# DEMOS
# ============================================================================

def demo_full_pipeline():
    """Demo completo del pipeline."""
    print("\n" + "="*70)
    print("🎯 CTGAN: SYNTHETIC TABULAR DATA GENERATION")
    print("="*70 + "\n")

    # 1. Create/load real data
    print("📥 Step 1: Load Real Data")
    print("-" * 70)
    real_data = create_sample_dataset()
    print(real_data.head())
    print(f"\nDataset shape: {real_data.shape}")
    print(f"Columns: {list(real_data.columns)}\n")

    # 2. Train CTGAN
    print("📥 Step 2: Train CTGAN")
    print("-" * 70)
    model = train_ctgan(real_data)

    # 3. Generate synthetic data
    print("📥 Step 3: Generate Synthetic Data")
    print("-" * 70)
    synthetic_data = generate_synthetic_data(model, n_samples=1000)
    print(synthetic_data.head())
    print()

    # 4. Validate
    validate_distributions(real_data, synthetic_data)
    validate_correlations(real_data, synthetic_data)
    validate_categorical_distributions(real_data, synthetic_data)
    validate_ml_utility(real_data, synthetic_data)
    generate_quality_report(real_data, synthetic_data)


def demo_use_cases():
    """Use cases."""
    print("\n" + "="*70)
    print("USE CASES")
    print("="*70 + "\n")

    print("""
1. 🔒 PRIVACY-PRESERVING DATA SHARING
   ────────────────────────────────────────
   • Generate synthetic customer data for external partners
   • No risk of exposing real customer information
   • Maintain statistical properties for analysis

2. 🧪 TESTING & DEVELOPMENT
   ────────────────────────────────────────
   • Generate test data for development environments
   • No need to anonymize production data
   • Unlimited test data generation

3. 📊 DATA AUGMENTATION
   ────────────────────────────────────────
   • Increase training data for ML models
   • Balance imbalanced datasets
   • Improve model generalization

4. 🎓 EDUCATION & RESEARCH
   ────────────────────────────────────────
   • Share datasets that look real without privacy concerns
   • Reproducible research with synthetic data
   • Teaching ML without sensitive data

5. 💼 REGULATORY COMPLIANCE
   ────────────────────────────────────────
   • GDPR/CCPA compliance (synthetic data ≠ personal data)
   • Share data with auditors/regulators
   • Train models without accessing production data
    """)


if __name__ == "__main__":
    demo_full_pipeline()
    demo_use_cases()

    print("\n" + "="*70)
    print("💡 BEST PRACTICES")
    print("="*70)
    print("  ✅ Train with enough epochs (100-300)")
    print("  ✅ Validate with multiple metrics")
    print("  ✅ Test ML utility (train on synthetic, test on real)")
    print("  ✅ Check correlation preservation")
    print("  ✅ Verify categorical distributions")
    print("  ✅ Use for non-critical applications first")
    print("  ✅ Never assume 100% privacy (membership attacks possible)")

    print("\n📚 Resources:")
    print("  • SDV (Synthetic Data Vault): https://sdv.dev/")
    print("  • CTGAN paper: https://arxiv.org/abs/1907.00503")
    print("  • Privacy risks: https://arxiv.org/abs/1909.13495")
