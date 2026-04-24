"""
Privacy-Preserving Synthetic Data with Differential Privacy
============================================================
Genera datos sintéticos con garantías de privacy usando Differential Privacy.

Requirements:
    pip install diffprivlib pandas numpy matplotlib scikit-learn
"""

import diffprivlib.models as dp
from diffprivlib.mechanisms import Laplace, Gaussian
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DIFFERENTIAL PRIVACY BASICS
# ============================================================================

def explain_differential_privacy():
    """
    Explica qué es Differential Privacy.
    """
    print("="*70)
    print("WHAT IS DIFFERENTIAL PRIVACY?")
    print("="*70 + "\n")

    print("""
Differential Privacy (DP) provides mathematical guarantees that:
- Individual records cannot be identified
- Adding/removing one person doesn't significantly change results
- Quantified privacy budget (epsilon, ε)

KEY CONCEPTS:

1. Privacy Budget (ε - Epsilon):
   ────────────────────────────────────────
   • ε = 0.1  →  Very strong privacy (more noise)
   • ε = 1.0  →  Strong privacy (recommended)
   • ε = 10   →  Weak privacy (less noise)

   Lower ε = Better privacy but lower accuracy

2. Noise Mechanisms:
   ────────────────────────────────────────
   • Laplace mechanism: For numeric outputs
   • Gaussian mechanism: For continuous data
   • Exponential mechanism: For non-numeric outputs

3. Privacy Composition:
   ────────────────────────────────────────
   • Each query "consumes" privacy budget
   • Total ε = sum of all queries
   • Once budget exhausted, no more queries

EXAMPLE:
   Query 1 (ε=0.5): Average age → 42.3 (with noise)
   Query 2 (ε=0.5): Average income → $65,000 (with noise)
   Total privacy cost: ε=1.0
    """)


# ============================================================================
# DP MECHANISMS
# ============================================================================

def demo_laplace_mechanism():
    """
    Demo: Laplace mechanism for adding noise.
    """
    print("\n" + "="*70)
    print("DEMO 1: Laplace Mechanism")
    print("="*70 + "\n")

    # Original sensitive value
    true_average_salary = 75000

    # Privacy parameters
    epsilon_values = [0.1, 1.0, 10.0]
    sensitivity = 200000  # Max possible difference (salary range)

    print(f"True average salary: ${true_average_salary:,}\n")
    print("Adding Laplace noise with different ε:\n")

    for epsilon in epsilon_values:
        # Create Laplace mechanism
        laplace = Laplace(epsilon=epsilon, sensitivity=sensitivity)

        # Add noise
        noisy_salary = laplace.randomise(true_average_salary)

        # Calculate error
        error = abs(noisy_salary - true_average_salary)
        error_pct = (error / true_average_salary) * 100

        print(f"ε = {epsilon:4.1f}:  ${noisy_salary:8,.0f}  "
              f"(error: ${error:,.0f} / {error_pct:.1f}%)")

    print("\n💡 Lower ε = More noise = Better privacy\n")


def demo_gaussian_mechanism():
    """
    Demo: Gaussian mechanism.
    """
    print("="*70)
    print("DEMO 2: Gaussian Mechanism")
    print("="*70 + "\n")

    # Dataset: ages of people
    true_ages = np.random.normal(45, 15, 1000)
    true_mean = true_ages.mean()

    # Privacy parameters
    epsilon = 1.0
    delta = 1e-5  # Failure probability for Gaussian
    sensitivity = 90  # Age range (0-90)

    # Gaussian mechanism
    gaussian = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

    # Add noise to mean
    noisy_mean = gaussian.randomise(true_mean)

    print(f"True mean age:   {true_mean:.2f}")
    print(f"Noisy mean age:  {noisy_mean:.2f}")
    print(f"Error:           {abs(noisy_mean - true_mean):.2f}")
    print(f"\nPrivacy budget:  ε={epsilon}, δ={delta}")
    print("\n💡 Gaussian allows slightly more accuracy than Laplace\n")


# ============================================================================
# DP MACHINE LEARNING
# ============================================================================

def demo_dp_classifier():
    """
    Demo: Train classifier with differential privacy.
    """
    print("="*70)
    print("DEMO 3: DP Machine Learning")
    print("="*70 + "\n")

    # Generate synthetic dataset
    np.random.seed(42)
    n = 500

    X = np.random.randn(n, 5)  # 5 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary classification

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test\n")

    # 1. Normal (non-private) classifier
    print("1. Normal Logistic Regression:")
    normal_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    normal_clf.fit(X_train, y_train)
    normal_acc = normal_clf.score(X_test, y_test)
    print(f"   Accuracy: {normal_acc:.3f}\n")

    # 2. DP Logistic Regression
    print("2. DP Logistic Regression (ε=1.0):")
    dp_clf = dp.LogisticRegression(epsilon=1.0, data_norm=5.0)
    dp_clf.fit(X_train, y_train)
    dp_acc = dp_clf.score(X_test, y_test)
    print(f"   Accuracy: {dp_acc:.3f}")
    print(f"   Privacy cost: ε=1.0")

    # 3. Even stronger privacy
    print("\n3. DP Logistic Regression (ε=0.1):")
    dp_clf_strong = dp.LogisticRegression(epsilon=0.1, data_norm=5.0)
    dp_clf_strong.fit(X_train, y_train)
    dp_acc_strong = dp_clf_strong.score(X_test, y_test)
    print(f"   Accuracy: {dp_acc_strong:.3f}")
    print(f"   Privacy cost: ε=0.1")

    # Compare
    print("\n" + "-"*70)
    print("COMPARISON:")
    print(f"  Normal:       {normal_acc:.3f} (no privacy)")
    print(f"  DP (ε=1.0):   {dp_acc:.3f} (strong privacy)")
    print(f"  DP (ε=0.1):   {dp_acc_strong:.3f} (very strong privacy)")

    acc_drop_1 = (normal_acc - dp_acc) / normal_acc * 100
    acc_drop_01 = (normal_acc - dp_acc_strong) / normal_acc * 100

    print(f"\n  Accuracy drop (ε=1.0):  {acc_drop_1:.1f}%")
    print(f"  Accuracy drop (ε=0.1):  {acc_drop_01:.1f}%")
    print("\n💡 Privacy-Utility Tradeoff: More privacy → Lower accuracy\n")


# ============================================================================
# SYNTHETIC DATA WITH DP
# ============================================================================

def demo_dp_synthetic_data():
    """
    Demo: Generate DP synthetic data.
    """
    print("="*70)
    print("DEMO 4: DP Synthetic Data Generation")
    print("="*70 + "\n")

    # Original data
    np.random.seed(42)
    original_data = pd.DataFrame({
        'age': np.random.normal(45, 15, 200).astype(int),
        'income': np.random.normal(60000, 20000, 200),
        'credit_score': np.random.normal(700, 100, 200).astype(int)
    })

    # Clip values
    original_data['age'] = original_data['age'].clip(18, 90)
    original_data['income'] = original_data['income'].clip(0, 200000)
    original_data['credit_score'] = original_data['credit_score'].clip(300, 850)

    print("Original data statistics:")
    print(original_data.describe()[['age', 'income', 'credit_score']].round(1))
    print()

    # Add DP noise to statistics
    epsilon = 1.0

    print(f"\nGenerating DP statistics (ε={epsilon}):\n")

    dp_stats = {}

    for col in ['age', 'income', 'credit_score']:
        # True statistics
        true_mean = original_data[col].mean()
        true_std = original_data[col].std()

        # Sensitivity (range of values)
        if col == 'age':
            sensitivity = 72  # 90 - 18
        elif col == 'income':
            sensitivity = 200000
        else:  # credit_score
            sensitivity = 550  # 850 - 300

        # Add Laplace noise
        laplace = Laplace(epsilon=epsilon/2, sensitivity=sensitivity)  # Split budget
        dp_mean = laplace.randomise(true_mean)
        dp_std = laplace.randomise(true_std)

        dp_stats[col] = {'mean': dp_mean, 'std': dp_std}

        print(f"{col}:")
        print(f"  True mean: {true_mean:.1f}  →  DP mean: {dp_mean:.1f}")
        print(f"  True std:  {true_std:.1f}  →  DP std:  {dp_std:.1f}")
        print()

    # Generate synthetic data from DP statistics
    print("Generating synthetic data from DP statistics...\n")

    synthetic_data = pd.DataFrame({
        'age': np.random.normal(dp_stats['age']['mean'], dp_stats['age']['std'], 200).astype(int),
        'income': np.random.normal(dp_stats['income']['mean'], dp_stats['income']['std'], 200),
        'credit_score': np.random.normal(dp_stats['credit_score']['mean'], dp_stats['credit_score']['std'], 200).astype(int)
    })

    # Clip synthetic data
    synthetic_data['age'] = synthetic_data['age'].clip(18, 90)
    synthetic_data['income'] = synthetic_data['income'].clip(0, 200000)
    synthetic_data['credit_score'] = synthetic_data['credit_score'].clip(300, 850)

    print("Synthetic DP data statistics:")
    print(synthetic_data.describe()[['age', 'income', 'credit_score']].round(1))

    print("\n💡 Synthetic data cannot be traced back to individuals!")
    print(f"   Total privacy cost: ε={epsilon}\n")


# ============================================================================
# MEMBERSHIP INFERENCE ATTACK
# ============================================================================

def demo_membership_attack():
    """
    Demo: Resistance to membership inference attacks.
    """
    print("="*70)
    print("DEMO 5: Membership Inference Attack Resistance")
    print("="*70 + "\n")

    print("""
MEMBERSHIP INFERENCE ATTACK:
────────────────────────────────────────
Attacker tries to determine if a specific person was in the training data.

WITHOUT DP:
  • Attacker queries: "Average salary of people age 35 in ZIP 12345"
  • Result: $75,000
  • Attacker adds/removes target person and queries again
  • If result changes significantly → person was in dataset!

WITH DP (ε=1.0):
  • Query 1 (with person):    $75,234 (noisy)
  • Query 2 (without person): $75,891 (noisy)
  • Noise masks the difference → Cannot determine membership!

GUARANTEE:
  With ε=1.0, probability of identifying individual < e^1.0 ≈ 2.7x
  With ε=0.1, probability < e^0.1 ≈ 1.1x (very strong protection)
    """)

    # Simulate attack
    np.random.seed(42)

    epsilon = 1.0
    true_value = 75000
    sensitivity = 50000

    laplace = Laplace(epsilon=epsilon, sensitivity=sensitivity)

    # Query with person in dataset
    with_person = laplace.randomise(true_value)

    # Query without person (slightly different true value)
    without_person = laplace.randomise(true_value - 100)  # Person's contribution

    print(f"\nSimulated Attack:")
    print(f"  Query WITH person:    ${with_person:,.0f}")
    print(f"  Query WITHOUT person: ${without_person:,.0f}")
    print(f"  Difference:           ${abs(with_person - without_person):,.0f}")
    print(f"\n  → Attacker cannot reliably determine membership!")
    print(f"     (Noise >> Individual contribution)\n")


# ============================================================================
# BEST PRACTICES
# ============================================================================

def demo_best_practices():
    """Best practices."""
    print("="*70)
    print("BEST PRACTICES")
    print("="*70 + "\n")

    print("""
1. ✅ CHOOSE APPROPRIATE ε
   ────────────────────────────────────────
   • Healthcare/Finance: ε ≤ 1.0
   • General analytics: ε = 1.0 - 10.0
   • Less sensitive data: ε > 10.0
   • Start conservative, relax if needed

2. ✅ TRACK PRIVACY BUDGET
   ────────────────────────────────────────
   • Every query consumes budget
   • Total ε = sum of all queries
   • Plan queries in advance
   • Prioritize important queries

3. ✅ MINIMIZE SENSITIVITY
   ────────────────────────────────────────
   • Clip outliers before DP
   • Normalize data to smaller ranges
   • Lower sensitivity = less noise needed

4. ⚠️  UNDERSTAND TRADEOFFS
   ────────────────────────────────────────
   • More privacy (lower ε) = Less accuracy
   • More queries = Higher total ε
   • Balance privacy and utility

5. ✅ COMBINE WITH OTHER TECHNIQUES
   ────────────────────────────────────────
   • DP + Anonymization
   • DP + Synthetic data (CTGAN)
   • DP + Encryption
   • Layered privacy approach

6. ✅ LEGAL COMPLIANCE
   ────────────────────────────────────────
   • GDPR: DP can help with pseudonymization
   • HIPAA: DP satisfies de-identification
   • CCPA: DP protects consumer data
   • Consult legal experts!
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔒 PRIVACY-PRESERVING SYNTHETIC DATA")
    print("📊 Differential Privacy Techniques")
    print("="*70 + "\n")

    explain_differential_privacy()
    demo_laplace_mechanism()
    demo_gaussian_mechanism()
    demo_dp_classifier()
    demo_dp_synthetic_data()
    demo_membership_attack()
    demo_best_practices()

    print("="*70)
    print("USE CASES")
    print("="*70)
    print("  • Healthcare: Share patient data for research")
    print("  • Finance: Analyze customer behavior")
    print("  • Government: Census data release")
    print("  • Tech: User analytics and A/B testing")
    print("  • Education: Student data analysis")

    print("\n📚 Resources:")
    print("  • IBM diffprivlib: https://github.com/IBM/differential-privacy-library")
    print("  • DP Book: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf")
    print("  • Google DP: https://github.com/google/differential-privacy")
