# Theory — Machine Learning Fundamentals

## Why this module matters

Machine Learning is the basis of modern AI systems. Understanding the fundamentals allows you to choose the right algorithm, evaluate results critically, and diagnose problems in production. This Module builds the intuition necessary to work with supervised and unsupervised Models.

______________________________________________________________________

## 1. What is Machine Learning?

Machine Learning (ML) is the field of AI that allows systems to learn patterns from Data to make Predictions or make decisions **without being explicitly programmed**.

### Key difference with traditional programming

- **Traditional programming:** human defines explicit rules (if/else).
- **Machine Learning:** the system learns rules from Examples.

______________________________________________________________________

## 2. Types of Machine Learning

### Supervised Learning

**Definition:** You have Data with known tags (`and`). The Model learns the relationship between inputs (`X`) and outputs (`y`).

**Subtypes:**

- **Regression:** predict continuous values ​​(house price, temperature).
- Algorithms: Linear Regression, Polynomial Regression, Random Forest Regressor.
- **Classification:** predict discrete classes (spam/non-spam, cat/dog).
- Algorithms: Logistic Regression, Decision Trees, SVM, Random Forest, gradient Boosting.

### Unsupervised Learning

**Definition:** There are no tags (`and`). The Model finds hidden patterns or Structure in the Data.

**Subtypes:**

- **Clustering:** group similar data (customer segmentation).
  - Algorithms: K-Means, DBSCAN, Hierarchical Clustering.
- **Dimensionality reduction:** compress features while maintaining information.
  - Algorithms: PCA, t-SNE, UMAP.

### Reinforcement Learning (Introduction)

The agent learns to make decisions through Test and error, receiving rewards or penalties.

📹 **Videos recommended:**

1. [Machine Learning Crash Course - Google](https://www.youtube.com/watch?v=nKW8Ndu7Mjw) - 40 min
1. [StatQuest: Machine Learning Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) - series complete
1. [ML Course - Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning) - certifiable complete cursor

______________________________________________________________________

## 3. Typical ML Pipeline

### Step 1: Define the Problem

- What do I want to predict? (target/Objective).
- Is it regression or Classification?
- What Success Metric do you use?

### Step 2: Collect and explore data

- EDA (Exploratory Data Analysis).
- Identify Data quality, outliers, distributions.

### Step 3: Prepare Data

- Cleaning: nulls, duplicates, outliers.
- Feature engineering: create new features.
- Encoding: convert categorical to numeric.
- Scaling: normalize or standardize.

### Step 4: Split Data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Important:** **NEVER** use Test Data during Training or tuning.

### Paso 5: Train Model baseline

Start with the simplest Model possible to establish a reference point.

### Step 6: Evaluate yourself with correct Metrics

Choose Metric according to the Problem and cost of Errors.

### Step 7: Adjust and improve

- Try other Algorithms.
- Adjust Hyperparameters.
- Cross validation.
- Additional feature engineering.

### Paso 8: Validation final

Evaluate yourself on test set **just once** at the end.

📹 **Videos recommended:**

1. [ML Workflow Explained - Krish Naik](https://www.youtube.com/watch?v=fiz1ORTBGpY) - 25 min
1. [Train/Test Split - StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw) - 8 min

______________________________________________________________________

## 4. Evaluation Metrics

### For Regression Problems

- **MAE (Mean Absolute error):** average of the absolute error. Easy to interpret.
- **MSE (Mean Squared error):** penalizes large errors more strongly.
- **RMSE (Root Mean Squared error):** root of MSE, in the same units as `y`.
- **R² (R-squared):** % of variance explained by the Model (0-1, higher is better).

### For Classification Problems

- **accuracy:** % of correct Predictions. **Be careful with unbalanced datasets.**
- **Precision:** Of those I predicted as positive, how many are really positive?
- **recall (Sensitivity):** of the really positive ones, how many did I detect?
- **f1-Score:** harmonic mean of precision and recall.
- **auc-ROC:** area under the ROC curve. Measures the discrimination capacity of the Model.

### How to choose Metric

- **Regression:** RMSE if outliers matter, MAE if you want robustness.
- **Classification balanced:** accuracy can suffice.
- **Unbalanced classification:** use f1, precision/recall according to context.
- Example: fraud detection → prioritize recall (not lose frauds).
  - Example: Spam filter → balance between precision and recall.

📹 **Videos recommended:**

1. [Regression Metrics - StatQuest](https://www.youtube.com/watch?v=lgZ-s4XNPcs) - 10 min
1. [Classification Metrics - StatQuest](https://www.youtube.com/watch?v=4jRBRDbJemM) - 15 min
1. [Confusion Matrix Explained - Krish Naik](https://www.youtube.com/watch?v=wpp3VfzgNcI) - 12 min

📚 **Resources written:**

- [Scikit-learn Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Google ML Crash Course - Classification](https://developers.google.com/machine-learning/crash-course/classification)

______________________________________________________________________

## 5. Critical concepts

### overfitting

**Symptom:** The Model learns the training set too well, including noise and irrelevant details. High performance in train, low in test.

**Causes:**

- Model too complex for the amount of Data.
- Train for too many iterations.
- No use Regularization.

**Solutions:**

- Get more Data.
- Simplify the Model (less features, less depth).
- Regularization (L1, L2).
- Early stopping.
- Cross-validation.

### underfitting

**Symptom:** The Model is too simple and does not capture relevant patterns. Low performance in train and test.

**Causes:**

- Model too simple.
- Insufficient or uninformative features.

**Solutions:**

- Use more complex Model.
- Add more features.
- Reduce Regularization.

### Bias-Variance Tradeoff

- **High bias:** underfitting (Very simple Model).
- **High variance:** overfitting (Model very sensitive to Training Data).
- **Objective:** find optimal balance.

### Data Leakage

**Definition:** Use information from the future or the test set during Training.

**Examples common:**

- Normalize before doing train/test split.
- Use features that include the target indirectly.
- Train with Data after the Prediction moment.

**Prevention:**

- Always split Data FIRST.
- Apply transformations only on train, then apply to test.
- Review features that have perfect correlation with the target.

📹 **Videos recommended:**

1. [Overfitting and Underfitting - StatQuest](https://www.youtube.com/watch?v=EuBBz3bI-aA) - 14 min
1. [Bias-Variance Tradeoff - Krish Naik](https://www.youtube.com/watch?v=EuBBz3bI-aA) - 20 min
1. [Data Leakage Explained - Kaggle](https://www.youtube.com/watch?v=jmUCgGDsG7g) - 10 min

______________________________________________________________________

## 6. Fundamental algorithms

### Linear Regression

- Simpler model. Assume linear relationship between `X` and `y`.
- Interpretable and fast.

### Logistic Regression

- Binary classification.
- Output: probability (0-1).

### Decision Trees

- Successive splits based on features.
- Easy to interpret.
- Prone to overfitting.

### Random Forest

- Ensemble of multiple trees.
- Reduces overfitting.
- High performance without much tuning.

### gradient Boosting (XGBoost, LightGBM)

- Build sequential trees that correct previous errors.
- Status of the art in tabular Problems.

### K-Nearest Neighbors (KNN)

- Classification/regression based on nearest neighbors.
- Simple but computationally expensive.

### K-Means Clustering

- Unsupervised clustering.
- Groups Data into `k` clusters.

📹 **Videos recommended:**

1. [Decision Trees - StatQuest](https://www.youtube.com/watch?v=7VeUPuFGJHk) - 20 min
1. [Random Forest - StatQuest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) - 10 min
1. [Gradient Boosting - StatQuest](https://www.youtube.com/watch?v=3CC4N4z3GJc) - 15 min
1. [K-Means Clustering - StatQuest](https://www.youtube.com/watch?v=4b5d3muPQmA) - 9 min

📚 **Resources written:**

- [Scikit-learn Algorithm Cheatsheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Towards Data Science - ML Algorithms](https://towardsdatascience.com/a-tour-of-machine-learning-algorithms-466b8bf75c0a)

______________________________________________________________________

## 7. Cross Validation and Tuning

### Cross-Validation (K-Fold)

Divide training set into `k` parts (folds). Train `k` times, each time using a different part like Validation.

**Advantage:** Better performance estimation. Reduces variance.

### Hyperparameter Tuning

**Grid Search:** test all the defined Hyperparameters combinations.
**Random Search:** try random combinations (more efficient).
**Bayesian Optimization:** intelligent search guided by previous Results.

📹 **Videos recommended:**

1. [Cross Validation - StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw) - 6 min
1. [Hyperparameter Tuning - Krish Naik](https://www.youtube.com/watch?v=gfUT7iUt0yM) - 25 min

______________________________________________________________________

## 8. Buenas Practices

- ✅ Start with a simple baseline before making it more complex.
- ✅ Validate yourself with cross-validation before playing test set.
- ✅ Document decisions (why you chose a Model, what Metrics matter).
- ✅ Use scikit-learn pipelines for reproducibility.
- ✅ Monitor both Training and Validation Metrics.
- ✅ Do not optimize Hyperparameters by looking at test set.
- ✅ Try multiple Models before deciding.

📚 **General resources:**

- [Hands-On ML with Scikit-Learn (Book)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Learn - Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning)

______________________________________________________________________

## Final comprehension checklist

Before moving to the next Module, you should be able to:

- ✅ Differentiate regression vs Classification and choose according to Problem.
- ✅ Build a complete pipeline from Raw Data to Prediction.
- ✅ Select appropriate Metrics according to business impact.
- ✅ Detect overfitting/underfitting in Learning graphs.
- ✅ Prevent data leakage by applying transformations correctly.
- ✅ Use cross-validation to validate Models in a robust way.
- ✅ Compare multiple Algorithms and justify your final choice.
- ✅ Interpret Results and communicate them to non-technical stakeholders.

If you answered "yes" to all, you're ready for deep learning.
