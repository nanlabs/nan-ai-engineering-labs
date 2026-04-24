# Práctica 02 — Árboles de Decisión y Random Forests

## 🎯 Objetivos

- Entrenar árboles de decisión y entender splits
- Implementar Random Forest y ensemble methods
- Comparar modelos y entender trade-offs
- Optimizar hiperparámetros con Grid Search

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Árbol de Decisión Básico

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar con diferentes profundidades
depths = [2, 3, 5, None]

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for idx, depth in enumerate(depths):
    # Entrenar
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    # Evaluar
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Visualizar árbol
    plot_tree(tree, feature_names=iris.feature_names,
              class_names=iris.target_names, filled=True, ax=axes[idx])
    axes[idx].set_title(f'Decision Tree (depth={depth}) - Accuracy: {acc:.4f}')

plt.tight_layout()
plt.savefig('decision_trees_depths.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Árboles de decisión entrenados con diferentes profundidades")
```

**✅ Solución - Feature Importance:**

```python
# Entrenar árbol final
tree_final = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_final.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': tree_final.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
print(importances)

# Visualizar
plt.figure(figsize=(10, 6))
plt.barh(importances['feature'], importances['importance'], color='steelblue', alpha=0.7)
plt.xlabel('Importance')
plt.title('Feature Importance - Decision Tree')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('tree_feature_importance.png', dpi=150)
plt.show()
```

### Ejercicio 1.2: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Entrenar Random Forest con diferentes n_estimators
n_estimators_list = [10, 50, 100, 200]

results = []

for n_est in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluar
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)

    results.append({
        'n_estimators': n_est,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    })

    print(f"n_estimators={n_est:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")

# Visualizar learning curves
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['train_accuracy'],
         marker='o', label='Train Accuracy', linewidth=2)
plt.plot(results_df['n_estimators'], results_df['test_accuracy'],
         marker='s', label='Test Accuracy', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest: Accuracy vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('random_forest_learning_curve.png', dpi=150)
plt.show()
```

### Ejercicio 1.3: Comparación de Modelos

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Entrenar múltiples modelos
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Naive Bayes': GaussianNB()
}

comparison = []

for name, model in models.items():
    # Entrenar
    model.fit(X_train, y_train)

    # Evaluar
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    comparison.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Overfitting Gap': train_acc - test_acc
    })

comparison_df = pd.DataFrame(comparison).sort_values('Test Accuracy', ascending=False)

print("\n=== Model Comparison ===")
print(comparison_df.to_string(index=False))

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
x = np.arange(len(comparison_df))
width = 0.35

axes[0].bar(x - width/2, comparison_df['Train Accuracy'], width,
            label='Train', alpha=0.7, color='steelblue')
axes[0].bar(x + width/2, comparison_df['Test Accuracy'], width,
            label='Test', alpha=0.7, color='coral')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Overfitting analysis
axes[1].bar(comparison_df['Model'], comparison_df['Overfitting Gap'],
            color='orange', alpha=0.7)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Overfitting Gap (Train - Test)')
axes[1].set_title('Overfitting Analysis')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Ejercicio 1.4: Hyperparameter Tuning con Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Definir grid de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)

print("\n=== Grid Search en ejecución ===")
grid_search.fit(X_train, y_train)

print("\n=== Best Parameters ===")
print(grid_search.best_params_)
print(f"\nBest CV Score: {grid_search.best_score_:.4f}")

# Evaluar mejor modelo
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test Score: {test_score:.4f}")

# Analizar resultados del grid search
results_df = pd.DataFrame(grid_search.cv_results_)

# Top 10 configuraciones
top_configs = results_df.nsmallest(10, 'rank_test_score')[
    ['param_n_estimators', 'param_max_depth', 'param_min_samples_split',
     'mean_test_score', 'std_test_score']
]

print("\n=== Top 10 Configurations ===")
print(top_configs.to_string(index=False))
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Visualización de Decision Boundaries

**Enunciado:**
Usa solo 2 features (petal length y petal width) y visualiza:

1. Decision boundary para Decision Tree (depth=3)
1. Decision boundary para Random Forest (n_estimators=100)
1. Grafica puntos de train/test con colores por clase
1. Compara cómo difieren los boundaries

### Ejercicio 2.2: Feature Engineering para Árboles

**Enunciado:**
Crea nuevas features derivadas:

- Ratios entre features (ej: sepal_length / sepal_width)
- Interacciones (ej: feature1 * feature2)
- Categorías binned (ej: size groups: small/medium/large)

Compara performance con y sin feature engineering.

### Ejercicio 2.3: Out-of-Bag (OOB) Score

**Enunciado:**
Random Forest usa bootstrap samples. Cada árbol no ve ~37% de datos (OOB).
Entrena RandomForest con `oob_score=True` y compara:

- OOB Score
- Train Score
- Test Score

¿Cuál es un mejor estimado del test performance?

### Ejercicio 2.4: Ensemble Voting Classifier

**Enunciado:**
Crea un `VotingClassifier` (sklearn) que combine:

- Decision Tree
- Random Forest
- SVM
- Logistic Regression

Usa voting='soft' (promedia probabilidades).
Compara accuracy del ensemble vs modelos individuales.

### Ejercicio 2.5: Análisis de Árboles Individuales

**Enunciado:**
Extrae 3 árboles del Random Forest entrenado:

```python
trees = rf.estimators_[:3]
```

Visualiza cada árbol con `plot_tree`.
Analiza: ¿usan las mismas features en la raíz?
¿Diferentes estructuras debido a randomness?

______________________________________________________________________

## ✅ Checklist de Competencias

- [ ] Entrenar y visualizar árboles de decisión
- [ ] Interpretar splits y feature importance
- [ ] Entrenar Random Forest con n_estimators apropiado
- [ ] Comparar múltiples algoritmos de ML
- [ ] Detectar overfitting (train vs test gap)
- [ ] Aplicar Grid Search para tuning de hiperparámetros
- [ ] Usar cross-validation para evaluación robusta
- [ ] Entender trade-off bias-variance

______________________________________________________________________

## 📚 Recursos

- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [Sklearn: Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
