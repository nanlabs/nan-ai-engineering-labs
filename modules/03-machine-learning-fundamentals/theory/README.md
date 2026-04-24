# Theory — Machine Learning Fundamentals

## Why this module matters

Machine Learning es la base de los sistemas de IA modernos. Entender los fundamentos te permite elegir el Algorithm correcto, evaluar Results de forma crítica y diagnosticar Problems en producción. Este Module construye la intuición necesaria para trabajar con Models supervisados y no supervisados.

______________________________________________________________________

## 1. ¿Qué es Machine Learning?

Machine Learning (ML) es el campo de la IA que permite a los sistemas aprender patrones a partir de Data para hacer Predictions o tomar decisiones **sin ser explícitamente programados**.

### Diferencia clave con programación tradicional

- **Programación tradicional:** humano define reglas explícitas (if/else).
- **Machine Learning:** el sistema aprende reglas a partir de Examples.

______________________________________________________________________

## 2. Types de Machine Learning

### Learning Supervisado

**Definición:** Tienes Data con etiquetas conocidas (`y`). El Model aprende la relación entre entradas (`X`) y salidas (`y`).

**Subtipos:**

- **Regresión:** predecir valores continuos (precio de casa, temperatura).
  - Algorithms: Regresión Lineal, Regresión Polinomial, Random Forest Regressor.
- **Classification:** predecir clases discretas (spam/no spam, gato/perro).
  - Algorithms: Regresión Logística, Decision Trees, SVM, Random Forest, gradient Boosting.

### Learning No Supervisado

**Definición:** No hay etiquetas (`y`). El Model encuentra patrones o Structure oculta en los Data.

**Subtipos:**

- **Clustering:** agrupar Data similares (segmentación de clientes).
  - Algorithms: K-Means, DBSCAN, Hierarchical Clustering.
- **Reducción de dimensionalidad:** comprimir features manteniendo información.
  - Algorithms: PCA, t-SNE, UMAP.

### Learning por Refuerzo (Introduction)

El agente aprende a tomar decisiones mediante Testing y error, recibiendo recompensas o penalizaciones.

📹 **Videos recomendados:**

1. [Machine Learning Crash Course - Google](https://www.youtube.com/watch?v=nKW8Ndu7Mjw) - 40 min
1. [StatQuest: Machine Learning Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) - serie completa
1. [ML Course - Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning) - curso completo certificable

______________________________________________________________________

## 3. Pipeline típico de ML

### Paso 1: Definir el Problem

- ¿Qué quiero predecir? (target/Objective).
- ¿Es regresión o Classification?
- ¿Qué Metric de éxito usar?

### Paso 2: Recolectar y explorar Data

- EDA (Exploratory Data Analysis).
- Identificar calidad de Data, outliers, distribuciones.

### Paso 3: Preparar Data

- Cleaning: nulos, duplicados, outliers.
- Feature engineering: crear nuevas features.
- Encoding: convertir categóricas a numéricas.
- Scaling: normalizar o estandarizar.

### Paso 4: Dividir Data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Importante:** **NUNCA** usar Data de test durante Training o tuning.

### Paso 5: Entrenar Model baseline

Empezar con el Model más simple posible para establecer un punto de referencia.

### Paso 6: Evaluar con Metrics correctas

Elegir Metric según el Problem y costo de Errors.

### Paso 7: Ajustar y mejorar

- Probar otros Algorithms.
- Ajustar Hyperparameters.
- Validation cruzada.
- Feature engineering adicional.

### Paso 8: Validation final

Evaluar en test set **una sola vez** al final.

📹 **Videos recomendados:**

1. [ML Workflow Explained - Krish Naik](https://www.youtube.com/watch?v=fiz1ORTBGpY) - 25 min
1. [Train/Test Split - StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw) - 8 min

______________________________________________________________________

## 4. Metrics de Evaluation

### Para Problems de Regresión

- **MAE (Mean Absolute error):** promedio del error absoluto. Fácil de interpretar.
- **MSE (Mean Squared error):** penaliza Errors grandes más fuertemente.
- **RMSE (Root Mean Squared error):** raíz de MSE, en mismas unidades que `y`.
- **R² (R-squared):** % de varianza explicada por el Model (0-1, más alto mejor).

### Para Problems de Classification

- **accuracy:** % de Predictions correctas. **Cuidado con datasets desbalanceados.**
- **Precision:** de los que predije como positivos, ¿cuántos lo son realmente?
- **recall (Sensitivity):** de los realmente positivos, ¿cuántos detecté?
- **f1-Score:** media armónica de precision y recall.
- **auc-ROC:** área bajo la ROC curve. Mide capacidad de discriminación del Model.

### Cómo elegir Metric

- **Regresión:** RMSE si outliers importan, MAE si querés robustez.
- **Classification balanceada:** accuracy puede bastar.
- **Classification desbalanceada:** usar f1, precision/recall según contexto.
  - Example: detección de fraude → priorizar recall (no perder fraudes).
  - Example: Filter de spam → balance entre precision y recall.

📹 **Videos recomendados:**

1. [Regression Metrics - StatQuest](https://www.youtube.com/watch?v=lgZ-s4XNPcs) - 10 min
1. [Classification Metrics - StatQuest](https://www.youtube.com/watch?v=4jRBRDbJemM) - 15 min
1. [Confusion Matrix Explained - Krish Naik](https://www.youtube.com/watch?v=wpp3VfzgNcI) - 12 min

📚 **Resources escritos:**

- [Scikit-learn Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Google ML Crash Course - Classification](https://developers.google.com/machine-learning/crash-course/classification)

______________________________________________________________________

## 5. Concepts críticos

### overfitting (sobreajuste)

**Síntoma:** El Model aprende demasiado bien el training set, incluyendo ruido y detalles irrelevantes. Performance alta en train, baja en test.

**Causas:**

- Model demasiado complejo para la cantidad de Data.
- Entrenar por demasiadas iteraciones.
- No usar Regularization.

**Soluciones:**

- Conseguir más Data.
- Simplificar el Model (menos features, menos profundidad).
- Regularization (L1, L2).
- Early stopping.
- Cross-validation.

### underfitting (subajuste)

**Síntoma:** El Model es demasiado simple y no captura patrones relevantes. Performance baja en train y test.

**Causas:**

- Model demasiado simple.
- Features insuficientes o poco informativas.

**Soluciones:**

- Usar Model más complejo.
- Agregar más features.
- Reducir Regularization.

### Bias-Variance Tradeoff

- **Bias alto:** underfitting (Model muy simple).
- **Variance alta:** overfitting (Model muy sensible a Data de Training).
- **Objective:** encontrar balance óptimo.

### Data Leakage

**Definición:** Usar información del futuro o del test set durante Training.

**Examples comunes:**

- Normalizar antes de hacer train/test split.
- Usar features que incluyen el target indirectamente.
- Entrenar con Data posteriores al momento de Prediction.

**Prevención:**

- Siempre dividir Data PRIMERO.
- Aplicar transformaciones solo en train, luego aplicar al test.
- Revisar features que tengan correlación perfecta con el target.

📹 **Videos recomendados:**

1. [Overfitting and Underfitting - StatQuest](https://www.youtube.com/watch?v=EuBBz3bI-aA) - 14 min
1. [Bias-Variance Tradeoff - Krish Naik](https://www.youtube.com/watch?v=EuBBz3bI-aA) - 20 min
1. [Data Leakage Explained - Kaggle](https://www.youtube.com/watch?v=jmUCgGDsG7g) - 10 min

______________________________________________________________________

## 6. Algorithms fundamentales

### Regresión Lineal

- Model más simple. Asume relación lineal entre `X` e `y`.
- Interpretable y rápido.

### Regresión Logística

- Classification binaria.
- Salida: probabilidad (0-1).

### Decision Trees

- Splits sucesivos basados en features.
- Fácil de interpretar.
- Propenso a overfitting.

### Random Forest

- Ensemble de múltiples árboles.
- Reduce overfitting.
- Alta performance sin mucho tuning.

### gradient Boosting (XGBoost, LightGBM)

- Construye árboles secuenciales que corrigen Errors previos.
- Estado del arte en Problems tabulares.

### K-Nearest Neighbors (KNN)

- Classification/regresión basada en vecinos más cercanos.
- Simple pero costoso computacionalmente.

### K-Means Clustering

- Clustering no supervisado.
- Agrupa Data en `k` clusters.

📹 **Videos recomendados:**

1. [Decision Trees - StatQuest](https://www.youtube.com/watch?v=7VeUPuFGJHk) - 20 min
1. [Random Forest - StatQuest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) - 10 min
1. [Gradient Boosting - StatQuest](https://www.youtube.com/watch?v=3CC4N4z3GJc) - 15 min
1. [K-Means Clustering - StatQuest](https://www.youtube.com/watch?v=4b5d3muPQmA) - 9 min

📚 **Resources escritos:**

- [Scikit-learn Algorithm Cheatsheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Towards Data Science - ML Algorithms](https://towardsdatascience.com/a-tour-of-machine-learning-algorithms-466b8bf75c0a)

______________________________________________________________________

## 7. Validation Cruzada y Tuning

### Cross-Validation (K-Fold)

Dividir training set en `k` partes (folds). Entrenar `k` veces, cada vez usando una parte diferente como Validation.

**Ventaja:** Mejor estimación de performance. Reduce varianza.

### Hyperparameter Tuning

**Grid Search:** probar todas las combinaciones de Hyperparameters definidas.
**Random Search:** probar combinaciones aleatorias (más eficiente).
**Bayesian Optimization:** búsqueda inteligente guiada por Results previos.

📹 **Videos recomendados:**

1. [Cross Validation - StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw) - 6 min
1. [Hyperparameter Tuning - Krish Naik](https://www.youtube.com/watch?v=gfUT7iUt0yM) - 25 min

______________________________________________________________________

## 8. Buenas Practices

- ✅ Empezar con baseline simple antes de complejizar.
- ✅ Validar con cross-validation antes de tocar test set.
- ✅ Documentar decisiones (por qué elegiste un Model, qué Metrics importan).
- ✅ Usar pipelines de scikit-learn para reproducibilidad.
- ✅ Monitorear tanto Metrics de Training como Validation.
- ✅ No optimizar Hyperparameters mirando test set.
- ✅ Probar múltiples Models antes de decidir.

📚 **Resources generales:**

- [Hands-On ML with Scikit-Learn (Book)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Learn - Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning)

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente Module, deberías poder:

- ✅ Diferenciar regresión vs Classification y elegir según Problem.
- ✅ Construir un pipeline completo desde Data crudos hasta Prediction.
- ✅ Seleccionar Metrics apropiadas según impacto de negocio.
- ✅ Detectar overfitting/underfitting en gráficas de Learning.
- ✅ Prevenir data leakage aplicando transformaciones correctamente.
- ✅ Usar cross-validation para validar Models de forma robusta.
- ✅ Comparar múltiples Algorithms y justificar tu elección final.
- ✅ Interpretar Results y comunicarlos a stakeholders no técnicos.

Si respondiste "sí" a todas, estás listo para deep learning.
