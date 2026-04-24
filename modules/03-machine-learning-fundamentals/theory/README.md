# Theory — Machine Learning Fundamentals

## Why this module matters

Machine Learning es la base de los sistemas de IA modernos. Entender los fundamentos te permite elegir el algoritmo correcto, evaluar resultados de forma crítica y diagnosticar problemas en producción. Este módulo construye la intuición necesaria para trabajar con modelos supervisados y no supervisados.

______________________________________________________________________

## 1. ¿Qué es Machine Learning?

Machine Learning (ML) es el campo de la IA que permite a los sistemas aprender patrones a partir de datos para hacer predicciones o tomar decisiones **sin ser explícitamente programados**.

### Diferencia clave con programación tradicional

- **Programación tradicional:** humano define reglas explícitas (if/else).
- **Machine Learning:** el sistema aprende reglas a partir de ejemplos.

______________________________________________________________________

## 2. Tipos de Machine Learning

### Aprendizaje Supervisado

**Definición:** Tienes datos con etiquetas conocidas (`y`). El modelo aprende la relación entre entradas (`X`) y salidas (`y`).

**Subtipos:**

- **Regresión:** predecir valores continuos (precio de casa, temperatura).
  - Algoritmos: Regresión Lineal, Regresión Polinomial, Random Forest Regressor.
- **Clasificación:** predecir clases discretas (spam/no spam, gato/perro).
  - Algoritmos: Regresión Logística, Decision Trees, SVM, Random Forest, Gradient Boosting.

### Aprendizaje No Supervisado

**Definición:** No hay etiquetas (`y`). El modelo encuentra patrones o estructura oculta en los datos.

**Subtipos:**

- **Clustering:** agrupar datos similares (segmentación de clientes).
  - Algoritmos: K-Means, DBSCAN, Hierarchical Clustering.
- **Reducción de dimensionalidad:** comprimir features manteniendo información.
  - Algoritmos: PCA, t-SNE, UMAP.

### Aprendizaje por Refuerzo (introducción)

El agente aprende a tomar decisiones mediante prueba y error, recibiendo recompensas o penalizaciones.

📹 **Videos recomendados:**

1. [Machine Learning Crash Course - Google](https://www.youtube.com/watch?v=nKW8Ndu7Mjw) - 40 min
1. [StatQuest: Machine Learning Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) - serie completa
1. [ML Course - Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning) - curso completo certificable

______________________________________________________________________

## 3. Pipeline típico de ML

### Paso 1: Definir el problema

- ¿Qué quiero predecir? (target/objetivo).
- ¿Es regresión o clasificación?
- ¿Qué métrica de éxito usar?

### Paso 2: Recolectar y explorar datos

- EDA (Exploratory Data Analysis).
- Identificar calidad de datos, outliers, distribuciones.

### Paso 3: Preparar datos

- Limpieza: nulos, duplicados, outliers.
- Feature engineering: crear nuevas features.
- Encoding: convertir categóricas a numéricas.
- Scaling: normalizar o estandarizar.

### Paso 4: Dividir datos

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Importante:** **NUNCA** usar datos de test durante entrenamiento o tuning.

### Paso 5: Entrenar modelo baseline

Empezar con el modelo más simple posible para establecer un punto de referencia.

### Paso 6: Evaluar con métricas correctas

Elegir métrica según el problema y costo de errores.

### Paso 7: Ajustar y mejorar

- Probar otros algoritmos.
- Ajustar hiperparámetros.
- Validación cruzada.
- Feature engineering adicional.

### Paso 8: Validación final

Evaluar en test set **una sola vez** al final.

📹 **Videos recomendados:**

1. [ML Workflow Explained - Krish Naik](https://www.youtube.com/watch?v=fiz1ORTBGpY) - 25 min
1. [Train/Test Split - StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw) - 8 min

______________________________________________________________________

## 4. Métricas de evaluación

### Para problemas de Regresión

- **MAE (Mean Absolute Error):** promedio del error absoluto. Fácil de interpretar.
- **MSE (Mean Squared Error):** penaliza errores grandes más fuertemente.
- **RMSE (Root Mean Squared Error):** raíz de MSE, en mismas unidades que `y`.
- **R² (R-squared):** % de varianza explicada por el modelo (0-1, más alto mejor).

### Para problemas de Clasificación

- **Accuracy:** % de predicciones correctas. **Cuidado con datasets desbalanceados.**
- **Precision:** de los que predije como positivos, ¿cuántos lo son realmente?
- **Recall (Sensitivity):** de los realmente positivos, ¿cuántos detecté?
- **F1-Score:** media armónica de precision y recall.
- **AUC-ROC:** área bajo la curva ROC. Mide capacidad de discriminación del modelo.

### Cómo elegir métrica

- **Regresión:** RMSE si outliers importan, MAE si querés robustez.
- **Clasificación balanceada:** accuracy puede bastar.
- **Clasificación desbalanceada:** usar F1, precision/recall según contexto.
  - Ejemplo: detección de fraude → priorizar recall (no perder fraudes).
  - Ejemplo: filtro de spam → balance entre precision y recall.

📹 **Videos recomendados:**

1. [Regression Metrics - StatQuest](https://www.youtube.com/watch?v=lgZ-s4XNPcs) - 10 min
1. [Classification Metrics - StatQuest](https://www.youtube.com/watch?v=4jRBRDbJemM) - 15 min
1. [Confusion Matrix Explained - Krish Naik](https://www.youtube.com/watch?v=wpp3VfzgNcI) - 12 min

📚 **Recursos escritos:**

- [Scikit-learn Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Google ML Crash Course - Classification](https://developers.google.com/machine-learning/crash-course/classification)

______________________________________________________________________

## 5. Conceptos críticos

### Overfitting (sobreajuste)

**Síntoma:** El modelo aprende demasiado bien el training set, incluyendo ruido y detalles irrelevantes. Performance alta en train, baja en test.

**Causas:**

- Modelo demasiado complejo para la cantidad de datos.
- Entrenar por demasiadas iteraciones.
- No usar regularización.

**Soluciones:**

- Conseguir más datos.
- Simplificar el modelo (menos features, menos profundidad).
- Regularización (L1, L2).
- Early stopping.
- Cross-validation.

### Underfitting (subajuste)

**Síntoma:** El modelo es demasiado simple y no captura patrones relevantes. Performance baja en train y test.

**Causas:**

- Modelo demasiado simple.
- Features insuficientes o poco informativas.

**Soluciones:**

- Usar modelo más complejo.
- Agregar más features.
- Reducir regularización.

### Bias-Variance Tradeoff

- **Bias alto:** underfitting (modelo muy simple).
- **Variance alta:** overfitting (modelo muy sensible a datos de entrenamiento).
- **Objetivo:** encontrar balance óptimo.

### Data Leakage

**Definición:** Usar información del futuro o del test set durante entrenamiento.

**Ejemplos comunes:**

- Normalizar antes de hacer train/test split.
- Usar features que incluyen el target indirectamente.
- Entrenar con datos posteriores al momento de predicción.

**Prevención:**

- Siempre dividir datos PRIMERO.
- Aplicar transformaciones solo en train, luego aplicar al test.
- Revisar features que tengan correlación perfecta con el target.

📹 **Videos recomendados:**

1. [Overfitting and Underfitting - StatQuest](https://www.youtube.com/watch?v=EuBBz3bI-aA) - 14 min
1. [Bias-Variance Tradeoff - Krish Naik](https://www.youtube.com/watch?v=EuBBz3bI-aA) - 20 min
1. [Data Leakage Explained - Kaggle](https://www.youtube.com/watch?v=jmUCgGDsG7g) - 10 min

______________________________________________________________________

## 6. Algoritmos fundamentales

### Regresión Lineal

- Modelo más simple. Asume relación lineal entre `X` e `y`.
- Interpretable y rápido.

### Regresión Logística

- Clasificación binaria.
- Salida: probabilidad (0-1).

### Decision Trees

- Splits sucesivos basados en features.
- Fácil de interpretar.
- Propenso a overfitting.

### Random Forest

- Ensemble de múltiples árboles.
- Reduce overfitting.
- Alta performance sin mucho tuning.

### Gradient Boosting (XGBoost, LightGBM)

- Construye árboles secuenciales que corrigen errores previos.
- Estado del arte en problemas tabulares.

### K-Nearest Neighbors (KNN)

- Clasificación/regresión basada en vecinos más cercanos.
- Simple pero costoso computacionalmente.

### K-Means Clustering

- Clustering no supervisado.
- Agrupa datos en `k` clusters.

📹 **Videos recomendados:**

1. [Decision Trees - StatQuest](https://www.youtube.com/watch?v=7VeUPuFGJHk) - 20 min
1. [Random Forest - StatQuest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) - 10 min
1. [Gradient Boosting - StatQuest](https://www.youtube.com/watch?v=3CC4N4z3GJc) - 15 min
1. [K-Means Clustering - StatQuest](https://www.youtube.com/watch?v=4b5d3muPQmA) - 9 min

📚 **Recursos escritos:**

- [Scikit-learn Algorithm Cheatsheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Towards Data Science - ML Algorithms](https://towardsdatascience.com/a-tour-of-machine-learning-algorithms-466b8bf75c0a)

______________________________________________________________________

## 7. Validación Cruzada y Tuning

### Cross-Validation (K-Fold)

Dividir training set en `k` partes (folds). Entrenar `k` veces, cada vez usando una parte diferente como validación.

**Ventaja:** Mejor estimación de performance. Reduce varianza.

### Hyperparameter Tuning

**Grid Search:** probar todas las combinaciones de hiperparámetros definidas.
**Random Search:** probar combinaciones aleatorias (más eficiente).
**Bayesian Optimization:** búsqueda inteligente guiada por resultados previos.

📹 **Videos recomendados:**

1. [Cross Validation - StatQuest](https://www.youtube.com/watch?v=fSytzGwwBVw) - 6 min
1. [Hyperparameter Tuning - Krish Naik](https://www.youtube.com/watch?v=gfUT7iUt0yM) - 25 min

______________________________________________________________________

## 8. Buenas prácticas

- ✅ Empezar con baseline simple antes de complejizar.
- ✅ Validar con cross-validation antes de tocar test set.
- ✅ Documentar decisiones (por qué elegiste un modelo, qué métricas importan).
- ✅ Usar pipelines de scikit-learn para reproducibilidad.
- ✅ Monitorear tanto métricas de entrenamiento como validación.
- ✅ No optimizar hiperparámetros mirando test set.
- ✅ Probar múltiples modelos antes de decidir.

📚 **Recursos generales:**

- [Hands-On ML with Scikit-Learn (Book)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Learn - Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning)

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente módulo, deberías poder:

- ✅ Diferenciar regresión vs clasificación y elegir según problema.
- ✅ Construir un pipeline completo desde datos crudos hasta predicción.
- ✅ Seleccionar métricas apropiadas según impacto de negocio.
- ✅ Detectar overfitting/underfitting en gráficas de aprendizaje.
- ✅ Prevenir data leakage aplicando transformaciones correctamente.
- ✅ Usar cross-validation para validar modelos de forma robusta.
- ✅ Comparar múltiples algoritmos y justificar tu elección final.
- ✅ Interpretar resultados y comunicarlos a stakeholders no técnicos.

Si respondiste "sí" a todas, estás listo para deep learning.
