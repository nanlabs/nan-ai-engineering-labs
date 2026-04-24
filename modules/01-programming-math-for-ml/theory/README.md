# Theory — Programación y Matemática para ML

## Why this module matters

Sin base matemática y de programación, usar modelos de ML se vuelve una caja negra. Este módulo te da herramientas para entender qué hace un modelo internamente, por qué falla y cómo mejorarlo con fundamento técnico sólido.

______________________________________________________________________

## 1. Python esencial para ML (desde cero)

### Fundamentos de Python

- **Tipos de datos:** `int`, `float`, `str`, `list`, `dict`, `tuple`.
- **Control de flujo:** `if`, `for`, `while`, funciones, list comprehensions.
- **Conceptos clave:** variables, operadores, módulos, manejo de errores básico.

### Librerías fundamentales

- **NumPy:** operaciones numéricas eficientes, vectores, matrices, broadcasting.
- **Pandas:** manipulación de datos tabulares, DataFrames, Series, operaciones de limpieza.
- **Matplotlib / Seaborn:** visualización de datos, gráficos de línea, dispersión, histogramas.
- **Scikit-learn:** introducción a pipelines básicos de ML.

### Herramientas de trabajo

- **Jupyter Notebooks:** ambiente interactivo para experimentación.
- **Google Colab:** notebooks en la nube con GPUs gratuitas.
- **Reproducibilidad:** guardar código versionado, semillas aleatorias, entornos virtuales.

📹 **Videos recomendados (ver en este orden):**

1. [Python for Beginners - Microsoft](https://www.youtube.com/playlist?list=PLlrxD0HtieHhS8VzuMCfQD4uJ9yne1mE6) - 44 videos cortos
1. [NumPy Tutorial - freeCodeCamp](https://www.youtube.com/watch?v=QUT1VHiLmmI) - 1 hora
1. [Pandas Tutorial - Corey Schafer](https://www.youtube.com/playlist?list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS) - 10 videos

📚 **Recursos escritos:**

- [Python Data Science Handbook (Jake VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/) - gratuito online
- [NumPy Documentation - Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)

______________________________________________________________________

## 2. Álgebra lineal aplicada

### Conceptos fundamentales

- **Escalares, vectores y matrices:** representación y notación.
- **Operaciones con vectores:**
  - Suma y resta elemento a elemento
  - Producto punto (dot product): similitud entre vectores
  - Norma de un vector: magnitud/longitud
- **Operaciones con matrices:**
  - Suma/resta de matrices
  - Multiplicación matriz-vector y matriz-matriz
  - Transpuesta de una matriz
  - Identidad e inversa

### Intuición para ML

- Un modelo lineal combina características usando pesos (producto punto).
- Redes neuronales son secuencias de multiplicaciones matriz-vector + funciones no lineales.
- Eigenvalues y eigenvectors aparecen en PCA (reducción de dimensionalidad).

📹 **Videos recomendados (ver en orden):**

1. [Essence of Linear Algebra - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) - Serie completa (15 videos, ~3 horas)
1. [Linear Algebra for Machine Learning - Imperial College (Coursera)](https://www.coursera.org/learn/linear-algebra-machine-learning) (curso opcional certificable)

📚 **Recursos escritos:**

- [Linear Algebra Review - Stanford CS229](http://cs229.stanford.edu/section/cs229-linalg.pdf)
- [NumPy for Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)

______________________________________________________________________

## 3. Cálculo aplicado a optimización

### Conceptos clave

- **Derivada:** mide la tasa de cambio local de una función.
- **Derivada parcial:** cambio respecto a una sola variable (cuando hay múltiples).
- **Regla de la cadena:** base conceptual del algoritmo backpropagation.
- **Gradiente:** vector de derivadas parciales; apunta en dirección de máximo crecimiento.

### Intuición para ML

- Si conocés el gradiente del error respecto a los parámetros del modelo, podés ajustarlos en la dirección que reduce el error.
- El entrenamiento de redes neuronales es esencialmente calcular gradientes y actualizar pesos.

📹 **Videos recomendados (ver en orden):**

1. [Essence of Calculus - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - Serie completa (12 videos, ~3 horas)
1. [What is Backpropagation? - 3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U) - 14 min

📚 **Recursos escritos:**

- [Calculus for ML - Khan Academy](https://www.khanacademy.org/math/calculus-1)
- [Backpropagation Explained - Stanford CS231n](http://cs231n.github.io/optimization-2/)

______________________________________________________________________

## 4. Probabilidad y estadística

### Conceptos fundamentales

- **Medidas de tendencia central:** media, mediana, moda.
- **Medidas de dispersión:** varianza, desvío estándar, cuartiles, rango intercuartílico.
- **Distribuciones:** normal (gaussiana), binomial, uniforme, Poisson.
- **Probabilidad condicional y Teorema de Bayes.**
- **Correlación vs causalidad.**

### Intuición para ML

- Entender la distribución de tus datos te ayuda a elegir el modelo correcto y detectar outliers.
- La incertidumbre de predicción se cuantifica usando probabilidad.
- Naive Bayes, uno de los algoritmos de clasificación más simples, usa el Teorema de Bayes directamente.

📹 **Videos recomendados:**

1. [StatQuest - Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) - Serie completa
1. [Probability - Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library)

📚 **Recursos escritos:**

- [Think Stats (Allen Downey)](https://greenteapress.com/thinkstats2/html/index.html) - gratuito
- [Seeing Theory - Visual Probability](https://seeing-theory.brown.edu/)

______________________________________________________________________

## 5. Optimización y gradiente descendente

### Conceptos clave

- **Función de costo (loss function):** mide qué tan mal predice el modelo.
- **Gradiente descendente:**
  - Algoritmo iterativo para minimizar la función de costo.
  - Tasa de aprendizaje (learning rate): cuánto ajustar los parámetros en cada paso.
  - Convergencia: cuándo detener el entrenamiento.
- **Variantes:** SGD (stochastic), mini-batch, momentum, Adam.

### Intuición para ML

El entrenamiento de casi todos los modelos modernos (redes neuronales, regresión logística, etc.) se basa en optimización iterativa usando gradiente descendente.

📹 **Videos recomendados:**

1. [Gradient Descent - StatQuest](https://www.youtube.com/watch?v=sDv4f4s2SB8) - 9 min
1. [Optimizers Explained - Andrej Karpathy](https://www.youtube.com/watch?v=IHZwWFHWa-w) - 46 min

📚 **Recursos escritos:**

- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
- [CS231n - Optimization](http://cs231n.github.io/optimization-1/)

______________________________________________________________________

## 6. Reproducibilidad y buenas prácticas

### Fundamentos

- Fijar semillas aleatorias (`np.random.seed`, `random.seed`, `torch.manual_seed`).
- Separar datos correctamente: train / validation / test.
- Documentar experimentos en `notes/` del módulo.
- Usar Git para versionar código y decisiones.
- Usar ambientes virtuales (`venv`, `conda`) para reproducibilidad de paquetes.

📚 **Recursos:**

- [Reproducible ML - Google Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente módulo, deberías poder responder:

- ✅ ¿Podés explicar qué es una derivada con un ejemplo visual?
- ✅ ¿Podés calcular un producto punto en Python con NumPy?
- ✅ ¿Podés interpretar media, varianza y desvío estándar de un dataset real?
- ✅ ¿Podés implementar una actualización de parámetros con gradiente descendente a mano?
- ✅ ¿Entendés por qué fijar una semilla aleatoria es importante para reproducibilidad?

Si respondiste "sí" a todas, estás listo para avanzar.
