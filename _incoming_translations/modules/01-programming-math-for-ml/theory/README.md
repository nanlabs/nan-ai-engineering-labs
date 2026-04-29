# Theory — Programming and Mathematics for ML

## Why this module matters

Without a mathematical and programming foundation, using ML Models becomes a black box. This Module gives you tools to understand what a Model does internally, why it fails and how to improve it with a solid technical foundation.

______________________________________________________________________

## 1. Essential Python for ML (from scratch)

### Python Fundamentals

- **Data Types:** `int`, `float`, `str`, `list`, `dict`, `tuple`.
- **Flow control:** `if`, `for`, `while`, Functions, list comprehensions.
- **Key concepts:** variables, operators, Modules, basic Error handling.

### Fundamental libraries

- **NumPy:** efficient numerical operations, vectors, matrices, broadcasting.
- **Pandas:** manipulation of tabular Data, DataFrames, Series, Cleaning operations.
- **Matplotlib / Seaborn:** Data visualization, line graphs, scatter, histograms.
- **Scikit-learn:** Introduction to basic ML pipelines.

### Work tools

- **Jupyter Notebooks:** interactive environment for experimentation.
- **Google Colab:** cloud notebooks with free GPUs.
- **Reproducibilidad:** guardar code versionado, semillas aleatorias, entornos virtuales.

📹 **Recommended videos (watch in this order):**

1. [Python for Beginners - Microsoft](https://www.youtube.com/playlist?list=PLlrxD0HtieHhS8VzuMCfQD4uJ9yne1mE6) - 44 videos cortos
1. [NumPy Tutorial - freeCodeCamp](https://www.youtube.com/watch?v=QUT1VHiLmmI) - 1 hora
1. [Pandas Tutorial - Corey Schafer](https://www.youtube.com/playlist?list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS) - 10 videos

📚 **Resources written:**

- [Python Data Science Handbook (Jake VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/) - gratuito online
- [NumPy Documentation - Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)

______________________________________________________________________

## 2. Applied linear algebra

### Concepts fundamentales

- **Scalars, vectors and matrices:** representation and notation.
- **Operations with vectors:**
  - Add and subtract element by element
  - Production point (dot product): similarity entre vectors
  - Norm of a vector: magnitude/length
- **Operations with matrices:**
  - Addition/subtraction of matrices
- Matrix-vector and matrix-matrix multiplication
  - Transpose of a matrix
  - Identidad e inversa

### Intuition for ML

- A linear Model combines Features using weights (product point).
- neural networks are sequences of matrix-vector multiplications + non-linear Functions.
- Eigenvalues ​​and eigenvectors appear in PCA (dimensionality reduction).

📹 **Videos recommended (view in order):**

1. [Essence of Linear Algebra - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) - Series complete (15 videos, ~3 horas)
1. [Linear Algebra for Machine Learning - Imperial College (Coursera)](https://www.coursera.org/learn/linear-algebra-machine-learning) (cursor opcional certificable)

📚 **Resources written:**

- [Linear Algebra Review - Stanford CS229](http://cs229.stanford.edu/section/cs229-linalg.pdf)
- [NumPy for Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)

______________________________________________________________________

## 3. Calculation applied to optimization

### Concepts clave

- **Derived:** measures the local rate of change of a Function.
- **Partial derivative:** change with respect to a single variable (when there are multiple).
- **Chain rule:** conceptual basis of the backpropagation Algorithm.
- **Gradient:** vector of partial derivatives; points in the direction of maximum growth.

### Intuition for ML

- If you know the gradient of the error with respect to the Model parameters, you can adjust them in the direction that reduces the error.
- Training neural networks is essentially calculating gradients and updating weights.

📹 **Videos recommended (view in order):**

1. [Essence of Calculus - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - Series complete (12 videos, ~3 horas)
1. [What is Backpropagation? - 3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U) - 14 min

📚 **Resources written:**

- [Calculus for ML - Khan Academy](https://www.khanacademy.org/math/calculus-1)
- [Backpropagation Explained - Stanford CS231n](http://cs231n.github.io/optimization-2/)

______________________________________________________________________

## 4. Probability and statistics

### Concepts fundamentales

- **Trend central measurements:** mean, median, mode.
- **Dispersion measures:** variance, standard deviation, quartiles, interquartile range.
- **Distribuciones:** normal (gaussiana), binomial, uniforme, Poisson.
- **Conditional probability and Bayes' Theorem.**
- **Correlation vs causality.**

### Intuition for ML

- Understanding the distribution of your Data helps you choose the correct Model and detect outliers.
- Prediction uncertainty is quantified using probability.
- Naive Bayes, one of the simplest Classification Algorithms, uses Bayes' Theorem directly.

📹 **Videos recommended:**

1. [StatQuest - Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) - Series complete
1. [Probability - Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library)

📚 **Resources written:**

- [Think Stats (Allen Downey)](https://greenteapress.com/thinkstats2/html/index.html) - gratuito
- [Seeing Theory - Visual Probability](https://seeing-theory.brown.edu/)

______________________________________________________________________

## 5. Optimization and gradient descent

### Concepts clave

- **Cost function (loss function):** measures how poorly the Model predicts.
- **Gradiente descent:**
  - Iterative algorithm to minimize the cost function.
- Learning rate: how much to adjust the parameters in each step.
- Convergence: when to stop training.
- **Variants:** SGD (stochastic), mini-batch, momentum, Adam.

### Intuition for ML

The Training of almost all modern Models (neural networks, logistic regression, etc.) is based on iterative optimization using gradient descent.

📹 **Videos recommended:**

1. [Gradient Descent - StatQuest](https://www.youtube.com/watch?v=sDv4f4s2SB8) - 9 min
1. [Optimizers Explained - Andrej Karpathy](https://www.youtube.com/watch?v=IHZwWFHWa-w) - 46 min

📚 **Resources written:**

- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
- [CS231n - Optimization](http://cs231n.github.io/optimization-1/)

______________________________________________________________________

## 6. Reproducibility and good practices

### Fundamentos

- Fijar semillas aleatorias (`np.random.seed`, `random.seed`, `torch.manual_seed`).
- Separar Data correctly: train / validation / test.
- Document experiments in `notes/` of the Module.
- Use Git to version code and decisions.
- Use virtual environments (`venv`, `conda`) for package reproducibility.

📚 **Resources:**

- [Reproducible ML - Google Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

______________________________________________________________________

## Final comprehension checklist

Before moving on to the next Module, you should be able to answer:

- ✅ Can you explain what a derivative is with a visual Example?
- ✅ Can you calculate a product point in Python with NumPy?
- ✅ Can you interpret the mean, variance and standard deviation of a real dataset?
- ✅ Can you implement a parameter update with gradient descent by hand?
- ✅ Do you understand why setting a random seed is important for reproducibility?

If you answered "yes" to all, you are ready to move forward.
