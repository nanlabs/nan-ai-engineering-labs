# 🚀 Quick Start Guide — Get started in 10 minutes

> **Your first day in the AI/ML program: Setup, first Module, and first steps**

______________________________________________________________________

## ⏱️ Timeline of your first day

```
┌──────────────────────────────────────────────────┐
│  Minuto 0-3: Setup del entorno                   │
├──────────────────────────────────────────────────┤
│  Minuto 3-8: Lee overview del Module 1           │
├──────────────────────────────────────────────────┤
│  Minuto 8-30: Primera practice con NumPy         │
├──────────────────────────────────────────────────┤
│  Minuto 30-60: Primer example executedo          │
└──────────────────────────────────────────────────┘

¡Felicitaciones! Ya comenzaste tu camino hacia AI/ML 🎉
```

______________________________________________________________________

## 📥 Step 1: Setup (3 minutes)

### Clone the repository

```bash
git clone https://github.com/matiasz8/training-ai.git
cd training-ai
```

### Create your virtual environment

```bash
# Python 3.8+ requerido
python --version  # Verifica que tengas Python 3.8+

# Create entorno virtual
python -m venv venv

# Activa el entorno
source venv/bin/activate        # Linux/Mac
# o
venv\scripts\activate           # Windows

# Verifica activation (you should ver "venv" en tu prompt)
which python  # Ought apuntar a tu venv
```

### Install base dependencies

```bash
# Instala solo lo necesario para empezar
pip install --upgrade pip
pip install numpy pandas matplotlib jupyter scikit-learn

# Verifica installation
python -c "import numpy; print(f'NumPy {numpy.__version__} OK')"
python -c "import pandas; print(f'Pandas {pandas.__version__} OK')"
```

**✅ Checkpoint:** If you see the "OK" messages, you are ready to continue!

______________________________________________________________________

## 📖 Step 2: First reading (5 minutes)

### Read the overview of Module 1

```bash
cd modules/01-programming-math-for-ml/
cat README.md  # o abre en tu editor
```

**🎯 In this Module you will learn:**

- Python scientist: NumPy, Pandas, Matplotlib
- Applied linear algebra: vectors, matrices, product point
- Descriptive statistics: mean, variance, correlation
- Basic probability: distributions, Bayes theorem

**⏰ Estimated time:** 2-3 weeks (10-15h/week)

### Explore the Module Structure

```bash
tree -L 1  # o ls para ver carpetas

# You should ver:
# ├── README.md
# ├── STATUS.md
# ├── theory/
# ├── examples/
# ├── practices/
# ├── mini-project/
# ├── evaluation/
# └── notes/
```

______________________________________________________________________

## 💻 Step 3: First Practice (25 minutes)

### Open your first Jupyter notebook

```bash
# Desde modules/01-programming-math-for-ml/
jupyter notebook

# Se will open tu browser con Jupyter
# Navega a practices/ y abre practice-01-python-y-numpy.md
```

### Or work with Python directly

```bash
# Create un archivo de practice
touch mi_primera_practica.py

# Copia el content de practices/practice-01-python-y-numpy.md
# Y execute section por section
```

### Your first code: Vectors with NumPy

```python
import numpy as np

# Create tu primer vector
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

# Operaciones vectoriales
suma = v1 + v2
print(f"Suma: {suma}")

producto_punto = np.dot(v1, v2)
print(f"Production point: {producto_punto}")

# Magnitud (norma)
magnitud = np.linalg.norm(v1)
print(f"Magnitud: {magnitud:.2f}")
```

**🎉 Expected output:**

```
Suma: [ 6  8 10 12]
Production point: 70
Magnitud: 5.48
```

**✅ Checkpoint:** If you ran the code and saw the Results, you are already programming ML!

______________________________________________________________________

## 🎯 Step 4: First Example complete (30 minutes)

### Run your first Example with real Data

```bash
cd examples/
cat 01-vectores-producto-punto.md  # Lee el example
```

### Copy the code and run it

```python
import numpy as np
import matplotlib.pyplot as plt

# Simula ratings de users a movies
usuario_gustos = np.array([5, 4, 1, 2])  # [Action, Drama, Comedia, Terror]
pelicula_1 = np.array([5, 3, 0, 1])      # Mad Max (mucha action)
pelicula_2 = np.array([2, 5, 4, 0])      # Drama familiar

# Calculate similarity (product point)
similitud_1 = np.dot(usuario_gustos, pelicula_1)
similitud_2 = np.dot(usuario_gustos, pelicula_2)

print(f"Recommendation para Movie 1: {similitud_1}")
print(f"Recommendation para Movie 2: {similitud_2}")

if similitud_1 > similitud_2:
    print("✅ Recomiendo Movie 1 (action)")
else:
    print("✅ Recomiendo Movie 2 (drama)")
```

**🎓 What you just did:**

- You implemented a **basic recommendation system**
- You used **linear algebra** (product point) for calculator similarity
- You applied **conceptual ML**: preference matching

**✅ Checkpoint:** You have just created your first ML Algorithm!

______________________________________________________________________

## 📊 Step 5: Track your progress

### Update your STATUS.md

```bash
# Abre STATUS.md en tu editor
vim STATUS.md  # o code STATUS.md, nano, etc.
```

### Mark your progress

```markdown
## Status actual: 🟡 En progress

### Progreso por section:
- ✅ Setup completed (2024-03-05)
- ✅ Primer example executedo (2024-03-05)
- 🟡 Theory: 0/3 completada
- 🟡 Examples: 1/2 executedos
- ⏳ Practices: 0/2 completadas

### Next pasos:
1. Leer theory/README.md (algebra lineal)
2. Complete practice-01-python-y-numpy.md
3. Execute example-02-estadistica-descriptiva.md

### Notes/Aprendizajes:
- El product point mide similarity entre vectors
- NumPy have operaciones vectoriales muy fast
```

______________________________________________________________________

## 🗺️ Step 6: Plan for the next days

### Week 1: Python Fundamentals for ML

```
Día 1-2: Theory + Examples de Python/NumPy
Día 3-4: Practices de Python/NumPy
Día 5-7: Theory + Examples de Algebra Lineal
```

### Week 2: Applied Math

```
Día 8-10:  Practices de Algebra Lineal
Día 11-12: Theory + Examples de Statistics
Día 13-14: Practices de Statistics
```

### Week 3: Consolidation

```
Día 15-18: Mini-project del module
Día 19-20: Evaluation y next module
```

**⏰ Recommended time:** 1.5-2 hours/day

______________________________________________________________________

## 🎯 Checklist of your first day

Mark what you complete:

- [ ] ✅ I cloned the repository
- [ ] ✅ I created and I activated virtual environment
- [ ] ✅ I installed NumPy, Pandas, Matplotlib
- [ ] ✅ I read README.md from Module 1
- [ ] ✅ I executed mi primer code (vectors NumPy)
- [ ] ✅ I completed the Example of recommendation
- [ ] ✅ I updated mi STATUS.md
- [ ] ✅ I understand flow: theory → examples → practices

**If you checked 6+ items:** Excellent start! 🎉

______________________________________________________________________

## 💡 Tips to keep up the peace

### 📅 **Consistency > Intensity**

```
Mejor:    1.5h/día durante 5 days = 7.5h/week ✅
Que:      7.5h en un solo Saturday 😫
```

### 🎯 **Daily Mini-Objectives**

```
✅ Hoy: Execute 1 example
✅ Mañana: Complete 1 practice
✅ Pasado: Leer 1 section de theory
```

### 📝 **Document your learnings**

Use the `notes/` folder to save:

- ❓ Doubts you have
- 💡Insights you discover
- 🐛 Bugs that you solve
- 🔗 Useful links that you find

### 🤝 **Compare your progress**

- Sube mini-projects a GitHub
- Compare on LinkedIn/Twitter
- Join ML communities (Discord, Reddit)

______________________________________________________________________

## 🚨 Troubleshooting common

### ❌ "ModuleNotFoundError: No module named 'numpy'"

```bash
# Verifica que el venv this activado
which python  # Debe apuntar a tu venv

# Reinstala si es necesario
pip install numpy pandas matplotlib
```

### ❌ "Permission denied" when creating venv

```bash
# Make sure de tener permisos
sudo apt install python3-venv  # Ubuntu/Debian
# o
brew install python  # Mac
```

### ❌ Jupyter does not open

```bash
# Instala jupyter si no lo hiciste
pip install jupyter

# Inicia en el puerto specific
jupyter notebook --port 8888
```

### ❌ "Python version too old"

```bash
# Verifica version
python --version

# Necesitas Python 3.8+
# Instala from python.org o usa pyenv
```

______________________________________________________________________

## 📚 Additional resources to get started

### 🎥 **Videos recommended** (optional)

- [NumPy in 10 minutes](https://www.youtube.com/results?search_query=numpy+tutorial) - Concepts basic
- [Matplotlib crash course](https://www.youtube.com/results?search_query=matplotlib+tutorial) - Visualization

### 📖 **Documentation official**

- [NumPy docs](https://numpy.org/doc/) - Complete reference
- [Pandas docs](https://pandas.pydata.org/docs/) - For data wrangling

### 🤝 **Communities**

- Reddit: r/learnmachinelearning, r/MachineLearning
- Discord: Fast.ai, ML Collective
- Stack Overflow: "python" and "numpy" tag

______________________________________________________________________

## 🎉 Congratulations

If you got this far, now:

- ✅ You configured your development environment
- ✅ You ran your first ML code
- ✅ You understand the Structure of the program
- ✅ You have a clear plan to follow

**Next immediate step:**

```bash
cd modules/01-programming-math-for-ml/theory/
cat README.md  # Lee la theory complete
```

______________________________________________________________________

## 🔗 Links useful

- [← Return to main README](../README.md)
- [📖 See all modules](SUMMARY-MODULES.md)
- [🗺️ Ver roadmap complete](LEARNING-PATH.md)
- [💪 Consistency Tips](STUDY-RHYTHM.md)

______________________________________________________________________

<div align="center">

### 🚀 The journey of 1000 miles begins with a single step

**You've already taken the first step. Now move on.** 💪

[Continue with Theory →](../modules/01-programming-math-for-ml/theory/README.md)

</div>
