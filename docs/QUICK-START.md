# 🚀 Quick Start Guide — ¡Empieza en 10 minutos

> **Tu primer día en el programa AI/ML: Setup, primer módulo, y primeros pasos**

______________________________________________________________________

## ⏱️ Timeline de tu primer día

```
┌──────────────────────────────────────────────────┐
│  Minuto 0-3: Setup del entorno                   │
├──────────────────────────────────────────────────┤
│  Minuto 3-8: Lee overview del Módulo 1           │
├──────────────────────────────────────────────────┤
│  Minuto 8-30: Primera práctica con NumPy         │
├──────────────────────────────────────────────────┤
│  Minuto 30-60: Primer ejemplo ejecutado          │
└──────────────────────────────────────────────────┘

¡Felicitaciones! Ya comenzaste tu camino hacia AI/ML 🎉
```

______________________________________________________________________

## 📥 Step 1: Setup (3 minutos)

### Clone el repositorio

```bash
git clone https://github.com/matiasz8/training-ai.git
cd training-ai
```

### Crea tu entorno virtual

```bash
# Python 3.8+ requerido
python --version  # Verifica que tengas Python 3.8+

# Crea entorno virtual
python -m venv venv

# Activa el entorno
source venv/bin/activate        # Linux/Mac
# o
venv\scripts\activate           # Windows

# Verifica activación (deberías ver "venv" en tu prompt)
which python  # Debería apuntar a tu venv
```

### Instala dependencias base

```bash
# Instala solo lo necesario para empezar
pip install --upgrade pip
pip install numpy pandas matplotlib jupyter scikit-learn

# Verifica instalación
python -c "import numpy; print(f'NumPy {numpy.__version__} OK')"
python -c "import pandas; print(f'Pandas {pandas.__version__} OK')"
```

**✅ Checkpoint:** Si ves los mensajes "OK", estás listo para continuar!

______________________________________________________________________

## 📖 Step 2: Primera lectura (5 minutos)

### Lee el overview del Módulo 1

```bash
cd modules/01-programming-math-for-ml/
cat README.md  # o abre en tu editor
```

**🎯 En este módulo aprenderás:**

- Python científico: NumPy, Pandas, Matplotlib
- Álgebra lineal aplicada: vectores, matrices, producto punto
- Estadística descriptiva: media, varianza, correlación
- Probabilidad básica: distribuciones, teorema de Bayes

**⏰ Tiempo estimado:** 2-3 semanas (10-15h/semana)

### Explora la estructura del módulo

```bash
tree -L 1  # o ls para ver carpetas

# Deberías ver:
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

## 💻 Step 3: Primera práctica (25 minutos)

### Abre tu primer Jupyter notebook

```bash
# Desde modules/01-programming-math-for-ml/
jupyter notebook

# Se abrirá tu browser con Jupyter
# Navega a practices/ y abre practice-01-python-y-numpy.md
```

### O trabaja con Python directamente

```bash
# Crea un archivo de práctica
touch mi_primera_practica.py

# Copia el contenido de practices/practice-01-python-y-numpy.md
# Y ejecuta sección por sección
```

### Tu primer código: Vectores con NumPy

```python
import numpy as np

# Crea tu primer vector
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

# Operaciones vectoriales
suma = v1 + v2
print(f"Suma: {suma}")

producto_punto = np.dot(v1, v2)
print(f"Producto punto: {producto_punto}")

# Magnitud (norma)
magnitud = np.linalg.norm(v1)
print(f"Magnitud: {magnitud:.2f}")
```

**🎉 Output esperado:**

```
Suma: [ 6  8 10 12]
Producto punto: 70
Magnitud: 5.48
```

**✅ Checkpoint:** Si ejecutaste el código y viste los resultados, ¡ya estás programando ML!

______________________________________________________________________

## 🎯 Step 4: Primer ejemplo completo (30 minutos)

### Ejecuta tu primer ejemplo con datos reales

```bash
cd examples/
cat 01-vectores-producto-punto.md  # Lee el ejemplo
```

### Copia el código y ejecútalo

```python
import numpy as np
import matplotlib.pyplot as plt

# Simula ratings de usuarios a películas
usuario_gustos = np.array([5, 4, 1, 2])  # [Acción, Drama, Comedia, Terror]
pelicula_1 = np.array([5, 3, 0, 1])      # Mad Max (mucha acción)
pelicula_2 = np.array([2, 5, 4, 0])      # Drama familiar

# Calcula similitud (producto punto)
similitud_1 = np.dot(usuario_gustos, pelicula_1)
similitud_2 = np.dot(usuario_gustos, pelicula_2)

print(f"Recomendación para Película 1: {similitud_1}")
print(f"Recomendación para Película 2: {similitud_2}")

if similitud_1 > similitud_2:
    print("✅ Recomiendo Película 1 (acción)")
else:
    print("✅ Recomiendo Película 2 (drama)")
```

**🎓 Lo que acabas de hacer:**

- Implementaste un **sistema de recomendación básico**
- Usaste **álgebra lineal** (producto punto) para calcular similitud
- Aplicaste **ML conceptual**: matching de preferencias

**✅ Checkpoint:** ¡Acabas de crear tu primer algoritmo de ML!

______________________________________________________________________

## 📊 Step 5: Trackea tu progreso

### Actualiza tu STATUS.md

```bash
# Abre STATUS.md en tu editor
vim STATUS.md  # o code STATUS.md, nano, etc.
```

### Marca tu progreso

```markdown
## Estado actual: 🟡 En progreso

### Progreso por sección:
- ✅ Setup completado (2024-03-05)
- ✅ Primer ejemplo ejecutado (2024-03-05)
- 🟡 Theory: 0/3 completada
- 🟡 Examples: 1/2 ejecutados
- ⏳ Practices: 0/2 completadas

### Próximos pasos:
1. Leer theory/README.md (álgebra lineal)
2. Completar practice-01-python-y-numpy.md
3. Ejecutar example-02-estadistica-descriptiva.md

### Notas/Aprendizajes:
- El producto punto mide similitud entre vectores
- NumPy hace operaciones vectoriales muy rápido
```

______________________________________________________________________

## 🗺️ Step 6: Plan para los próximos días

### Semana 1: Fundamentos de Python para ML

```
Día 1-2: Theory + Examples de Python/NumPy
Día 3-4: Practices de Python/NumPy
Día 5-7: Theory + Examples de Álgebra Lineal
```

### Semana 2: Matemáticas aplicadas

```
Día 8-10:  Practices de Álgebra Lineal
Día 11-12: Theory + Examples de Estadística
Día 13-14: Practices de Estadística
```

### Semana 3: Consolidación

```
Día 15-18: Mini-proyecto del módulo
Día 19-20: Evaluación y siguiente módulo
```

**⏰ Tiempo recomendado:** 1.5-2 horas/día

______________________________________________________________________

## 🎯 Checklist de tu primer día

Marca lo que completaste:

- [ ] ✅ Cloné el repositorio
- [ ] ✅ Creé y activé entorno virtual
- [ ] ✅ Instalé NumPy, Pandas, Matplotlib
- [ ] ✅ Leí README.md del Módulo 1
- [ ] ✅ Ejecuté mi primer código (vectores NumPy)
- [ ] ✅ Completé el ejemplo de recomendación
- [ ] ✅ Actualicé mi STATUS.md
- [ ] ✅ Entiendo el flujo: theory → examples → practices

**Si marcaste 6+ items:** ¡Excelente comienzo! 🎉

______________________________________________________________________

## 💡 Tips para mantener el ritmo

### 📅 **Consistencia > Intensidad**

```
Mejor:    1.5h/día durante 5 días = 7.5h/semana ✅
Que:      7.5h en un solo sábado 😫
```

### 🎯 **Mini-objetivos diarios**

```
✅ Hoy: Ejecutar 1 ejemplo
✅ Mañana: Completar 1 práctica
✅ Pasado: Leer 1 sección de theory
```

### 📝 **Documenta tus aprendizajes**

Usa la carpeta `notes/` para guardar:

- ❓ Dudas que tengas
- 💡 Insights que descubras
- 🐛 Bugs que resuelvas
- 🔗 Links útiles que encuentres

### 🤝 **Comparte tu progreso**

- Sube mini-projects a GitHub
- Comparte en LinkedIn/Twitter
- Únete a comunidades de ML (Discord, Reddit)

______________________________________________________________________

## 🚨 Troubleshooting común

### ❌ "ModuleNotFoundError: No module named 'numpy'"

```bash
# Verifica que el venv esté activado
which python  # Debe apuntar a tu venv

# Reinstala si es necesario
pip install numpy pandas matplotlib
```

### ❌ "Permission denied" al crear venv

```bash
# Asegúrate de tener permisos
sudo apt install python3-venv  # Ubuntu/Debian
# o
brew install python  # Mac
```

### ❌ Jupyter no se abre

```bash
# Instala jupyter si no lo hiciste
pip install jupyter

# Inicia en el puerto específico
jupyter notebook --port 8888
```

### ❌ "Python version too old"

```bash
# Verifica versión
python --version

# Necesitas Python 3.8+
# Instala desde python.org o usa pyenv
```

______________________________________________________________________

## 📚 Recursos adicionales para empezar

### 🎥 **Videos recomendados** (opcional)

- [NumPy in 10 minutes](https://www.youtube.com/results?search_query=numpy+tutorial) - Conceptos básicos
- [Matplotlib crash course](https://www.youtube.com/results?search_query=matplotlib+tutorial) - Visualización

### 📖 **Documentación oficial**

- [NumPy docs](https://numpy.org/doc/) - Referencia completa
- [Pandas docs](https://pandas.pydata.org/docs/) - Para data wrangling

### 🤝 **Comunidades**

- Reddit: r/learnmachinelearning, r/MachineLearning
- Discord: Fast.ai, ML Collective
- Stack Overflow: Etiqueta "python" y "numpy"

______________________________________________________________________

## 🎉 ¡Felicitaciones

Si llegaste hasta acá, ya:

- ✅ Configuraste tu entorno de desarrollo
- ✅ Ejecutaste tu primer código de ML
- ✅ Entiendes la estructura del programa
- ✅ Tienes un plan claro para seguir

**Próximo paso inmediato:**

```bash
cd modules/01-programming-math-for-ml/theory/
cat README.md  # Lee la teoría completa
```

______________________________________________________________________

## 🔗 Links útiles

- [← Volver al README principal](../README.md)
- [📖 Ver todos los módulos](RESUMEN-MODULOS.md)
- [🗺️ Ver roadmap completo](LEARNING-PATH.md)
- [💪 Tips de consistencia](STUDY-RHYTHM.md)

______________________________________________________________________

<div align="center">

### 🚀 El viaje de 1000 millas comienza con un solo paso

**Ya diste el primer paso. Ahora sigue adelante.** 💪

[Continuar con Theory →](../modules/01-programming-math-for-ml/theory/README.md)

</div>
