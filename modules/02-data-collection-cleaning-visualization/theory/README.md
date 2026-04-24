# Theory — Data Collection, Cleaning & Visualization

## Why this module matters

La calidad de los datos determina el límite superior de performance de cualquier modelo de ML. Sin datos limpios, estructurados y bien entendidos, incluso el mejor algoritmo fallará. Este módulo te da herramientas para transformar datos crudos en datasets confiables listos para entrenar modelos.

______________________________________________________________________

## 1. Tipos de datos

### Clasificación por estructura

- **Datos estructurados:** tablas SQL, CSVs, Excel (filas y columnas).
- **Datos semi-estructurados:** JSON, XML, logs.
- **Datos no estructurados:** texto libre, imágenes, audio, video.

### Implicancias para ML

- Modelos tradicionales (regresión, árboles) trabajan con datos tabulares estructurados.
- Deep learning maneja eficientemente datos no estructurados (imágenes, texto).

______________________________________________________________________

## 2. Fuentes de datos

### Estrategias de obtención

- **APIs:** datos de servicios web (Twitter API, GitHub API, OpenWeather).
- **Web scraping:** extraer datos de HTML usando BeautifulSoup o Scrapy.
- **Bases de datos:** SQL (PostgreSQL, MySQL), NoSQL (MongoDB).
- **Archivos:** CSV, JSON, Parquet, Excel.
- **Datasets públicos:** Kaggle, UCI ML Repository, Google Dataset Search.

### Consideraciones críticas

- Siempre verificar licencias y términos de uso de los datos.
- Evaluar frecuencia de actualización y confiabilidad de la fuente.
- Considerar volumen necesario para entrenar modelos robustos.

📹 **Videos recomendados:**

1. [Working with APIs in Python - Corey Schafer](https://www.youtube.com/watch?v=tb8gHvYlCFs) - 18 min
1. [Web Scraping with BeautifulSoup - freeCodeCamp](https://www.youtube.com/watch?v=XVv6mJpFOb0) - 2 horas
1. [Pandas for Data Analysis - Keith Galli](https://www.youtube.com/watch?v=vmEHCJofslg) - 1 hora

______________________________________________________________________

## 3. Calidad de datos y limpieza

### Problemas comunes en datos reales

- **Valores nulos:** datos faltantes (NaN, None, null, espacios vacíos).
- **Duplicados:** filas repetidas que inflan métricas artificialmente.
- **Outliers:** valores atípicos que pueden ser errores o casos extremos legítimos.
- **Formatos inconsistentes:** fechas en múltiples formatos, mayúsculas/minúsculas mezcladas.
- **Tipos de datos incorrectos:** números guardados como strings, categorías como números.

### Técnicas de limpieza por problema

**Para valores nulos:**

- Eliminar filas/columnas con muchos nulos (cuando el % es alto, >50%).
- Imputar con media, mediana, moda (según distribución y contexto).
- Forward fill / backward fill para series temporales.
- Usar modelos predictivos para imputar (KNN Imputer, regresión).

**Para duplicados:**

- Detectar con `.duplicated()` y eliminar con `.drop_duplicates()`.
- Decidir qué fila conservar (primera, última, o custom).

**Para outliers:**

- Método IQR (rango intercuartílico): valores fuera de \[Q1 - 1.5*IQR, Q3 + 1.5*IQR\].
- Z-score: valores con |z| > 3 son outliers potenciales.
- Visualización con boxplots para decisión informada.

**Para formatos:**

- Normalizar strings: `.str.lower()`, `.str.strip()`, remover caracteres especiales.
- Parsing de fechas: `pd.to_datetime()` con formato explícito.
- Conversión de tipos: `.astype()` con validación previa.

📹 **Videos recomendados:**

1. [Data Cleaning with Pandas - Corey Schafer](https://www.youtube.com/watch?v=eMOA1pPVUc4) - 20 min
1. [Handling Missing Data - StatQuest](https://www.youtube.com/watch?v=jb3CVnBYgQc) - 10 min
1. [Complete Data Cleaning Tutorial - Alex The Analyst](https://www.youtube.com/watch?v=bDhvCp3_lYw) - 30 min

📚 **Recursos escritos:**

- [Pandas Data Cleaning Guide](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [DataCamp - Data Cleaning with Python](https://www.datacamp.com/tutorial/data-cleaning-python)
- [Kaggle - Data Cleaning Course](https://www.kaggle.com/learn/data-cleaning)

______________________________________________________________________

## 4. Feature Engineering

### Conceptos clave

**Feature engineering** es crear nuevas features a partir de existentes para mejorar la capacidad predictiva del modelo.

### Transformaciones comunes

- **Logaritmos:** comprimir rangos grandes, normalizar distribuciones asimétricas.
- **Raíz cuadrada / potencias:** ajustar relaciones no lineales.
- **Binning:** convertir variables continuas en categorías (edad → rangos etarios).
- **Interacciones:** multiplicar features relacionadas (área = ancho × alto).

### Encoding de variables categóricas

- **One-Hot Encoding:** crear columnas binarias por cada categoría.
- **Label Encoding:** asignar números enteros a categorías (solo si hay orden).
- **Target Encoding:** reemplazar categoría por media del target en esa categoría.
- **Frequency Encoding:** reemplazar por frecuencia de aparición.

### Feature Scaling (normalización)

- **Min-Max Scaling:** escalar a rango \[0, 1\]: `(x - min) / (max - min)`.
- **Standardization (Z-score):** media 0, std 1: `(x - mean) / std`.
- **Cuándo usar cada uno:**
  - Min-Max: cuando necesitás rango acotado (redes neuronales).
  - Standardization: cuando la distribución importa (regresión, SVM).

### Ejemplos aplicados

- De una fecha extraer: día de la semana, mes, trimestre, es_fin_de_semana.
- De texto extraer: longitud, número de palabras, presencia de palabras clave.
- De transacciones: monto promedio últimos 30 días, desviación estándar, máximo.

📹 **Videos recomendados:**

1. [Feature Engineering Course - Applied AI](https://www.youtube.com/watch?v=2N7hCn40YdY) - 1 hora
1. [Feature Scaling Explained - Krish Naik](https://www.youtube.com/watch?v=mnKm3YP56PY) - 15 min
1. [Categorical Encoding - Data Science Dojo](https://www.youtube.com/watch?v=irHhDMbw3xo) - 20 min

📚 **Recursos escritos:**

- [Feature Engineering - Google ML Guide](https://developers.google.com/machine-learning/data-prep/transform/introduction)
- [Feature Engineering for Machine Learning - Towards Data Science](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

______________________________________________________________________

## 5. Exploratory Data Analysis (EDA)

### Objetivo del EDA

Entender la estructura, distribución y relaciones en tus datos **ANTES** de entrenar modelos. El EDA previene errores costosos y revela insights para feature engineering.

### Pasos típicos del EDA

**1. Resumen estadístico inicial:**

```python
df.info()          # tipos, nulos
df.describe()      # estadísticas numéricas
df.value_counts()  # frecuencias categóricas
```

**2. Análisis de distribuciones:**

- Histogramas para variables numéricas.
- KDE plots para densidad de probabilidad.
- Countplots para categóricas.

**3. Análisis de relaciones:**

- Scatter plots para relación entre dos variables numéricas.
- Box plots agrupados para comparar distribuciones por categoría.
- Matriz de correlación (heatmap) para detectar multicolinealidad.

**4. Detección de patrones:**

- Agrupaciones naturales (clusters).
- Tendencias temporales.
- Outliers y anomalías.

### Herramientas de visualización

**Matplotlib:** gráficos base altamente personalizables.
**Seaborn:** visualizaciones estadísticas elegantes con menos código.
**Plotly:** gráficos interactivos para exploración dinámica.

📹 **Videos recomendados:**

1. [EDA with Pandas - Corey Schafer](https://www.youtube.com/watch?v=Wb2Tp35dZ-I) - 35 min
1. [Data Visualization with Seaborn - Keith Galli](https://www.youtube.com/watch?v=6GUZXDef2U0) - 1 hora
1. [Complete EDA Tutorial - Ken Jee](https://www.youtube.com/watch?v=QWgg4w1SpJ8) - 40 min

📚 **Recursos escritos:**

- [From Data to Viz](https://www.data-to-viz.com/) - guía de elección de gráficos
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Python Graph Gallery](https://www.python-graph-gallery.com/)
- [Kaggle EDA Notebooks](https://www.kaggle.com/code?tags=13204) - ejemplos reales

______________________________________________________________________

## 6. Buenas prácticas

### Durante la preparación de datos

- **Documentar transformaciones:** registrar todos los pasos aplicados.
- **No contaminar test set:** separar datos ANTES de cualquier transformación.
- **Reproducibilidad:** fijar semillas aleatorias al dividir train/test.
- **Pipelines automatizados:** usar `sklearn.pipeline.Pipeline` para encadenar transformaciones.
- **Versionar datasets:** guardar snapshots de datos procesados.

### Verificación de calidad (checklist)

- ✅ Revisar % de nulos por columna.
- ✅ Validar rangos esperados de valores numéricos.
- ✅ Confirmar que no hay duplicados no intencionales.
- ✅ Verificar consistencia de formatos de fecha/hora.
- ✅ Detectar features con correlación perfecta (redundantes).
- ✅ Validar balance de clases en problemas de clasificación.

📚 **Recursos adicionales:**

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) - estructura de proyectos
- [Great Expectations](https://greatexpectations.io/) - validación automatizada de datos
- [Pandas Profiling](https://github.com/ydataai/ydata-profiling) - EDA automático

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente módulo, deberías poder:

- ✅ Cargar y explorar un dataset real de Kaggle usando Pandas.
- ✅ Detectar y manejar valores nulos con al menos 2 estrategias diferentes justificadas.
- ✅ Identificar outliers y decidir si eliminarlos, mantenerlos o tratarlos según contexto.
- ✅ Realizar EDA completo con visualizaciones que respondan preguntas concretas de negocio.
- ✅ Aplicar feature engineering básico (encoding, scaling, nuevas features derivadas).
- ✅ Crear un pipeline reproducible de preprocesamiento documentado step-by-step.
- ✅ Explicar por qué la calidad de datos impacta directamente en el performance del modelo.

Si respondiste "sí" a todas, estás listo para modelar sobre bases sólidas.
