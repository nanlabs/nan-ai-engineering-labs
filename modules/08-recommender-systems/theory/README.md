# Theory — Recommender Systems

## Why this module matters

Sistemas de recomendación impulsan miles de millones de dólares en comercio electrónico (Amazon, Netflix, Spotify). Dominar estos sistemas te permite crear experiencias personalizadas que aumentan engagement, ventas y satisfacción del usuario.

______________________________________________________________________

## 1. ¿Qué es un sistema de recomendación?

**Sistema de recomendación (RecSys):** Sugiere **ítems relevantes** (productos, películas, canciones, contenido) a cada usuario basándose en:

- Comportamiento histórico del usuario.
- Perfil del usuario.
- Comportamiento de usuarios similares.
- Características de los ítems.

### Aplicaciones

- E-commerce: "Clientes que compraron esto también compraron..." (Amazon).
- Streaming: "Recomendado para ti" (Netflix, Spotify).
- Redes sociales: feed personalizado (Facebook, Instagram).
- Búsqueda: resultados personalizados (Google).
- Dating apps: perfiles sugeridos (Tinder).

📹 **Videos recomendados:**

1. [Recommender Systems Explained - IBM](https://www.youtube.com/watch?v=giIXNoiqO_U) - 10 min
1. [How Netflix Recommends - Vox](https://www.youtube.com/watch?v=wQABXV0_M5Y) - 8 min

______________________________________________________________________

## 2. Enfoques principales

### Popularidad (Baseline)

Recomendar ítems más populares globalmente.

**Ventaja:** Simple, siempre funciona.
**Desventaja:** No personalizado, favorece ítems ya populares.

### Basado en contenido (Content-Based)

Recomendar ítems **similares** a los que al usuario le gustaron en el pasado.

**Cómo funciona:**

1. Representar ítems con features (género, autor, palabras clave).
1. Calcular similitud entre ítems.
1. Recomendar ítems similares a los consumidos por el usuario.

**Ventaja:**

- No necesita datos de otros usuarios.
- Explica por qué recomienda algo.

**Desventaja:**

- No descubre ítems fuera del perfil del usuario ("burbuja de filtro").
- Requiere features de calidad.

**Ejemplo:**
Usuario vio películas de terror → recomendar más terror.

### Filtrado colaborativo (Collaborative Filtering)

Recomendar basado en **patrones de comportamiento colectivo**.

**Tipos:**

#### User-Based Collaborative Filtering

- Encontrar usuarios similares al usuario objetivo.
- Recomendar ítems que esos usuarios similares consumieron.

**Lógica:** "Usuarios similares a ti también disfrutaron...".

#### Item-Based Collaborative Filtering

- Encontrar ítems similares a los que el usuario consumíó.
- Similitud basada en patrones de consumo colectivo.

**Lógica:** "Usuarios que les gustó X también les gustó Y".

**Ventaja:**

- Descubre patrones no obvios.
- No necesita features de ítems.

**Desventaja:**

- Cold start: no funciona con usuarios/ítems nuevos.
- Sparsity: matriz usuario-ítem muy vacía.

📹 **Videos recomendados:**

1. [Collaborative Filtering - StatQuest](https://www.youtube.com/watch?v=h9gpufJFF-0) - 15 min
1. [Content-Based vs Collaborative - Krish Naik](https://www.youtube.com/watch?v=Eeg1DEeWUjA) - 20 min

______________________________________________________________________

## 3. Matriz usuario-ítem

### Representación

```
           Ítem1  Ítem2  Ítem3  Ítem4
Usuario1     5      ?      3      ?
Usuario2     ?      4      ?      5
Usuario3     4      3      ?      ?
```

- **Filas:** usuarios.
- **Columnas:** ítems.
- **Valores:** rating (1-5), compra (0/1), clics, tiempo de visualización.
- **?** = faltante (usuario no interactúó con ítem).

### Objective

Predecir valores faltantes para recomendar ítems con mayor predicción.

______________________________________________________________________

## 4. Matrix Factorization

**Idea:** Descomponer matriz usuario-ítem en dos matrices de baja dimensión.

```
R (users × items) ≈ U (users × k) × V (k × items)
```

Donde `k` es número de factores latentes (features ocultas).

**Técnicas:**

- **SVD (Singular Value Decomposition):** descomposición matemática.
- **ALS (Alternating Least Squares):** optimización iterativa (usado en Apache Spark).
- **Deep Learning:** embeddings aprendidos con redes neuronales.

📹 **Videos recomendados:**

1. [Matrix Factorization - StatQuest](https://www.youtube.com/watch?v=ZspR5PZemcs) - 20 min
1. [SVD for Recommendations - 3Blue1Brown](https://www.youtube.com/watch?v=P5mlg91as1c) - 15 min

📚 **Recursos escritos:**

- [Matrix Factorization Techniques - Netflix Prize](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)

______________________________________________________________________

## 5. Sistemas híbridos

Combinar múltiples enfoques para aprovechar fortalezas de cada uno.

**Estrategias:**

- **Weighted:** combinar scores con pesos.
- **Switching:** usar un enfoque u otro según contexto.
- **Cascade:** filtrar con un enfoque, refinar con otro.
- **Feature Combination:** usar features de contenido + colaborativo en mismo modelo.

**Ejemplo:** Netflix combina filtrado colaborativo + contenido + popularidad + contexto temporal.

______________________________________________________________________

## 6. Problemas típicos

### Cold Start

**Problema:** Usuario o ítem nuevo sin historial.

**Soluciones:**

- **Usuario nuevo:** recomendar ítems populares, preguntar preferencias iniciales.
- **Ítem nuevo:** usar content-based, promoción inicial (exploration).

### Sparsity

**Problema:** Matriz usuario-ítem muy vacía (la mayoría de usuarios no interactúa con la mayoría de ítems).

**Soluciones:**

- Matrix factorization (reduce dimensionalidad).
- Usar datos implícitos (clics, vistas) además de ratings explícitos.

### Popularity Bias

**Problema:** Sistema siempre recomienda ítems populares, ignora ítems de nicho.

**Soluciones:**

- Diversificación intencional.
- Re-ranking para balancear popularidad y relevancia.

### Filter Bubble

**Problema:** Usuario solo ve contenido similar a lo que ya consumió (echo chamber).

**Soluciones:**

- Exploration vs Exploitation (bandits).
- Inyectar serendipia (recomendar ítems inesperados pero potencialmente interesantes).

📹 **Videos recomendados:**

1. [Cold Start Problem - Stanford CS 329S](https://www.youtube.com/watch?v=ZFCvvzvbVV4) - 20 min

______________________________________________________________________

## 7. Métricas de evaluación

### Offline metrics

#### Precision@K

¿De los K ítems recomendados, cuántos son relevantes?

```
Precision@K = (# relevantes en top-K) / K
```

#### Recall@K

¿De todos los ítems relevantes, cuántos capturé en top-K?

```
Recall@K = (# relevantes en top-K) / (total relevantes)
```

#### MAP@K (Mean Average Precision)

Promedio de precision a diferentes valores de K.

#### NDCG (Normalized Discounted Cumulative Gain)

Considera orden de recomendaciones: ítems relevantes arriba valen más.

#### Coverage

¿Qué % del catálogo es recomendado al menos una vez?

### Online metrics (A/B testing)

- Click-Through Rate (CTR).
- Conversion rate.
- Time spent.
- Revenue per user.

**Importante:** Métricas offline no siempre correlacionan con impacto de negocio. Validar en producción.

📹 **Videos recomendados:**

1. [Recommendation Metrics - Google ML](https://www.youtube.com/watch?v=eZp4oQLtlPM) - 15 min

📚 **Recursos escritos:**

- [Evaluation Metrics for Recommender Systems](https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093)

______________________________________________________________________

## 8. Deep Learning para recomendaciones

### Neural Collaborative Filtering (NCF)

Reemplazar matrix factorization lineal con redes neuronales.

**Ventaja:** Captura interacciones no lineales.

### Two-Tower Models

Dos redes neuronales:

- Torre 1: embeddings de usuario.
- Torre 2: embeddings de ítem.
- Similitud: producto punto o cosine similarity.

**Uso:** búsqueda eficiente en catálogos grandes.

### Sequence Models (RNN, Transformers)

Modelar historial de usuario como secuencia temporal.

**Ejemplo:** Predecir próximo producto basándose en secuencia de compras.

**Modelos:** GRU4Rec, BERT4Rec, SASRec.

📹 **Videos recomendados:**

1. [Deep Learning for RecSys - NVIDIA](https://www.youtube.com/watch?v=Kw5cU7lyYgs) - 40 min
1. [Neural Collaborative Filtering - Paper Explained](https://www.youtube.com/watch?v=rQTK3NmMPtE) - 20 min

📚 **Recursos escritos:**

- [Neural Collaborative Filtering Paper](https://arxiv.org/abs/1708.05031)
- [Transformers4Rec (NVIDIA)](https://github.com/NVIDIA-Merlin/Transformers4Rec)

______________________________________________________________________

## 9. Contexto y bandits

### Contextual Recommendations

Incorporar contexto:

- Hora del día, día de semana.
- Dispositivo (móvil vs desktop).
- Ubicación geográfica.
- Sesión actual (lo que hizo el usuario hace 5 minutos).

### Multi-Armed Bandits

Balance exploration (probar nuevas recomendaciones) vs exploitation (recomendar lo que sabemos que funciona).

**Algoritmos:**

- ε-greedy.
- UCB (Upper Confidence Bound).
- Thompson Sampling.
- Contextual Bandits.

📹 **Videos recomendados:**

1. [Multi-Armed Bandits - StatQuest](https://www.youtube.com/watch?v=e3L4VocZnnQ) - 12 min

______________________________________________________________________

## 10. Buenas prácticas

- ✅ Empezar con baseline de popularidad.
- ✅ Probar filtrado colaborativo item-based (simple y efectivo).
- ✅ Separar datos temporalmente (no aleatoriamente) si el tiempo importa.
- ✅ Medir diversidad y coverage, no solo precision.
- ✅ Auditar sesgos de exposición (no confundir click con preferencia).
- ✅ Implementar A/B testing en producción para validar impacto real.
- ✅ Considerar cold start desde el diseño inicial.
- ✅ Monitorear distribución de recomendaciones (evitar filter bubble).
- ✅ Documentar trade-offs (relevancia vs diversidad vs novedad).

📚 **Recursos generales:**

- [Recommender Systems Textbook (Free)](https://www.springer.com/gp/book/9783319296579)
- [Google RecSys Course](https://developers.google.com/machine-learning/recommendation)
- [Surprise Library (Python)](http://surpriselib.com/) - scikit-learn para RecSys

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente módulo, deberías poder:

- ✅ Explicar diferencia entre content-based y collaborative filtering.
- ✅ Describir qué es cold start y proponer soluciones.
- ✅ Construir matriz usuario-ítem y calcular similitudes.
- ✅ Implementar filtrado colaborativo item-based con cosine similarity.
- ✅ Justificar elección de métrica @K según objetivos de negocio.
- ✅ Identificar popularity bias y proponer estrategias de diversificación.
- ✅ Explicar trade-offs exploration vs exploitation.
- ✅ Diseñar experimento A/B para validar sistema de recomendación.

Si respondiste "sí" a todas, estás listo para sistemas de recomendación avanzados y en producción.
