# Theory — Ethics, Bias & Explainability

## Why this module matters

Models de IA influyen en decisiones sobre créditos, contrataciones, justicia penal y salud. Decisiones injustas, sesgadas o opacas tienen consecuencias reales en vidas humanas. Este Module te equipa para construir sistemas responsables, justos y transparentes.

______________________________________________________________________

## 1. Ética en IA: Fundamentos

### ¿Por qué la ética importa?

- **Impacto social:** IA afecta oportunidades, derechos y bienestar.
- **Responsabilidad legal:** Regulaciones emergentes (EU AI Act, GDPR).
- **Reputación:** Sistemas injustos dañan confianza y marca.
- **Sostenibilidad:** Sistemas no éticos no perduran.

### Principios fundamentales

1. **Beneficencia:** IA debe beneficiar a las personas.
1. **No maleficencia:** No causar daño.
1. **Autonomía:** Respetar decisión humana.
1. **Justicia:** Distribución equitativa de beneficios y riesgos.
1. **Explicabilidad:** Decisiones deben ser comprensibles.

📹 **Videos recomendados:**

1. [AI Ethics Explained - MIT](https://www.youtube.com/watch?v=AaU6tI2pb3M) - 15 min
1. [Ethics of AI - Lex Fridman](https://www.youtube.com/watch?v=gmaONaP7TzI) - 30 min

📚 **Resources escritos:**

- [AI Ethics Guidelines - EU](https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai)

______________________________________________________________________

## 2. Types de sesgo en IA

### Sesgo de muestreo (Sampling Bias)

**Problem:** Dataset no representa población Objective.

**Example:** Entrenar detector de enfermedades con Images solo de hospitales de élite.

**Mitigación:**

- Muestreo estratificado.
- Verificar distribución demográfica.

### Sesgo histórico (Historical Bias)

**Problem:** Data reflejan discriminación pasada.

**Example:** Sistema de contratación entrenado con histórico donde 90% de ingenieros eran hombres.

**Mitigación:**

- Rebalancear Data.
- Intervenir en features problemáticas.

### Sesgo de medición (Measurement Bias)

**Problem:** Método de medición es sistemáticamente erróneo para ciertos grupos.

**Example:** Oxímetros menos precisos en piel oscura.

**Mitigación:**

- Validar instrumentos de medición por subgrupo.
- Usar múltiples fuentes de Data.

### Sesgo de etiqueta (Label Bias)

**Problem:** Etiquetas humanas contienen prejuicios.

**Example:** Moderadores etiquetan Content de minorías como "ofensivo" más frecuentemente.

**Mitigación:**

- Múltiples anotadores.
- Auditoría de acuerdos inter-anotador.

### Sesgo de agregación (Aggregation Bias)

**Problem:** Model único para grupos con Features diferentes.

**Example:** Mismo Model de Prediction de diabetes para todas las etnias (cuando factores de riesgo difieren).

**Mitigación:**

- Models específicos por subgrupo.
- Features contextuales.

📹 **Videos recomendados:**

1. [Bias in Machine Learning - Google](https://www.youtube.com/watch?v=59bMh59JQDo) - 8 min
1. [Understanding Fairness in ML - Microsoft](https://www.youtube.com/watch?v=jIU9JH9RsF0) - 15 min

______________________________________________________________________

## 3. Fairness (Equidad)

### Definiciones de fairness

#### Paridad demográfica (Demographic Parity)

**Definición:** Tasa de Predictions positivas debe ser igual entre grupos.

**Example:** % de aprobación de crédito debe ser igual para hombres y mujeres.

#### Igualdad de oportunidad (Equalized Odds)

**Definición:** Tasas de verdaderos positivos y falsos positivos deben ser iguales entre grupos.

**Example:** Model de contratación debe tener misma tasa de éxito para candidatos calificados de todos los grupos.

#### Calibración (Calibration)

**Definición:** Probabilidades predichas deben reflejar frecuencias reales por grupo.

### Trade-offs

**Teorema de imposibilidad:** No se pueden satisfacer todas las definiciones de fairness simultáneamente (excepto en casos triviales).

**Decisión:** Elegir definición de fairness según contexto y stakeholders.

📹 **Videos recomendados:**

1. [Fairness in ML - Moritz Hardt](https://www.youtube.com/watch?v=jIXIuYdnyyk) - 1 hora (fundamental)

📚 **Resources escritos:**

- [Fairness Definitions Explained](https://fairmlbook.org/)
- [AI Fairness 360 (IBM)](https://aif360.mybluemix.net/)

______________________________________________________________________

## 4. Explainability (Explicabilidad)

### ¿Por qué explicabilidad?

- **Confianza:** Usuarios confían en lo que entienden.
- **Debugging:** Identificar Errors del Model.
- **Cumplimiento:** GDPR requiere "derecho a explicación".
- **Justicia:** Decisiones que afectan vidas deben ser explicables.

### Explicaciones globales

**Objective:** Entender comportamiento general del Model.

**Métodos:**

- **Feature Importance:** Qué features son más importantes globalmente.
- **Partial Dependence Plots (PDP):** Cómo cambia Prediction al variar una feature.
- **Surrogate Models:** Entrenar Model simple (interpretable) que aproxime Model complejo.

### Explicaciones locales

**Objective:** Explicar UNA Prediction específica.

**Métodos:**

#### LIME (Local Interpretable Model-agnostic Explanations)

- Perturbar entrada.
- Entrenar Model lineal local alrededor de esa Prediction.
- Interpretabilidad del Model lineal.

#### SHAP (SHapley Additive exPlanations)

- Basado en Theory de juegos (valores Shapley).
- Asigna contribución de cada feature a la Prediction.
- Propiedades teóricas deseables (consistencia, aditividad).

**Usage:** SHAP es el estándar de facto en industria.

📹 **Videos recomendados:**

1. [Explainable AI - StatQuest](https://www.youtube.com/watch?v=C80SQe16Rao) - 20 min
1. [SHAP Explained - Ritvik Math](https://www.youtube.com/watch?v=VB9uV-x0gtg) - 15 min
1. [LIME Explained - Krish Naik](https://www.youtube.com/watch?v=d6j6bofhj2M) - 20 min

📚 **Resources escritos:**

- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Interpretable ML Book (Free)](https://christophm.github.io/interpretable-ml-book/)

______________________________________________________________________

## 5. Models interpretables vs explicaciones post-hoc

### Models inherentemente interpretables

- **Regresión lineal/logística:** Coeficientes directamente interpretables.
- **Decision Trees:** Lógica visual clara.
- **Reglas (if-then):** Transparentes por diseño.

**Ventaja:** Interpretabilidad garantizada.
**Desventaja:** Puede sacrificar performance.

### Explicaciones post-hoc (black-box)

Explicar Model complejo (Random Forest, XGBoost, Neural Networks) después de entrenar.

**Herramientas:** SHAP, LIME.

**Ventaja:** No sacrificar performance.
**Desventaja:** Explicación es aproximación, no verdad absoluta.

### Trade-off: accuracy vs Interpretability

No siempre es necesario sacrificar accuracy por interpretabilidad. Probar ambos enfoques.

______________________________________________________________________

## 6. Riesgo y gobernanza

### Evaluation de impacto

**Preguntas clave:**

- ¿Quién se ve afectado por las decisiones del Model?
- ¿Cuál es el costo de un error (FP y FN)?
- ¿Hay asimetría de poder (usuarios vulnerables)?

### Documentación: Model Cards

**Content:**

- Propósito del Model.
- Data de Training (distribución, limitaciones).
- Metrics de performance por subgrupo.
- Casos de Usage apropiados e inapropiados.
- Consideraciones éticas.

📚 **Resources escritos:**

- [Model Cards Paper (Google)](https://arxiv.org/abs/1810.03993)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

### Monitoreo continuo

- **Performance por subgrupo:** ¿El Model funciona peor para algún grupo?
- **Distributional shift:** ¿Cambió la distribución de Predictions?
- **Feedback humano:** ¿Usuarios reportan Problems?

### Proceso de apelación

Si el Model toma decisión adversa, debe haber proceso humano de revisión.

______________________________________________________________________

## 7. Casos de estudio reales

### COMPAS (Justicia Penal)

**Problem:** Algorithm de Prediction de reincidencia mostró sesgo racial (más falsos positivos para afroamericanos).

**Lesson:** Medir fairness por subgrupo desde el inicio.

### Amazon Recruiting Tool

**Problem:** Sistema de contratación penalizaba CVs con palabra "mujer".

**Lesson:** Sesgo histórico en Data se amplifica.

### Facial Recognition

**Problem:** Sistemas comerciales tenían tasas de error mucho más altas en mujeres de piel oscura.

**Lesson:** Evaluar performance en subgrupos diversos.

📹 **Videos recomendados:**

1. [AI Bias: Real Examples - Vox](https://www.youtube.com/watch?v=Ok5sKLXqynQ) - 10 min

______________________________________________________________________

## 8. Buenas Practices

- ✅ Incluir revisión ética desde fase de diseño (no al final).
- ✅ Auditar dataset: distribución, subgrupos, posibles sesgos.
- ✅ Medir fairness en Validation (no solo accuracy global).
- ✅ Documentar decisiones y trade-offs (Model Cards).
- ✅ Involucrar stakeholders afectados en diseño.
- ✅ Implementar explicabilidad (SHAP) desde desarrollo.
- ✅ Monitoreo continuo de fairness en producción.
- ✅ Establecer proceso de apelación humana.

📚 **Resources generales:**

- [Fairness and Machine Learning (Book - Free)](https://fairmlbook.org/)
- [Google Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
- [Microsoft Responsible AI Resources](https://www.microsoft.com/en-us/ai/responsible-ai-resources)

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente Module, deberías poder:

- ✅ Identificar Types de sesgo (muestreo, histórico, medición, etiqueta) en dataset.
- ✅ Explicar diferencias entre paridad demográfica e igualdad de oportunidad.
- ✅ Medir fairness para múltiples subgrupos demográficos.
- ✅ Elegir entre explicación global (SHAP global) vs local (SHAP por Prediction).
- ✅ Implementar SHAP para interpretar Model complejo.
- ✅ Documentar Model con Model Card.
- ✅ Proponer mitigación concreta para sesgo detectado.
- ✅ Diseñar proceso de monitoreo de fairness en producción.

Si respondiste "sí" a todas, estás listo para construir sistemas de IA responsables y éticos.
