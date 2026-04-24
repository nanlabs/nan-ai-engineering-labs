# Ejemplo 02 — Prompt Engineering Práctico con LLMs

## Contexto

Aprenderás técnicas de **prompt engineering** para obtener mejores respuestas de Large Language Models (LLMs). Explorarás estrategias como zero-shot, few-shot, chain-of-thought y estructuración de prompts.

## Objective

Dominar técnicas de prompting para tareas de: clasificación, extracción de información, razonamiento y generación estructurada.

**Nota:** Usaremos ejemplos conceptuales. En producción, reemplaza con APIs como OpenAI GPT, Anthropic Claude, o modelos locales vía HuggingFace.

______________________________________________________________________

## 🚀 Paso 1: Setup simulado (sin API real)

```python
# Simulación de respuestas de LLM para demostración
# En producción: usar openai.ChatCompletion.create() o similar

class MockLLM:
    """
    Simulador de LLM para demostración
    En producción: reemplazar con API real (OpenAI, Claude, etc.)
    """
    def generate(self, prompt, temperature=0.7):
        """Simula respuesta de LLM"""
        # Aquí iría la llamada real a la API
        return f"[SIMULATED RESPONSE to: {prompt[:50]}...]"

llm = MockLLM()

# En producción (ejemplo con OpenAI):
# import openai
# openai.api_key = 'tu-api-key'
#
# def call_llm(prompt):
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7
#     )
#     return response.choices[0].message.content
```

______________________________________________________________________

## 📚 Técnica 1: Zero-Shot Prompting

### ¿Qué es?

Dar instrucciones directas sin ejemplos previos.

### Ejemplo: Clasificación de sentimientos

```python
# ❌ PROMPT MALO (vago)
bad_prompt = "Analiza este texto: 'El producto llegó roto y el servicio al cliente fue terrible.'"

# ✅ PROMPT BUENO (específico)
good_prompt = """
Clasifica el sentimiento del siguiente texto como POSITIVO, NEGATIVO o NEUTRAL.

Texto: "El producto llegó roto y el servicio al cliente fue terrible."

Sentimiento:
"""

print("=== ZERO-SHOT PROMPTING ===\n")
print("Prompt malo:", bad_prompt)
print("\nPrompt bueno:", good_prompt)

# Respuesta esperada del LLM:
# "NEGATIVO"
```

**Mejoras aplicadas:**

- ✅ Instrucción clara y específica
- ✅ Opciones de respuesta definidas (POSITIVO, NEGATIVO, NEUTRAL)
- ✅ Formato estructurado
- ✅ Delimitación clara del input ("Texto: ...")

______________________________________________________________________

## 🎯 Técnica 2: Few-Shot Prompting

### ¿Qué es?

Proporcionar ejemplos de entrada-salida antes de la tarea real. El modelo aprende el patrón.

### Ejemplo: Extracción de información

```python
few_shot_prompt = """
Extrae el nombre del producto, precio y opinión del siguiente texto.

Ejemplo 1:
Texto: "Compré el iPhone 14 por $999 y es increíble."
Resultado:
- Producto: iPhone 14
- Precio: $999
- Opinión: Positiva

Ejemplo 2:
Texto: "La laptop Dell costó $1200 pero tiene problemas de batería."
Resultado:
- Producto: Laptop Dell
- Precio: $1200
- Opinión: Negativa

Ejemplo 3:
Texto: "El mouse Logitech vale $25 y funciona bien."
Resultado:
- Producto: Mouse Logitech
- Precio: $25
- Opinión: Positiva

Ahora tu turno:
Texto: "Compré los AirPods Pro por $249 y el sonido es decepcionante."
Resultado:
"""

print("=== FEW-SHOT PROMPTING ===\n")
print(few_shot_prompt)

# Respuesta esperada:
# - Producto: AirPods Pro
# - Precio: $249
# - Opinión: Negativa
```

**Ventajas:**

- ✅ No requiere fine-tuning del modelo
- ✅ El LLM aprende el formato deseado
- ✅ Más preciso que zero-shot para tareas complejas

**Recomendaciones:**

- Usar 2-5 ejemplos (más no siempre es mejor)
- Ejemplos diversos (diferentes cases)
- Mantener formato consistente

______________________________________________________________________

## 🧠 Técnica 3: Chain-of-Thought (CoT) Prompting

### ¿Qué es?

Pedir al modelo que "piense en voz alta" paso a paso antes de responder. Mejora razonamiento en problemas complejos.

### Ejemplo: Problema matemático

```python
# ❌ SIN CHAIN-OF-THOUGHT
direct_prompt = """
Problema: Juan tiene 3 cajas con 4 manzanas cada una. Regala 5 manzanas. ¿Cuántas le quedan?
Respuesta:
"""

# ✅ CON CHAIN-OF-THOUGHT
cot_prompt = """
Resuelve el siguiente problema paso a paso:

Problema: Juan tiene 3 cajas con 4 manzanas cada una. Regala 5 manzanas. ¿Cuántas le quedan?

Paso 1: Calcular el total de manzanas inicial
Paso 2: Restar las manzanas regaladas
Paso 3: Respuesta final

Solución:
"""

print("=== CHAIN-OF-THOUGHT PROMPTING ===\n")
print("Sin CoT:\n", direct_prompt)
print("\nCon CoT:\n", cot_prompt)

# Respuesta esperada (con CoT):
# Paso 1: Juan tiene 3 cajas × 4 manzanas = 12 manzanas
# Paso 2: Regala 5 manzanas: 12 - 5 = 7 manzanas
# Paso 3: Le quedan 7 manzanas
```

### Ejemplo: Razonamiento lógico

```python
cot_logic_prompt = """
Pregunta: Si todos los gatos son animales, y algunos animales son mascotas, ¿es cierto que algunos gatos son mascotas?

Razona paso a paso:
1. Identifica las premisas
2. Analiza la lógica
3. Determina la conclusión

Razonamiento:
"""

print("\n=== CoT para razonamiento lógico ===\n")
print(cot_logic_prompt)

# Respuesta esperada:
# 1. Premisas:
#    - Todos los gatos son animales
#    - Algunos animales son mascotas
# 2. Análisis:
#    - No hay relación directa entre "gatos" y "mascotas"
#    - Que algunos animales sean mascotas no implica que los gatos lo sean
# 3. Conclusión: NO se puede concluir que algunos gatos son mascotas (falacia lógica)
```

**Cuándo usar CoT:**

- ✅ Problemas matemáticos
- ✅ Razonamiento lógico
- ✅ Tareas multi-paso
- ✅ Cuando necesitas explicabilidad

______________________________________________________________________

## 🏗️ Técnica 4: Estructuración de Roles y Contexto

### System Message + User Message

```python
structured_prompt = """
System: Eres un asistente experto en análisisde datos financieros. Responde de manera concisa y técnica.

User: Analiza esta tendencia: Las acciones de TechCorp subieron 15% en enero, bajaron 8% en febrero, y subieron 22% en marzo. ¿Es una buena inversión?
"""

print("=== ROLE-BASED PROMPTING ===\n")
print(structured_prompt)

# Respuesta esperada (más técnica gracias al rol definido):
# "Análisis de retorno acumulado:
# - Enero: +15%
# - Febrero: -8% (sobre base de 115%) = neto ~105.8%
# - Marzo: +22% (sobre base de 105.8%) = neto ~129.1%
# Retorno acumulado: ~29.1% en trimestre Q1 (excelente)
# Sin embargo, volatilidad intramensual significativa.
# Recomendación: Buena señala alcista, pero verificar fundamentals y diversificar riesgo."
```

**Beneficios de estructurar roles:**

- ✅ El modelo adopta "personalidad" específica
- ✅ Respuestas más alineadas con expertise requerido
- ✅ Tono y nivel técnico apropiados

______________________________________________________________________

## 📋 Técnica 5: Output Formatting (JSON estructurado)

### ¿Por qué?

Facilita parseo automático de respuestas para integrar con pipelines de datos.

### Ejemplo: Extracción a JSON

```python
json_prompt = """
Extrae información del siguiente texto y devuélvela en formato JSON.

Texto: "María González, especialista en Machine Learning con 5 años de experiencia, trabaja en TechCorp como Senior Data Scientist. Su email es mgonzalez@techcorp.com."

Formato esperado:
{
  "nombre": "...",
  "especialidad": "...",
  "experiencia_años": ...,
  "empresa": "...",
  "puesto": "...",
  "email": "..."
}

JSON:
"""

print("=== OUTPUT FORMATTING (JSON) ===\n")
print(json_prompt)

# Respuesta esperada:
# {
#   "nombre": "María González",
#   "especialidad": "Machine Learning",
#   "experiencia_años": 5,
#   "empresa": "TechCorp",
#   "puesto": "Senior Data Scientist",
#   "email": "mgonzalez@techcorp.com"
# }
```

**Validación del output:**

```python
import json

def validate_llm_json(response):
    """Validar que el LLM devolvió JSON válido"""
    try:
        data = json.loads(response)
        required_keys = ["nombre", "especialidad", "experiencia_años", "empresa", "puesto", "email"]

        for key in required_keys:
            if key not in data:
                return False, f"Falta clave: {key}"

        return True, data
    except json.JSONDecodeError as e:
        return False, f"JSON inválido: {e}"

# Ejemplo de uso
simulated_response = '{"nombre": "María González", "especialidad": "Machine Learning", ...}'
is_valid, result = validate_llm_json(simulated_response)
print(f"Válido: {is_valid}")
```

______________________________________________________________________

## 🛡️ Técnica 6: Constraint Enforcement (Restricciones)

### Forzar límites claros

```python
constrained_prompt = """
Resume el siguiente artículo en EXACTAMENTE 3 puntos clave. No uses más de 50 palabras en total.

Artículo: "La inteligencia artificial está transformando la industria financiera. Los bancos utilizan ML para detectar fraudes, los robo-advisors gestionan portafolios automáticamente, y el análisis predictivo mejora las decisiones de inversión. Sin embargo, persisten desafíos en regulación, transparencia y sesgo algorítmico."

Resumen (3 puntos, máx 50 palabras):
"""

print("=== CONSTRAINT ENFORCEMENT ===\n")
print(constrained_prompt)

# Respuesta esperada:
# 1. ML detecta fraudes en banca
# 2. Robo-advisors automatizan gestión de portafolios
# 3. Desafíos: regulación y sesgo algorítmico
```

**Restricciones útiles:**

- **Longitud:** "máximo 100 palabras", "exactamente 5 ítems"
- **Formato:** "solo bullets", "tabla markdown"
- **Tono:** "lenguaje simple", "técnico y formal"
- **Idioma:** "responde en español"
- **Prohibiciones:** "no uses analogías", "evita jerga"

______________________________________________________________________

## 💡 Técnica 7: Self-Consistency (Auto-consistencia)

### ¿Qué es?

Generar múltiples respuestas independientes y elegir la más consistente (votación).

### Implementación conceptual

```python
def self_consistency(prompt, num_samples=5):
    """
    Generar múltiples respuestas y votar por la más común
    """
    responses = []
    for i in range(num_samples):
        # En producción: modificar temperature para variabilidad
        response = llm.generate(prompt, temperature=0.8)
        responses.append(response)

    # Votar (contar respuestas más frecuentes)
    from collections import Counter
    vote_result = Counter(responses).most_common(1)[0]
    return vote_result[0], vote_result[1] / num_samples  # respuesta, confianza

# Ejemplo de uso
math_prompt = """
Problema: Si un tren sale de Madrid a las 10:00 AM a 120 km/h y otro sale de Barcelona (620 km de distancia) a las 10:30 AM a 100 km/h hacia Madrid, ¿a qué hora se encuentran?

Responde solo la hora (formato HH:MM).
"""

# answer, confidence = self_consistency(math_prompt, num_samples=5)
# print(f"Respuesta: {answer} (confianza: {confidence:.0%})")
```

**Cuándo usar:**

- Problemas de razonamiento complejo
- Cuando necesitas alta confiabilidad
- Tareas con respuesta inequívoca

______________________________________________________________________

## 🎨 Técnica 8: Multi-turn Conversation (Refinamiento iterativo)

### Ejemplo: Refinando código

```python
# Turn 1: Generar borrador
prompt_1 = """
Genera una función Python para calcular el factorial de un número.
"""
# Respuesta (simulada):
# def factorial(n):
#     result = 1
#     for i in range(1, n+1):
#         result *= i
#     return result

# Turn 2: Pedir mejora
prompt_2 = """
Mejora la función anterior:
1. Agregar validación de input
2. Usar recursión en lugar de loop
3. Agregar docstrings
"""
# Respuesta mejorada (simulada):
# def factorial(n):
#     \"\"\"Calcula factorial de n recursivamente.\"\"\"
#     if not isinstance(n, int) or n < 0:
#         raise ValueError("n debe ser entero no negativo")
#     return 1 if n == 0 else n * factorial(n-1)

# Turn 3: Optimizar
prompt_3 = """
Ahora optimiza usando lru_cache para memoization.
"""
# Respuesta final (simulada):
# from functools import lru_cache
#
# @lru_cache(maxsize=None)
# def factorial(n):
#     \"\"\"Calcula factorial con memoization.\"\"\"
#     if not isinstance(n, int) or n < 0:
#         raise ValueError("n debe ser entero no negativo")
#     return 1 if n == 0 else n * factorial(n-1)
```

**Ventajas:**

- Refinamiento incremental
- El LLM recuerda contexto previo
- Ideal para tareas complejas iterativas

______________________________________________________________________

## 📊 Comparación de técnicas

```python
import pandas as pd

comparison = pd.DataFrame({
    'Technique': [
        'Zero-Shot',
        'Few-Shot',
        'Chain-of-Thought',
        'Role-Based',
        'JSON Output',
        'Constraints',
        'Self-Consistency',
        'Multi-Turn'
    ],
    'Complexity': ['Low', 'Medium', 'Medium', 'Low', 'Low', 'Low', 'High', 'High'],
    'Cost (API calls)': [1, 1, 1, 1, 1, 1, 5, 3],
    'Accuracy': ['Medium', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Very High', 'High'],
    'Use Case': [
        'Tareas simples',
        'Clasificación, extracción',
        'Razonamiento matemático',
        'Respuesta personalizada',
        'Integración pipelines',
        'Limitar outputs',
        'Problemas críticos',
        'Refinamiento código'
    ]
})

print(comparison.to_string(index=False))
```

**Salida:**

```
          Technique Complexity  Cost (API calls)  Accuracy                Use Case
          Zero-Shot        Low                 1    Medium          Tareas simples
           Few-Shot     Medium                 1      High  Clasificación, extracción
  Chain-of-Thought     Medium                 1      High   Razonamiento matemático
         Role-Based        Low                 1    Medium    Respuesta personalizada
        JSON Output        Low                 1    Medium     Integración pipelines
        Constraints        Low                 1    Medium           Limitar outputs
  Self-Consistency       High                 5 Very High        Problemas críticos
         Multi-Turn       High                 3      High      Refinamiento código
```

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Mejores prácticas para prompts efectivos

1. **Ser específico:**

   - ❌ "Analiza este texto"
   - ✅ "Clasifica el sentimiento como POSITIVO, NEGATIVO o NEUTRAL"

1. **Usar delimitadores claros:**

   ```
   Input: """texto aquí"""
   Output:
   ```

1. **Especificar formato de salida:**

   - "Responde en JSON"
   - "Lista numerada de 5 ítems"
   - "Tabla markdown"

1. **Dar contexto/rol:**

   - "Eres un experto en..."
   - "Actúa como un profesor de..."

1. **Pedir razonamiento paso a paso:**

   - Para matemáticas, lógica, problemas complejos

1. **Usar ejemplos (few-shot):**

   - 2-5 ejemplos representativos
   - Formato consistente

1. **Iterar y refinar:**

   - Probar múltiples formulaciones
   - Ajustar temperatura (0 = determinista, 1 = creativo)

______________________________________________________________________

## 🎓 Lecciones aprendidas

### ✅ Parámetros de API importantes

**Temperature (0-1):**

- `0.0`: Determinista, respuestas consistentes (para clasificación, extracción)
- `0.7`: Balance (general purpose)
- `1.0`: Creativo, variado (para generación de ideas, storytelling)

**Max tokens:**

- Limita longitud de respuesta
- Evita respuestas cortadas: calcular tokens necesarios (1 palabra ≈ 1.3 tokens)

**Top-p (nucleus sampling):**

- Alternativa a temperature
- `0.9` = considera solo top 90% de probabilidad de tokens

**Frequency/Presence penalty:**

- Penaliza repetición de tokens
- Útil para textos creativos largos

### ✅ Errores comunes

- ❌ **Prompts vagos:** "Analiza esto" → el LLM no sabe qué hacer
- ❌ **No especificar formato:** Output inconsistente
- ❌ **Muy pocos ejemplos (few-shot):** 1 ejemplo no es suficiente
- ❌ **Demasiado contexto:** Exceder límite de tokens (4k-128k según modelo)
- ❌ **No validar outputs:** JSON malformado, respuestas fuera de especificación

### 💡 Debugging de prompts

1. **Prompt no funciona:**

   - Revisar especificidad
   - Agregar ejemplos
   - Usar CoT para problemas complejos

1. **Output inconsistente:**

   - Bajar temperature (hacia 0)
   - Agregar más restricciones
   - Usar self-consistency

1. **Respuestas cortadas:**

   - Aumentar max_tokens
   - Simplificar prompt

1. **Demasiado costoso:**

   - Reducir num_samples (self-consistency)
   - Usar modelos más pequeños para tareas simples
   - Cachear respuestas comunes

______________________________________________________________________

## 🔧 Template de prompt completo

```python
PROMPT_TEMPLATE = """
System: {system_role}

Task: {task_description}

Context: {additional_context}

Examples:
{few_shot_examples}

Input:
{user_input}

Instructions:
{specific_instructions}

Output format:
{output_format}

Constraints:
{constraints}

Output:
"""

# Ejemplo de uso
filled_prompt = PROMPT_TEMPLATE.format(
    system_role="Eres un asistente de análisis de datos",
    task_description="Extraer información de reseñas de productos",
    additional_context="Reseñas de e-commerce español",
    few_shot_examples="Ejemplo 1:...\nEjemplo 2:...",
    user_input="La laptop es buena pero cara",
    specific_instructions="1. Identifica producto\n2. Clasifica sentimiento\n3. Extrae precio si existe",
    output_format="JSON con claves: producto, sentimiento, precio",
    constraints="- Sentimiento debe ser POSITIVO, NEGATIVO o NEUTRAL\n- Si no hay precio, usa null"
)

print(filled_prompt)
```

### 📌 Checklist de prompting

- ✅ Instrucción clara y específica
- ✅ Formato de salida definido
- ✅ Ejemplos representativos (2-5 para few-shot)
- ✅ Delimitadores para input/output
- ✅ Restricciones explícitas
- ✅ Contexto/rol cuando sea relevante
- ✅ CoT para razonamiento complejo
- ✅ Validación de outputs
- ✅ Temperature apropiada (0 = determinista, 1 = creativo)
- ✅ Control de costos (max_tokens, num_samples)

______________________________________________________________________

## 📚 Recursos adicionales

- **Papers:**

  - "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (Wei et al., 2022)
  - "Few-Shot Learning with Language Models" (Brown et al., 2020)

- **Herramientas:**

  - [OpenAI Playground](https://platform.openai.com/playground): Experimentar con prompts
  - [LangChain](https://langchain.com): Framework para aplicaciones con LLMs
  - [PromptBase](https://promptbase.com): Marketplace de prompts

- **Práctica:**

  - [Learn Prompting](https://learnprompting.org): Curso interactivo
  - [Anthropic Prompt Library](https://docs.anthropic.com/claude/prompt-library): Ejemplos de Claude
