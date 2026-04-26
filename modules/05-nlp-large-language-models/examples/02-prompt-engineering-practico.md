# Example 02 — Practical Prompt Engineering with LLMs

## Context

You will learn **prompt engineering** techniques to obtain better responses from Large Language Models (LLMs). You will explore strategies such as zero-shot, few-shot, chain-of-thought, and prompt structuring.

## Objective

Master prompting techniques for tasks of: Classification, information extraction, reasoning and structured generation.

**Note:** We will use conceptual Examples. In production, replace with APIs like OpenAI GPT, Anthropic Claude, or local Models via HuggingFace.

______________________________________________________________________

## 🚀 Step 1: Mock setup (not real API)

```python
# Simulation de responses de LLM para demonstration
# En production: use openai.ChatCompletion.create() o similar

class MockLLM:
    """
    Simulador de LLM para demonstration
    En production: reemplazar con API real (OpenAI, Claude, etc.)
    """
    def generate(self, prompt, temperature=0.7):
        """Simula response de LLM"""
        # Here would go la llamada real a la API
        return f"[SIMULATED RESPONSE to: {prompt[:50]}...]"

llm = MockLLM()

# En production (example con OpenAI):
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

## 📚 Technique 1: Zero-Shot Prompting

### What is it?

Give direct instructions without previous examples.

### Example: Classification of feelings

```python
# ❌ PROMPT MALO (vago)
bad_prompt = "Analiza este text: 'El product arrive roto y el servicio al client, clientele fue terrible.'"

# ✅ PROMPT BUENO (specific)
good_prompt = """
Clasifica el sentiment del next text como POSITIVO, NEGATIVO o NEUTRAL.

Texto: "El product arrive roto y el servicio al client, clientele fue terrible."

Sentiment:
"""

print("=== ZERO-SHOT PROMPTING ===\n")
print("Prompt malo:", bad_prompt)
print("\nPrompt bueno:", good_prompt)

# Answer expected del LLM:
# "NEGATIVO"
```

**Improvements applied:**

- ✅ Clear and specific instruction
- ✅ Defined response options (POSITIVE, NEGATIVE, NEUTRAL)
- ✅ Structured format
- ✅ Clear delimitation of the input ("Text: ...")

______________________________________________________________________

## 🎯 Technique 2: Few-Shot Prompting

### What is it?

Provide input-output examples before the actual task. The Model learns the pattern.

### Example: Information extraction

```python
few_shot_prompt = """
Extrae el nombre del product, price y opinion del next text.

Example 1:
Texto: "I bought el iPhone 14 por $999 y es incredible."
Result:
- Production: iPhone 14
- Precio: $999
- Opinion: Positiva

Example 2:
Texto: "La laptop Dell cost $1200 but tiene problems de battery."
Result:
- Production: Laptop Dell
- Precio: $1200
- Opinion: Negativa

Example 3:
Texto: "El mouse Logitech vale $25 y funciona bien."
Result:
- Production: Mouse Logitech
- Precio: $25
- Opinion: Positiva

Ahora tu turno:
Texto: "I bought los AirPods Pro por $249 y el sonido es decepcionante."
Result:
"""

print("=== FEW-SHOT PROMPTING ===\n")
print(few_shot_prompt)

# Answer expected:
# - Production: AirPods Pro
# - Precio: $249
# - Opinion: Negativa
```

**Advantages:**

- ✅ Does not require fine-tuning of the Model
- ✅ The LLM learns the desired format
- ✅ More precise than zero-shot for complex tasks

**Recommendations:**

- Use 2-5 Examples (more is not always better)
- Diverse examples (different cases)
- Maintain consistent format

______________________________________________________________________

## 🧠 Technique 3: Chain-of-Thought (CoT) Prompting

### What is it?

Ask the Model to "think out loud" step by step before responding. Improves reasoning in complex problems.

### Example: Math problem

```python
# ❌ SIN CHAIN-OF-THOUGHT
direct_prompt = """
Problem: Juan tiene 3 cajas con 4 manzanas each una. Regala 5 manzanas. ¿How many le quedan?
Answer:
"""

# ✅ CON CHAIN-OF-THOUGHT
cot_prompt = """
Resuelve el next problem paso a paso:

Problem: Juan tiene 3 cajas con 4 manzanas each una. Regala 5 manzanas. ¿How many le quedan?

Paso 1: Calculate el total de manzanas inicial
Paso 2: Restar las manzanas regaladas
Paso 3: Answer final

Solution:
"""

print("=== CHAIN-OF-THOUGHT PROMPTING ===\n")
print("Sin CoT:\n", direct_prompt)
print("\nCon CoT:\n", cot_prompt)

# Answer expected (con CoT):
# Paso 1: Juan tiene 3 cajas × 4 manzanas = 12 manzanas
# Paso 2: Regala 5 manzanas: 12 - 5 = 7 manzanas
# Paso 3: Le quedan 7 manzanas
```

### Example: Logical Reasoning

```python
cot_logic_prompt = """
Pregunta: Si todos los gatos son animales, y algunos animales son mascotas, ¿es cierto que algunos gatos son mascotas?

Razona paso a paso:
1. Identifica las premisas
2. Analiza la logic
3. Determina la conclusion

Razonamiento:
"""

print("\n=== CoT para razonamiento logical ===\n")
print(cot_logic_prompt)

# Answer expected:
# 1. Premisas:
#    - Todos los gatos son animales
#    - Algunos animales son mascotas
# 2. Analysis:
#    - No hay relationship directa entre "gatos" y "mascotas"
#    - Que algunos animales sean mascotas no implica que los gatos lo sean
# 3. Conclusion: NO se can concluir que algunos gatos son mascotas (falacia logic)
```

**When use CoT:**

- ✅ Mathematical problems
- ✅ Logical reasoning
- ✅ Multi-step tasks
- ✅ When you need explainability

______________________________________________________________________

## 🏗️ Technique 4: Structuring Roles and Context

### System Message + User Message

```python
structured_prompt = """
System: Eres un asistente experto en analysisof data financieros. Respond, Response, Responds, Responded, Responder de manera concisa y technique.

User: Analiza this trend: Las acciones de TechCorp subieron 15% en enero, bajaron 8% en febrero, y subieron 22% en marzo. ¿Es una buena investment?
"""

print("=== ROLE-BASED PROMPTING ===\n")
print(structured_prompt)

# Answer expected (más technique gracias al rol definido):
# "Analysis de retorno acumulado:
# - Enero: +15%
# - Febrero: -8% (about base de 115%) = neto ~105.8%
# - Marzo: +22% (about base de 105.8%) = neto ~129.1%
# Retorno acumulado: ~29.1% en trimestre Q1 (excelente)
# Sin embargo, volatilidad intramensual significativa.
# Recommendation: Buena points out alcista, but verificar fundamentals y diversificar risk."
```

**Benefits of structuring roles:**

- ✅ The Model adopts a specific "personality"
- ✅ Answers more aligned with required expertise
- ✅ Appropriate tone and technical level

______________________________________________________________________

## 📋 Technique 5: Output Formatting (Structured JSON)

### Because?

Facilitates automatic parsing of responses to integrate with Data pipelines.

### Example: Extract to JSON

```python
json_prompt = """
Extrae information del next text y give it back en format JSON.

Texto: "Maria Gonzalez, especialista en Machine Learning con 5 years de experiencia, trabaja en TechCorp como Senior Data Scientist. Su email es mgonzalez@techcorp.com."

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

# Answer expected:
# {
#   "nombre": "Maria Gonzalez",
#   "especialidad": "Machine Learning",
#   "experiencia_años": 5,
#   "empresa": "TechCorp",
#   "puesto": "Senior Data Scientist",
#   "email": "mgonzalez@techcorp.com"
# }
```

**Output validation:**

```python
import json

def validate_llm_json(response):
    """Validate que el LLM returned JSON valid"""
    try:
        data = json.loads(response)
        required_keys = ["nombre", "especialidad", "experiencia_años", "empresa", "puesto", "email"]

        for key in required_keys:
            if key not in data:
                return False, f"Falta clave: {key}"

        return True, data
    except json.JSONDecodeError as e:
        return False, f"JSON invalid: {e}"

# Example de usage
simulated_response = '{"nombre": "Maria Gonzalez", "especialidad": "Machine Learning", ...}'
is_valid, result = validate_llm_json(simulated_response)
print(f"Valid: {is_valid}")
```

______________________________________________________________________

## 🛡️ Technique 6: Constraint Enforcement (Restrictions)

### Force clear boundaries

```python
constrained_prompt = """
Resume el next article en EXACTAMENTE 3 puntos clave. No uses más de 50 words en total.

Article: "La inteligencia artificial this transformando la industria financiera. Los bancos utilizan ML para detect fraudes, los robo-advisors gestionan portafolios automatically, y el analysis prediction improvement las decisions de investment. Sin embargo, persistent challenges en regulation, transparencia y sesgo algorithmic."

Summary (3 puntos, máx 50 words):
"""

print("=== CONSTRAINT ENFORCEMENT ===\n")
print(constrained_prompt)

# Answer expected:
# 1. ML detecta fraudes en banca
# 2. Robo-advisors automatizan management de portafolios
# 3. Challenges: regulation y sesgo algorithmic
```

**Useful restrictions:**

- **Length:** "maximum 100 words", "exactly 5 items"
- **Format:** "bullets only", "markdown table"
- **Tone:** "simple language", "technical and formal"
- **Language:** "answers in Spanish"
- **Prohibitions:** "do not use analogies", "avoid jargon"

______________________________________________________________________

## 💡 Technique 7: Self-Consistency

### What is it?

Generate multiple independent responses and choose the most consistent one (voting).

### Conceptual implementation

```python
def self_consistency(prompt, num_samples=5):
    """
    Generate multiple responses y votar por la más common
    """
    responses = []
    for i in range(num_samples):
        # En production: modificar temperature para variabilidad
        response = llm.generate(prompt, temperature=0.8)
        responses.append(response)

    # Votar (contar responses más frecuentes)
    from collections import Counter
    vote_result = Counter(responses).most_common(1)[0]
    return vote_result[0], vote_result[1] / num_samples  # response, confianza

# Example de usage
math_prompt = """
Problem: Si un tren sale de Madrid a las 10:00 AM a 120 km/h y otro sale de Barcelona (620 km de distancia) a las 10:30 AM a 100 km/h hacia Madrid, ¿a qué hora se encuentran?

Respond, Response, Responds, Responded, Responder solo la hora (format HH:MM).
"""

# answer, confidence = self_consistency(math_prompt, num_samples=5)
# print(f"Answer: {answer} (confianza: {confidence:.0%})")
```

**When use:**

- Complex reasoning problems
- When you need high reliability
- Tasks with unambiguous response

______________________________________________________________________

## 🎨 Technique 8: Multi-turn Conversation (Iterative Refinement)

### Example: Refining code

```python
# Turn 1: Generate borrador
prompt_1 = """
Genera una function Python para calculator el factorial de un number.
"""
# Answer (simulada):
# def factorial(n):
#     result = 1
#     for i in range(1, n+1):
#         result *= i
#     return result

# Turn 2: Pedir improvement
prompt_2 = """
Improvement la function anterior:
1. Add validation de input
2. Wear recursion en lugar de loop
3. Add docstrings
"""
# Answer mejorada (simulada):
# def factorial(n):
#     \"\"\"Calculate factorial de n recursivamente.\"\"\"
#     if not isinstance(n, int) or n < 0:
#         raise ValueError("n must ser entero no negativo")
#     return 1 if n == 0 else n * factorial(n-1)

# Turn 3: Optimizar
prompt_3 = """
Ahora optimiza using lru_cache para memoization.
"""
# Answer final (simulada):
# from functools import lru_cache
#
# @lru_cache(maxsize=None)
# def factorial(n):
#     \"\"\"Calculate factorial con memoization.\"\"\"
#     if not isinstance(n, int) or n < 0:
#         raise ValueError("n must ser entero no negativo")
#     return 1 if n == 0 else n * factorial(n-1)
```

**Advantages:**

- Incremental refinement
- The LLM remembers previous context
- Ideal for complex iterative tasks

______________________________________________________________________

## 📊 Comparison of techniques

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
        'Classification, extraction',
        'Razonamiento mathematical',
        'Answer personalizada',
        'Integration pipelines',
        'Limitar outputs',
        'Problems critics',
        'Refinamiento code'
    ]
})

print(comparison.to_string(index=False))
```

**Output:**

```
          Technique Complexity  Cost (API calls)  Accuracy                Use Case
          Zero-Shot        Low                 1    Medium          Tareas simples
           Few-Shot     Medium                 1      High  Classification, extraction
  Chain-of-Thought     Medium                 1      High   Razonamiento mathematical
         Role-Based        Low                 1    Medium    Answer personalizada
        JSON Output        Low                 1    Medium     Integration pipelines
        Constraints        Low                 1    Medium           Limitar outputs
  Self-Consistency       High                 5 Very High        Problems critics
         Multi-Turn       High                 3      High      Refinamiento code
```

______________________________________________________________________

## 📝Executive summary

### ✅ Best Practices for effective prompts

1. **Be specific:**

   - ❌ "Analyze this text"
   - ✅ "Classify the Sentiment as POSITIVE, NEGATIVE or NEUTRAL"

1. **Use clear delimiters:**

   ```
   Input: """text here"""
   Output:
   ```

1. **Specify output format:**

   - "Respond, Response, Responds, Responded, Reply in JSON"
- "Numbered list of 5 items"
   - "Tabla markdown"

1. **Dar context/rol:**

   - "You are an expert in..."
- "Act like a teacher of..."

1. **Ask for step-by-step reasoning:**

- For mathematics, logic, complex problems

1. **Use Examples (few-shot):**

   - 2-5 Representative Examples
   - Consistent format

1. **Iterate and refine:**

   - Try multiple formulations
   - Adjust temperature (0 = deterministic, 1 = creative)

______________________________________________________________________

## 🎓 Lessons learned

### ✅ Important API parameters

**Temperature (0-1):**

- `0.0`: Deterministic, consistent responses (for Classification, extraction)
- `0.7`: Balance (general purpose)
- `1.0`: Creative, varied (for idea generation, storytelling)

**Max tokens:**

- Limit response length
- Avoid cut off responses: calculator tokens required (1 word ≈ 1.3 tokens)

**Top-p (nucleus sampling):**

- Alternative to temperature
- `0.9` = considers only top 90% probability of tokens

**Frequency/Presence penalty:**

- Penalizes repetition of tokens
- Useful for long creative texts

### ✅ Errors common

- ❌ **Vague Prompts:** "Analyze this" → the LLM doesn't know what to do
- ❌ **Do not specify format:** Inconsistent output
- ❌ **Very few Examples (few-shot):** 1 Example is not enough
- ❌ **Too much context:** Exceed token limit (4k-128k according to Model)
- ❌ **No validate outputs:** Malformed JSON, responses out of specification

### 💡 Prompt debugging

1. **Prompt not working:**

   - Check specificity
   - Add Examples
   - Use CoT for complex problems

1. **Inconsistent output:**

   - Lower temperature (towards 0)
- Add more restrictions
   - Use self-consistency

1. **Answers cut:**

   - Increase max_tokens
   - Simplify prompt

1. **Too expensive:**

   - Reduce num_samples (self-consistency)
- Use smaller Models for simple tasks
   - Cache responses common

______________________________________________________________________

## 🔧 Complete prompt template

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

# Example de usage
filled_prompt = PROMPT_TEMPLATE.format(
    system_role="Eres un asistente de analysis de data",
    task_description="Extraer information de reviews de products",
    additional_context="Reviews de e-commerce Spanish",
    few_shot_examples="Example 1:...\nEjemplo 2:...",
    user_input="La laptop es buena but cara",
    specific_instructions="1. Identifica product\n2. Clasifica sentiment\n3. Extrae price si existe",
    output_format="JSON con claves: product, sentiment, price",
    constraints="- Sentiment must ser POSITIVO, NEGATIVO o NEUTRAL\n- Si no hay price, usa null"
)

print(filled_prompt)
```

### 📌 Prompting checklist

- ✅ Clear and specific instruction
- ✅ Defined output format
- ✅ Representative examples (2-5 for few-shot)
- ✅ Delimiters for input/output
- ✅ Explicit restrictions
- ✅ Context/role when relevant
- ✅ CoT for complex reasoning
- ✅ Output validation
- ✅ Appropriate temperature (0 = deterministic, 1 = creative)
- ✅ Cost control (max_tokens, num_samples)

______________________________________________________________________

## 📚 Additional Resources

- **Papers:**

  - "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (Wei et al., 2022)
  - "Few-Shot Learning with Language Models" (Brown et al., 2020)

- **Tools:**

  - [OpenAI Playground](https://platform.openai.com/playground): Experiment with prompts
  - [LangChain](https://langchain.com): Framework for applications with LLMs
  - [PromptBase](https://promptbase.com): Prompt Marketplace

- **Practice:**

  - [Learn Prompting](https://learnprompting.org): Cursor interaction
  - [Anthropic Prompt Library](https://docs.anthropic.com/claude/prompt-library): Examples by Claude
