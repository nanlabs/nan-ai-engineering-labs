# AI Agents — Sistemas Autónomos Inteligentes

## 🎯 Objetivo

Explorar arquitecturas de agentes de IA que pueden razonar, planificar y ejecutar tareas de forma autónoma usando LLMs y herramientas externas.

## 💡 Qué aprenderás

- Arquitectura de agentes: perception → reasoning → action
- Frameworks: LangChain, LlamaIndex, AutoGPT patterns
- ReAct (Reasoning + Acting) pattern para agentes
- Tool calling y function execution
- Memory systems (short-term y long-term)
- Multi-agent systems y collaboration
- Agent evaluation y safety considerations

## 📂 Contenido

### Examples

- **ex_01_simple_agent_langchain.py**: Agente básico con LangChain que responde preguntas usando herramientas (calculator, search)
- **ex_02_react_agent.py**: Implementación del pattern ReAct (thought → action → observation loop)
- **ex_03_multi_agent_system.py**: Sistema con múltiples agentes especializados que colaboran

### Notes

- Arquitecturas de agentes comparadas
- Limitaciones y failure modes
- Safety considerations (hallucinations, infinite loops, cost control)

## 🧪 Experimento Rápido

1. **Setup básico**: Instala LangChain (`pip install langchain openai`)
1. **Agente simple**: Crea agente con herramientas (calculator + Wikipedia search)
1. **Test**: Pregunta "¿Cuánto es la raíz cuadrada de la población de Francia?"
1. **Observa**: El agente debe:
   - Buscar población de Francia (tool: search)
   - Calcular raíz cuadrada (tool: calculator)
   - Combinar resultados

## 🔑 Conceptos Clave

### ReAct Pattern

```
Thought: Necesito encontrar la población actual de Francia
Action: search("población de Francia 2024")
Observation: La población es ~67 millones

Thought: Ahora puedo calcular la raíz cuadrada
Action: calculator("sqrt(67000000)")
Observation: 8185.35

Thought: Tengo la respuesta final
Answer: La raíz cuadrada de la población de Francia es aproximadamente 8,185
```

### Agent Architecture

```
┌─────────────┐
│   LLM Core  │ ← Reasoning engine
└──────┬──────┘
       │
┌──────┴──────────────────────┐
│                             │
▼                             ▼
┌─────────┐           ┌───────────┐
│  Tools  │           │  Memory   │
│ - Search│           │ - History │
│ - Calc  │           │ - Context │
│ - Code  │           │ - Facts   │
└─────────┘           └───────────┘
```

## 📊 Comparación de Frameworks

| Framework      | Pros                         | Cons               | Use Case               |
| -------------- | ---------------------------- | ------------------ | ---------------------- |
| **LangChain**  | Ecosystem rico, muchos tools | Puede ser complejo | Prototyping rápido     |
| **LlamaIndex** | Excelente para RAG           | Menos tools        | Query sobre documentos |
| **AutoGPT**    | Muy autónomo                 | Menos control      | Research tasks         |
| **Custom**     | Control total                | Más trabajo        | Production systems     |

## 🚧 Limitaciones y Riesgos

**Hallucinations**: Agentes pueden inventar resultados de tools

- **Mitigación**: Validar outputs, usar structured outputs

**Infinite loops**: Agente entra en ciclo sin salida

- **Mitigación**: Max iterations limit, timeout, circuit breakers

**Cost explosion**: Muchas llamadas a LLM API

- **Mitigación**: Budget limits, caching, cheaper models para planning

**Security**: Agente ejecuta código malicioso

- **Mitigación**: Sandboxing, whitelist de herramientas, human-in-the-loop

## 📚 Recursos Curados

**Frameworks:**

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [BabyAGI](https://github.com/yoheinakajima/babyagi)

**Papers:**

- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Toolformer: LMs Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Generative Agents (Stanford)](https://arxiv.org/abs/2304.03442)

**Tutoriales:**

- [LangChain Agent Tutorial](https://python.langchain.com/docs/tutorials/agents/)
- [Building Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)

## ✅ Checklist de Aprendizaje

- [ ] Entender arquitectura básica de agentes
- [ ] Implementar ReAct pattern
- [ ] Crear agente con tool calling
- [ ] Implementar memory system (conversación persistente)
- [ ] Evaluar agente con benchmark task
- [ ] Identificar y mitigar failure modes
- [ ] Diseñar multi-agent system simple

## 🔬 Ejercicio Práctico

**Construye un Research Assistant Agent** que pueda:

1. Recibir pregunta de investigación
1. Descomponerla en sub-preguntas
1. Buscar información en web/papers
1. Sintetizar respuesta final con referencias
1. Guardar hallazgos en memoria para consultas futuras

## 🎯 Impacto Real

- **Customer Support**: Agentes autónomos que resuelven tickets
- **Data Analysis**: Agentes que exploran datasets y generan insights
- **Software Engineering**: Agentes que escriben, prueban y debugean código
- **Research**: Agentes que conducen literature reviews automatizadas

## 🚀 Próximos Pasos

Una vez domines agentes básicos:

- Explora **ai-observability** para monitorear agentes en producción
- Revisa **guardrails** para hacer agentes más seguros
- Mira **llm-evals** para evaluar performance de agentes sistemáticamente

## Module objective

Pendiente de completar este apartado.

## What you will achieve

Pendiente de completar este apartado.

## Internal structure

Pendiente de completar este apartado.

## Level path (L1-L4)

Pendiente de completar este apartado.

## Recommended plan (by progress, not by weeks)

Pendiente de completar este apartado.

## Module completion criteria

Pendiente de completar este apartado.
