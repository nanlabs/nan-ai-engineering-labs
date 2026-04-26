# AI Agents — Intelligent Autonomous Systems

## 🎯 Objective

Explore AI agent architectures that can reason, plan, and execute tasks autonomously using LLMs and external tools.

## 💡 What will you learn

- Agent architecture: perception → reasoning → action
- Frameworks: LangChain, LlamaIndex, AutoGPT patterns
- ReAct (Reasoning + Acting) pattern for agents
- Tool calling and function execution
- Memory systems (short-term and long-term)
- Multi-agent systems and collaboration
- Agent evaluation and safety considerations

## 📂 Content

### Examples

- **ex_01_simple_agent_langchain.py**: Basic agent with LangChain that responds, response, responds, respond, answer questions using tools (calculator, search)
- **ex_02_react_agent.py**: Implementation of the ReAct pattern (thought → action → observation loop)
- **ex_03_multi_agent_system.py**: System with multiple specialized agents that collaborate

### Notes

- Agent architectures compared
- Limitations and failure modes
- Safety considerations (hallucinations, infinite loops, cost control)

## 🧪 Quick Experiment

1. **Setup basic**: Instala LangChain (`pip install langchain openai`)
1. **Simple agent**: Create agent with tools (calculator + Wikipedia search)
1. **Test**: Question "What is the square root of the population of France?"
1. **Note**: The agent must:
- Search population of France (tool: search)
- Calculate square root (tool: calculator)
   - Combiner Results

## 🔑 Concepts Clave

### ReAct Pattern

```
Thought: Necesito encontrar la population actual de Francia
Action: search("population de Francia 2024")
Observation: La population es ~67 millones

Thought: Ahora puedo calculator la root cuadrada
Action: calculator("sqrt(67000000)")
Observation: 8185.35

Thought: Tengo la response final
Answer: La root cuadrada de la population de Francia es aproximadamente 8,185
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

## 📊 Framework Comparison

| Framework | Pros | Cons | Use Case |
| -------------- | ---------------------------- | ------------------ | ---------------------- |
| **LangChain** | Rich ecosystem, many tools | It can be complex | fast prototyping |
| **LlamaIndex** | Excellent for RAG | Less tools | Query about documents |
| **AutoGPT** | Very autonomous | Less control | Research tasks |
| **Custom** | Full control | More work | Production systems |

## 🚧 Limitations and Risks

**Hallucinations**: Agents can invent Tool Results

- **Mitigation**: Validate outputs, use structured outputs

**Infinite loops**: Agent enters a loop without output

- **Mitigation**: Max iterations limit, timeout, circuit breakers

**Cost explosion**: Muchas llamadas a LLM API

- **Mitigation**: Budget limits, caching, cheaper models for planning

**Security**: Agente execute code malicioso

- **Mitigation**: Sandboxing, tool whitelist, human-in-the-loop

## 📚 Resources Curados

**Frameworks:**

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [BabyAGI](https://github.com/yoheinakajima/babyagi)

**Papers:**

- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Toolformer: LMs Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Generative Agents (Stanford)](https://arxiv.org/abs/2304.03442)

**Tutorials:**

- [LangChain Agent Tutorial](https://python.langchain.com/docs/tutorials/agents/)
- [Building Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)

## ✅ Learning Checklist

- [ ] Understand basic agent architecture
- [ ] Implement ReAct pattern
- [ ] Create agent with tool calling
- [ ] Implement memory system (persistent conversation)
- [ ] Evaluate agent with benchmark task
- [ ] Identify and mitigate failure modes
- [ ] Design simple multi-agent system

## 🔬 Practical Exercise

**Build a Research Assistant Agent** that can:

1. Receive research question
1. Break it down into sub-questions
1. Search information in web/papers
1. Synthesize final response with References
1. Save findings in memory for future reference

## 🎯 Impacto Real

- **Customer Support**: Autonomous agents who resolve tickets
- **Data Analysis**: Agents that explore datasets and generate insights
- **Software Engineering**: Agents that write, test and debug code
- **Research**: Agents conducting automated literature reviews

## 🚀 Next Steps

Once you master basic agents:

- Explore **ai-observability** to monitor agents in production
- Review **guardrails** to make agents safer
- Watch **llm-evals** to systematically evaluate agent performance

## Module objective

Build a practical foundation for designing, implementing, and evaluating autonomous AI agents that can reason, call tools, and coordinate multi-step workflows.

## What you will achieve

- Implement single-agent and multi-agent patterns.
- Design reliable tool-calling loops with clear stopping rules.
- Evaluate task success, latency, and cost trade-offs.
- Document failure modes and mitigation strategies.

## Internal structure

- `README.md`: concepts, architecture patterns, and decision criteria.
- `examples/`: runnable agent implementations (LangChain, ReAct, multi-agent).
- `practices/`: guided exercises for decomposition, orchestration, and recovery.

## Level path (L1-L4)

- L1: Run a baseline single agent with one tool.
- L2: Add controlled variation with multiple tools and constraints.
- L3: Build a robust workflow with retries and error handling.
- L4: Design a small multi-agent system with measurable outcomes.

## Recommended plan (by progress, not by weeks)

Start with the baseline example, then progressively add memory, tool diversity, and evaluation checkpoints. Move to multi-agent orchestration only after a stable single-agent pipeline is reproducible.

## Module completion criteria

- You can run at least two agent patterns end-to-end.
- You can explain when to use single-agent vs multi-agent design.
- You provide a short evaluation report with metrics and failure analysis.
