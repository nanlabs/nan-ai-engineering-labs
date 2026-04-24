"""
Simple Agent con LangChain - Calculator + Search Tools

Agente básico que puede razonar y usar herramientas para responder preguntas.
"""

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper

# Tools disponibles
def calculator(expression: str) -> str:
    """Evalúa expresiones matemáticas simples"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Error: Invalid expression"

# Setup tools
search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Útil para cálculos matemáticos. Input debe ser expresión Python válida."
    ),
    Tool(
        name="Search",
        func=search.run,
        description="Útil para buscar información actual en internet."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Útil para obtener información enciclopédica."
    )
]

# Inicializar agente
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ejemplos de uso
if __name__ == "__main__":
    # Ejemplo 1: Búsqueda + cálculo
    print("=" * 50)
    print("Ejemplo 1: Búsqueda + Cálculo")
    print("=" * 50)
    result = agent.run(
        "¿Cuál es la raíz cuadrada de la población de Francia?"
    )
    print(f"\nRespuesta: {result}\n")

    # Ejemplo 2: Multi-step reasoning
    print("=" * 50)
    print("Ejemplo 2: Multi-step Reasoning")
    print("=" * 50)
    result = agent.run(
        "Si compro 3 productos a $25.99 cada uno y tengo un descuento del 15%, ¿cuánto pago?"
    )
    print(f"\nRespuesta: {result}\n")

    # Ejemplo 3: Research question
    print("=" * 50)
    print("Ejemplo 3: Research")
    print("=" * 50)
    result = agent.run(
        "¿Quién ganó el último Oscar a mejor película y en qué año?"
    )
    print(f"\nRespuesta: {result}\n")

"""
Output esperado:
El agente debe:
1. Identificar qué tools necesita
2. Ejecutarlos en orden
3. Combinar resultados
4. Dar respuesta final

Observarás el loop ReAct:
Thought: ...
Action: tool_name
Action Input: ...
Observation: ...
... (repite hasta tener respuesta)
Final Answer: ...
"""
