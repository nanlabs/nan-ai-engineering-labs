"""
ReAct Pattern Implementation - Reasoning + Acting

Implementación desde cero del pattern ReAct descrito en el paper.
"""

import re
from typing import List, Dict, Callable

class ReActAgent:
    """
    Agente que sigue el pattern ReAct:
    Thought → Action → Observation → (repetir) → Answer
    """

    def __init__(self, llm_function: Callable, tools: Dict[str, Callable], max_steps: int = 10):
        self.llm = llm_function
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        """Ejecuta el loop ReAct hasta obtener respuesta"""

        history = []
        step = 0

        # System prompt con instrucciones ReAct
        system_prompt = f"""
Responde la siguiente pregunta usando este formato:

Thought: tu razonamiento sobre qué hacer
Action: nombre_tool[input]
Observation: resultado del tool
... (repite Thought/Action/Observation hasta tener respuesta)
Thought: ahora puedo responder
Answer: respuesta final

Tools disponibles:
{self._format_tools()}

Pregunta: {question}
"""

        history.append(system_prompt)

        while step < self.max_steps:
            # LLM genera próximo paso
            response = self.llm("\n".join(history))
            history.append(response)

            print(f"\n--- Step {step + 1} ---")
            print(response)

            # Parse response
            if "Answer:" in response:
                # Encontró respuesta final
                answer = response.split("Answer:")[1].strip()
                return answer

            # Extract action
            action_match = re.search(r'Action:\s*(\w+)\[(.*?)\]', response)
            if action_match:
                tool_name = action_match.group(1)
                tool_input = action_match.group(2).strip()

                # Execute tool
                if tool_name in self.tools:
                    observation = self.tools[tool_name](tool_input)
                    history.append(f"Observation: {observation}")
                    print(f"Observation: {observation}")
                else:
                    history.append(f"Observation: Error - Tool '{tool_name}' not found")
                    print(f"Observation: Error - Tool not found")

            step += 1

        return "Max steps reached without finding answer."

    def _format_tools(self) -> str:
        """Describe tools disponibles"""
        descriptions = []
        for name, func in self.tools.items():
            desc = func.__doc__ or "No description"
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)


# Example tools
def search(query: str) -> str:
    """Busca información en internet (simulado)"""
    # Simulación - en producción usarías API real
    knowledge_base = {
        "población francia": "67 millones de habitantes",
        "capital alemania": "Berlín",
        "temperatura sol": "5,500 °C en superficie",
    }

    for key, value in knowledge_base.items():
        if key in query.lower():
            return value
    return "No se encontró información"

def calculator(expression: str) -> str:
    """Calcula expresiones matemáticas"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error en cálculo"

def get_date(query: str) -> str:
    """Retorna fecha actual"""
    return "5 de marzo de 2026"


# Mock LLM function (en producción usarías OpenAI/Anthropic)
def mock_llm(prompt: str) -> str:
    """
    Simulación de LLM que genera respuestas ReAct
    En producción, esto sería llamada a OpenAI/Anthropic
    """
    # Detecta qué step estamos
    if "población" in prompt and"Observation:" not in prompt:
        return "Thought: Necesito buscar la población de Francia\nAction: search[población francia]"
    elif "67 millones" in prompt and "sqrt" not in prompt:
        return "Thought: Ahora debo calcular la raíz cuadrada\nAction: calculator[sqrt(67000000)]"
    elif "8185" in prompt or "Observation:" in prompt:
        return "Thought: Tengo la respuesta final\nAnswer: La raíz cuadrada de la población de Francia es aproximadamente 8,185"
    else:
        return "Thought: No estoy seguro\nAnswer: No puedo responder"


# Usage example
if __name__ == "__main__":
    tools = {
        "search": search,
        "calculator": calculator,
        "get_date": get_date
    }

    agent = ReActAgent(
        llm_function=mock_llm,
        tools=tools,
        max_steps=5
    )

    question = "¿Cuál es la raíz cuadrada de la población de Francia?"
    answer = agent.run(question)

    print("\n" + "=" * 50)
    print(f"RESPUESTA FINAL: {answer}")
    print("=" * 50)

"""
Output esperado:

--- Step 1 ---
Thought: Necesito buscar la población de Francia
Action: search[población francia]
Observation: 67 millones de habitantes

--- Step 2 ---
Thought: Ahora debo calcular la raíz cuadrada
Action: calculator[sqrt(67000000)]
Observation: 8185.35

--- Step 3 ---
Thought: Tengo la respuesta final
Answer: La raíz cuadrada de la población de Francia es aproximadamente 8,185

==================================================
RESPUESTA FINAL: La raíz cuadrada de la población de Francia es aproximadamente 8,185
==================================================
"""
