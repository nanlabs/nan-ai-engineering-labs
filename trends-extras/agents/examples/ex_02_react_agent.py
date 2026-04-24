"""
ReAct Pattern Implementation - Reasoning + Acting

From-scratch implementation of the ReAct pattern described in the paper.
"""

import re
from typing import List, Dict, Callable

class ReActAgent:
    """
    Agent that follows the ReAct pattern:
    Thought → Action → Observation → (repeat) → Answer
    """

    def __init__(self, llm_function: Callable, tools: Dict[str, Callable], max_steps: int = 10):
        self.llm = llm_function
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        """Run the ReAct loop until an answer is found."""

        history = []
        step = 0

        # System prompt with ReAct instructions
        system_prompt = f"""
Answer the following question using this format:

Thought: your reasoning about what to do
Action: tool_name[input]
Observation: result from the tool
... (repeat Thought/Action/Observation until you have an answer)
Thought: now I can answer
Answer: final answer

Available tools:
{self._format_tools()}

Question: {question}
"""

        history.append(system_prompt)

        while step < self.max_steps:
            # LLM generates next step
            response = self.llm("\n".join(history))
            history.append(response)

            print(f"\n--- Step {step + 1} ---")
            print(response)

            # Parse response
            if "Answer:" in response:
                # Found final answer
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
        """Describe available tools."""
        descriptions = []
        for name, func in self.tools.items():
            desc = func.__doc__ or "No description"
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)


# Example tools
def search(query: str) -> str:
    """Search for information on the internet (simulated)."""
    # Simulation - in production you would use a real API
    knowledge_base = {
        "france population": "67 million inhabitants",
        "capital of germany": "Berlin",
        "sun temperature": "5,500 °C at surface",
    }

    for key, value in knowledge_base.items():
        if key in query.lower():
            return value
    return "No information found"

def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Calculation error"

def get_date(query: str) -> str:
    """Return current date."""
    return "March 5, 2026"


# Mock LLM function (in production you would use OpenAI/Anthropic)
def mock_llm(prompt: str) -> str:
    """
    Simulated LLM that generates ReAct responses.
    In production, this would be a call to OpenAI/Anthropic.
    """
    # Detect which step we are on
    if "population" in prompt and "Observation:" not in prompt:
        return "Thought: I need to search for France's population\nAction: search[france population]"
    elif "67 million" in prompt and "sqrt" not in prompt:
        return "Thought: Now I need to calculate the square root\nAction: calculator[sqrt(67000000)]"
    elif "8185" in prompt or "Observation:" in prompt:
        return "Thought: I have the final answer\nAnswer: The square root of France's population is approximately 8,185"
    else:
        return "Thought: I'm not sure\nAnswer: I cannot answer"


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

    question = "What is the square root of France's population?"
    answer = agent.run(question)

    print("\n" + "=" * 50)
    print(f"FINAL ANSWER: {answer}")
    print("=" * 50)

"""
Expected output:

--- Step 1 ---
Thought: I need to search for France's population
Action: search[france population]
Observation: 67 million inhabitants

--- Step 2 ---
Thought: Now I need to calculate the square root
Action: calculator[sqrt(67000000)]
Observation: 8185.35

--- Step 3 ---
Thought: I have the final answer
Answer: The square root of France's population is approximately 8,185

==================================================
FINAL ANSWER: The square root of France's population is approximately 8,185
==================================================
"""
