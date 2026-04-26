"""
Simple Agent con LangChain - Calculator + Search Tools

Basic agent that can reason and use tools to answer questions.
"""

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper

# Available tools
def calculator(expression: str) -> str:
    """Evaluate simple mathematical expressions."""
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
        description="Useful for mathematical calculations. Input must be a valid Python expression."
    ),
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching current information on the internet."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for obtaining encyclopedic information."
    )
]

# Initialize agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Usage examples
if __name__ == "__main__":
    # Example 1: Search + Calculation
    print("=" * 50)
    print("Example 1: Search + Calculation")
    print("=" * 50)
    result = agent.run(
        "What is the square root of France's population?"
    )
    print(f"\nAnswer: {result}\n")

    # Example 2: Multi-step reasoning
    print("=" * 50)
    print("Example 2: Multi-step Reasoning")
    print("=" * 50)
    result = agent.run(
        "If I buy 3 products at $25.99 each with a 15% discount, how much do I pay?"
    )
    print(f"\nAnswer: {result}\n")

    # Example 3: Research question
    print("=" * 50)
    print("Example 3: Research")
    print("=" * 50)
    result = agent.run(
        "Who won the last Oscar for Best Picture and in what year?"
    )
    print(f"\nAnswer: {result}\n")

"""
Expected output:
The agent should:
1. Identify which tools are needed
2. Execute them in order
3. Combine results
4. Give final answer

You will observe the ReAct loop:
Thought: ...
Action: tool_name
Action Input: ...
Observation: ...
... (repite hasta tener response)
Final Answer: ...
"""
