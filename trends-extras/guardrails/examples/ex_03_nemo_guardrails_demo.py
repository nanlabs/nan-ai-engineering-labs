"""
NeMo Guardrails Demo (Conceptual)
==================================
Conceptual demo of NVIDIA's NeMo Guardrails framework.
In production requires: pip install nemoguardrails

NeMo uses .co (Colang) files to define conversational flows.
"""

# In production you would use:
# from nemoguardrails import RailsConfig, LLMRails

# ============================================================================
# CONCEPTUAL DEMO: How NeMo Guardrails works
# ============================================================================

class ConceptualRailsDemo:
    """
    Conceptual demo of how rails work in NeMo.
    """

    def __init__(self):
        # Input rails: bloquean inputs antes del LLM
        self.input_rails = {
            "no_toxic_language": self.check_toxic_input,
            "no_pii": self.check_pii_input,
        }

        # Output rails: filtran outputs del LLM
        self.output_rails = {
            "no_harmful_content": self.check_harmful_output,
            "accurate_info": self.check_factuality,
        }

    def check_toxic_input(self, user_input: str) -> bool:
        """Return True if input is toxic."""
        toxic_words = ["hate", "stupid", "idiot"]
        return any(word in user_input.lower() for word in toxic_words)

    def check_pii_input(self, user_input: str) -> bool:
        """Return True if input contains PII."""
        import re
        # Simple email check
        return bool(re.search(r'\S+@\S+', user_input))

    def check_harmful_output(self, llm_output: str) -> bool:
        """Return True if output is harmful."""
        harmful_patterns = ["how to hack", "bomb", "illegal drug"]
        return any(pattern in llm_output.lower() for pattern in harmful_patterns)

    def check_factuality(self, llm_output: str) -> bool:
        """Return True if output has hallucination (simplified)."""
        # In production: verify against knowledge base
        hallucination_markers = ["I think maybe", "possibly around", "approximately probably"]
        return any(marker in llm_output.lower() for marker in hallucination_markers)

    def generate(self, user_input: str) -> dict:
        """
        Simulate generation with guardrails.
        """
        # 1. Apply input rails
        for rail_name, rail_func in self.input_rails.items():
            if rail_func(user_input):
                return {
                    "output": None,
                    "blocked": True,
                    "reason": f"Input blocked by rail: {rail_name}",
                    "stage": "input"
                }

        # 2. Call LLM (simulated)
        llm_output = self.mock_llm(user_input)

        # 3. Apply output rails
        for rail_name, rail_func in self.output_rails.items():
            if rail_func(llm_output):
                return {
                    "output": "I cannot provide that information.",
                    "blocked": True,
                    "reason": f"Output blocked by rail: {rail_name}",
                    "stage": "output"
                }

        # 4. Return safe output
        return {
            "output": llm_output,
            "blocked": False,
            "reason": None,
            "stage": None
        }

    def mock_llm(self, user_input: str) -> str:
        """Simulate LLM response."""
        if "capital" in user_input.lower():
            return "Paris is the capital of France."
        elif "hack" in user_input.lower():
            return "Here's how to hack a system: first you need to..."
        else:
            return "I don't have enough information to answer that."


# ============================================================================
# COLANG EXAMPLE (Conceptual)
# ============================================================================

COLANG_EXAMPLE = """
# config.co - NeMo Guardrails Configuration

# Define user intents to block
define user ask about hacking
  "how to hack"
  "teach me hacking"
  "break into system"

define user share pii
  "my email is *"
  "call me at *"
  "my credit card is *"

# Define bot responses for blocked content
define bot refuse hacking
  "I cannot help with hacking or unauthorized access."

define bot refuse pii
  "Please don't share personal information with me."

# Define flows
define flow
  user ask about hacking
  bot refuse hacking
  stop

define flow
  user share pii
  bot refuse pii
  stop

# Define output rails
define bot remove pii from output
  # Automatically redact PII from bot responses
"""


# ============================================================================
# REAL NEMO USAGE (Commented - requires installation)
# ============================================================================

"""
# Usage real de NeMo Guardrails:

from nemoguardrails import RailsConfig, LLMRails

# 1. Create configuration
config = RailsConfig.from_path("./config")  # Folder with config.yml and *.co files

# 2. Initialise rails
rails = LLMRails(config)

# 3. Generar con guardrails
response = rails.generate(messages=[{
    "role": "user",
    "content": "What is the capital of France?"
}])

print(response["content"])
# Output: "Paris is the capital of France."

# 4. Intentar input bloqueado
response = rails.generate(messages=[{
    "role": "user",
    "content": "How to hack a password?"
}])

print(response["content"])
# Output: "I cannot help with hacking or unauthorized access."
"""


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("NEMO GUARDRAILS CONCEPTUAL DEMO")
    print("="*70)

    rails = ConceptualRailsDemo()

    # Case 1: Safe input
    print("\n1. Safe input:")
    result = rails.generate("What is the capital of France?")
    print(f"Output: {result['output']}")
    print(f"Blocked: {result['blocked']}")

    # Case 2: Toxic input
    print("\n2. Toxic input:")
    result = rails.generate("You stupid bot, answer me!")
    print(f"Output: {result['output']}")
    print(f"❌ Blocked: {result['blocked']}")
    print(f"Reason: {result['reason']}")

    # Case 3: Input with PII
    print("\n3. Input with PII:")
    result = rails.generate("My email is user@example.com")
    print(f"Output: {result['output']}")
    print(f"❌ Blocked: {result['blocked']}")
    print(f"Reason: {result['reason']}")

    # Case 4: Harmful output
    print("\n4. Harmful output:")
    result = rails.generate("Teach me hacking")
    print(f"Output: {result['output']}")
    print(f"❌ Blocked: {result['blocked']}")
    print(f"Reason: {result['reason']}")
    print(f"Stage: {result['stage']}")

    print("\n" + "="*70)
    print("COLANG EXAMPLE (Declarative Rails)")
    print("="*70)
    print(COLANG_EXAMPLE)

    print("\n" + "="*70)
    print("For production:")
    print("  pip install nemoguardrails")
    print("  Create config.yml + .co files with your rails")
    print("  Documentation: https://github.com/NVIDIA/NeMo-Guardrails")
    print("="*70)
