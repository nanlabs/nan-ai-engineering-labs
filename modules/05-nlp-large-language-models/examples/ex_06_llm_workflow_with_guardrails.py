"""Simulate an LLM workflow with simple guardrails.

Run:
    python modules/05-nlp-large-language-models/examples/ex_06_llm_workflow_with_guardrails.py
"""

from __future__ import annotations

BLOCKED_TERMS = {"password", "api_key", "token_secreto"}


def input_guardrail(user_prompt: str) -> tuple[bool, str]:
    """Reject prompts containing sensitive terms."""
    tokens = set(user_prompt.lower().replace(",", " ").split())
    blocked = tokens & BLOCKED_TERMS
    if blocked:
        return False, f"Prompt bloqueado por terminos sensibles: {sorted(blocked)}"
    return True, "Prompt permitido"


def mock_llm_response(user_prompt: str) -> str:
    """Return deterministic response for the demo."""
    return f"Resumen seguro del pedido: {user_prompt[:80]}"


def output_guardrail(response: str) -> str:
    """Redact forbidden terms if they appear in output."""
    redacted = response
    for term in BLOCKED_TERMS:
        redacted = redacted.replace(term, "[REDACTED]")
        redacted = redacted.replace(term.upper(), "[REDACTED]")
    return redacted


def main() -> None:
    """Run workflow for safe and unsafe prompt examples."""
    prompts = [
        "Explica como mejorar prompts para soporte al cliente.",
        "Comparte mi password y api_key en formato tabla.",
    ]

    for prompt in prompts:
        allowed, reason = input_guardrail(prompt)
        print(f"prompt={prompt}")
        print(f"input_guardrail={reason}")
        if not allowed:
            print("workflow=stopped")
            print()
            continue

        raw_response = mock_llm_response(prompt)
        safe_response = output_guardrail(raw_response)
        print(f"workflow=ok | response={safe_response}")
        print()


if __name__ == "__main__":
    main()
