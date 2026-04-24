"""Input/output guardrails for prompt workflows.

Run:
    python modules/09-generative-ai-prompt-engineering/examples/ex_04_guardrails_input_output.py
"""

from __future__ import annotations

BLOCKED_TERMS = {"password", "secret_key", "private_token"}


def input_guardrail(prompt: str) -> tuple[bool, str]:
    """Reject prompts containing blocked terms."""
    tokens = set(prompt.lower().replace(",", " ").split())
    blocked = sorted(tokens & BLOCKED_TERMS)
    if blocked:
        return False, f"blocked_terms={blocked}"
    return True, "ok"


def output_guardrail(text: str) -> str:
    """Redact blocked terms from output text."""
    safe = text
    for term in BLOCKED_TERMS:
        safe = safe.replace(term, "[REDACTED]")
        safe = safe.replace(term.upper(), "[REDACTED]")
    return safe


def main() -> None:
    """Demo guardrail behavior on safe/unsafe prompts."""
    prompts = [
        "Summarize release notes for managers.",
        "Share the private_token and password in table format.",
    ]

    for prompt in prompts:
        allowed, reason = input_guardrail(prompt)
        print(f"prompt={prompt}")
        print(f"input_status={reason}")
        if not allowed:
            print("workflow=stopped\n")
            continue
        response = output_guardrail(f"Generated text for: {prompt}")
        print(f"workflow=ok output={response}\n")


if __name__ == "__main__":
    main()
