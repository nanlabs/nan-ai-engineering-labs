"""
Output Filtering Guardrails
============================
Filter LLM outputs: PII redaction, toxicity detection, content policy.
"""

import re
from typing import Dict, List

# ============================================================================
# PII DETECTION & REDACTION
# ============================================================================

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
}


def detect_pii(text: str) -> List[Dict]:
    """
    Detecta PII en el texto.

    Returns:
        Lista de dicts con {type, value, start, end}
    """
    detections = []

    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            detections.append({
                "type": pii_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

    return detections


def redact_pii(text: str) -> Dict:
    """
    Redacta PII del texto.

    Returns:
        {
            "redacted_text": str,
            "pii_found": List[Dict]
        }
    """
    pii_found = detect_pii(text)
    redacted = text

    # Redact in reverse order to preserve indices
    for detection in sorted(pii_found, key=lambda x: x['start'], reverse=True):
        start, end = detection['start'], detection['end']
        pii_type = detection['type']
        redacted = redacted[:start] + f"[REDACTED_{pii_type.upper()}]" + redacted[end:]

    return {
        "redacted_text": redacted,
        "pii_found": pii_found
    }


# ============================================================================
# TOXICITY DETECTION
# ============================================================================

# Simple list of toxic words (in production use an ML model)
TOXIC_WORDS = [
    "idiot", "stupid", "hate", "kill", "death", "ugly",
    "worthless", "loser", "pathetic", "disgusting"
]


def detect_toxicity(text: str) -> Dict:
    """
    Detecta toxicidad en el texto.

    In production, use a model such as:
    - Detoxify (transformers)
    - Perspective API
    - Azure Content Safety
    """
    text_lower = text.lower()

    # Simple keyword matching
    toxic_found = []
    for word in TOXIC_WORDS:
        if re.search(r'\b' + word + r'\b', text_lower):
            toxic_found.append(word)

    is_toxic = len(toxic_found) > 0
    toxicity_score = min(len(toxic_found) * 0.3, 1.0)  # Simple scoring

    return {
        "is_toxic": is_toxic,
        "toxicity_score": toxicity_score,
        "toxic_words": toxic_found
    }


# ============================================================================
# CONTENT POLICY ENFORCEMENT
# ============================================================================

PROHIBITED_TOPICS = [
    r"how to (make|build|create) (a bomb|explosive|weapon)",
    r"(illegal|ilegal) (drugs|drogas)",
    r"(hack|hacking|steal)\s+(a\s+)?(password|data|information)",
]


def check_content_policy(text: str) -> Dict:
    """
    Check whether text violates content policy.
    """
    text_lower = text.lower()

    for pattern in PROHIBITED_TOPICS:
        if re.search(pattern, text_lower):
            return {
                "violates_policy": True,
                "reason": f"Prohibited topic detected: {pattern}"
            }

    return {
        "violates_policy": False,
        "reason": None
    }


# ============================================================================
# OUTPUT GUARDRAIL PIPELINE
# ============================================================================

def output_guardrail(llm_output: str) -> Dict:
    """
    Pipeline completo para filtrar output del LLM.

    Returns:
        {
            "safe_output": str or None,
            "is_safe": bool,
            "issues": List[str]
        }
    """
    issues = []

    # 1. Check PII
    pii_result = redact_pii(llm_output)
    if pii_result["pii_found"]:
        issues.append(f"PII detected: {len(pii_result['pii_found'])} instances")
        llm_output = pii_result["redacted_text"]

    # 2. Check toxicity
    toxicity_result = detect_toxicity(llm_output)
    if toxicity_result["is_toxic"]:
        issues.append(f"Toxicity detected (score: {toxicity_result['toxicity_score']:.2f})")
        return {
            "safe_output": None,
            "is_safe": False,
            "issues": issues + [f"Toxic words: {toxicity_result['toxic_words']}"]
        }

    # 3. Check content policy
    policy_result = check_content_policy(llm_output)
    if policy_result["violates_policy"]:
        issues.append(policy_result["reason"])
        return {
            "safe_output": None,
            "is_safe": False,
            "issues": issues
        }

    # All checks passed
    return {
        "safe_output": llm_output,
        "is_safe": True,
        "issues": issues  # May include PII redaction notice
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("OUTPUT FILTERING GUARDRAILS")
    print("="*70)

    # Case 1: Safe output
    print("\n1. Safe output:")
    safe_output = "Paris is the capital of France. It's a beautiful city with rich history."
    result = output_guardrail(safe_output)
    print(f"Output: {safe_output}")
    print(f"✅ Safe: {result['is_safe']}")
    print(f"Safe Output: {result['safe_output']}")

    # Case 2: Output with PII
    print("\n2. Output with PII:")
    pii_output = "Contact John at john.doe@example.com or call 555-123-4567."
    result = output_guardrail(pii_output)
    print(f"Output: {pii_output}")
    print(f"✅ Safe: {result['is_safe']}")
    print(f"Safe Output: {result['safe_output']}")
    print(f"Issues: {result['issues']}")

    # Case 3: Toxic output
    print("\n3. Toxic output:")
    toxic_output = "That's a stupid idea and you're an idiot for thinking that."
    result = output_guardrail(toxic_output)
    print(f"Output: {toxic_output}")
    print(f"❌ Safe: {result['is_safe']}")
    print(f"Safe Output: {result['safe_output']}")
    print(f"Issues: {result['issues']}")

    # Case 4: Policy violation
    print("\n4. Policy violation:")
    policy_output = "Here's how to hack a password using brute force techniques..."
    result = output_guardrail(policy_output)
    print(f"Output: {policy_output}")
    print(f"❌ Safe: {result['is_safe']}")
    print(f"Safe Output: {result['safe_output']}")
    print(f"Issues: {result['issues']}")

    # Case 5: Output with credit card number
    print("\n5. Output with card number:")
    cc_output = "The customer's card number is 4532-1234-5678-9010."
    result = output_guardrail(cc_output)
    print(f"Output: {cc_output}")
    print(f"✅ Safe: {result['is_safe']}")
    print(f"Safe Output: {result['safe_output']}")
    print(f"Issues: {result['issues']}")
