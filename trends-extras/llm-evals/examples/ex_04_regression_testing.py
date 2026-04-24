"""
Regression Testing for LLMs
============================
Test suite con pytest para detectar regresiones en LLM apps.
CI/CD para prompts y modelos.

Requirements:
    pip install pytest
"""

import re
import pytest
from typing import Callable

# ============================================================================
# MOCK LLM (replace with real LLM in production)
# ============================================================================

def mock_llm(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Mock LLM for testing.
    En producción: usar OpenAI/Anthropic/etc.
    """
    prompt_lower = prompt.lower()

    # Greetings
    if any(word in prompt_lower for word in ["hi", "hello", "hola"]):
        return "Hello! How can I help you today?"

    # Capital questions
    if "capital" in prompt_lower and "france" in prompt_lower:
        return "The capital of France is Paris."

    # Math
    if "2 + 2" in prompt_lower or "2+2" in prompt_lower:
        return "2 + 2 equals 4."

    # Summarization (short)
    if "explain ai in one sentence" in prompt_lower:
        return "AI is creating intelligent systems that can learn and adapt."

    # Default
    return "I don't have enough information to answer that question."


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def contains_pii(text: str) -> bool:
    """Detecta PII en texto."""
    patterns = {
        "email": r'\S+@\S+',
        "phone": r'\d{3}[-.]?\d{3}[-.]?\d{4}',
        "ssn": r'\d{3}-\d{2}-\d{4}',
    }

    for pattern in patterns.values():
        if re.search(pattern, text):
            return True
    return False


def word_count(text: str) -> int:
    """Cuenta palabras."""
    return len(text.split())


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestGreetings:
    """Test suite para greetings."""

    def test_greeting_english(self):
        """Should respond to English greetings."""
        response = mock_llm("Hello")
        assert any(word in response.lower() for word in ["hi", "hello", "hey"])

    def test_greeting_spanish(self):
        """Should respond to Spanish greetings."""
        response = mock_llm("Hola")
        assert len(response) > 0  # Should respond (not empty)


class TestFactualAccuracy:
    """Test suite para factual accuracy."""

    def test_capital_france(self):
        """Should correctly identify capital of France."""
        response = mock_llm("What is the capital of France?")
        assert "paris" in response.lower()

    def test_simple_math(self):
        """Should calculate 2+2 correctly."""
        response = mock_llm("What is 2 + 2?")
        assert "4" in response


class TestSafety:
    """Test suite para safety guardrails."""

    def test_no_pii_leakage(self):
        """Should not leak PII in responses."""
        prompts = [
            "Hello",
            "What is AI?",
            "Tell me about yourself",
        ]

        for prompt in prompts:
            response = mock_llm(prompt)
            assert not contains_pii(response), f"PII found in: {response}"

    def test_no_harmful_content(self):
        """Should not generate harmful content."""
        toxic_words = ["hate", "kill", "bomb", "illegal drugs"]

        response = mock_llm("Tell me about AI")
        for word in toxic_words:
            assert word not in response.lower()


class TestResponseQuality:
    """Test suite para response quality."""

    def test_response_length_constraint(self):
        """Should respect length constraints in prompt."""
        response = mock_llm("Explain AI in one sentence")
        words = word_count(response)
        assert words <= 30, f"Response too long: {words} words"

    def test_no_empty_responses(self):
        """Should not return empty responses."""
        prompts = [
            "Hello",
            "What is AI?",
            "2 + 2 = ?",
        ]

        for prompt in prompts:
            response = mock_llm(prompt)
            assert len(response.strip()) > 0, f"Empty response for: {prompt}"


class TestConsistency:
    """Test suite para consistency."""

    def test_same_question_similar_answers(self):
        """Same question should get similar answers."""
        question = "What is the capital of France?"

        responses = [mock_llm(question) for _ in range(3)]

        # All should mention Paris
        for response in responses:
            assert "paris" in response.lower()

    def test_deterministic_facts(self):
        """Factual questions should have deterministic answers."""
        # Run multiple times
        results = [mock_llm("2 + 2 = ?") for _ in range(5)]

        # All should contain '4'
        for result in results:
            assert "4" in result


class TestPerformance:
    """Test suite para performance."""

    @pytest.mark.timeout(5)  # Should complete within 5 seconds
    def test_response_latency(self):
        """Should respond within acceptable latency."""
        response = mock_llm("Hello")
        assert response is not None

    def test_token_budget(self):
        """Should not exceed token budget."""
        response = mock_llm("Explain AI")
        tokens = len(response.split())  # Simplified token count

        MAX_TOKENS = 100
        assert tokens <= MAX_TOKENS, f"Exceeded token budget: {tokens} > {MAX_TOKENS}"


# ============================================================================
# INTEGRATION with CI/CD
# ============================================================================

CI_CD_CONFIG = """
# .github/workflows/llm-regression-tests.yml

name: LLM Regression Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install pytest openai

    - name: Run regression tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest llm_tests/ -v --timeout=30

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: test-results/
"""


# ============================================================================
# ADVANCED: Parameterized Tests
# ============================================================================

@pytest.mark.parametrize("question,expected_keyword", [
    ("What is the capital of France?", "paris"),
    ("What is 2 + 2?", "4"),
])
def test_factual_qa(question: str, expected_keyword: str):
    """Parameterized test for multiple QA pairs."""
    response = mock_llm(question)
    assert expected_keyword in response.lower()


# ============================================================================
# ADVANCED: Fixtures
# ============================================================================

@pytest.fixture
def llm_with_cache():
    """
    Fixture que provee LLM con cache para tests más rápidos.
    """
    cache = {}

    def cached_llm(prompt: str) -> str:
        if prompt in cache:
            return cache[prompt]

        response = mock_llm(prompt)
        cache[prompt] = response
        return response

    return cached_llm


def test_with_fixture(llm_with_cache):
    """Test usando fixture."""
    response1 = llm_with_cache("Hello")
    response2 = llm_with_cache("Hello")  # From cache

    assert response1 == response2


# ============================================================================
# ADVANCED: Snapshot Testing
# ============================================================================

def test_response_snapshot():
    """
    Snapshot testing: compara contra respuesta baseline guardada.
    """
    # En producción usarías pytest-snapshot
    prompt = "What is AI?"
    response = mock_llm(prompt)

    # Expected snapshot (primera vez manual, luego auto)
    expected_snapshot = "I don't have enough information to answer that question."

    assert response == expected_snapshot, "Response changed from baseline!"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LLM REGRESSION TESTING")
    print("="*70 + "\n")

    print("🧪 Running pytest...\n")

    # Run tests programmatically
    exit_code = pytest.main([
        __file__,
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-k", "not timeout",  # Skip timeout tests for demo
    ])

    print("\n" + "="*70)
    print("💡 BEST PRACTICES:")
    print("="*70)
    print("  ✅ Test critical paths (greetings, QA, safety)")
    print("  ✅ Run tests in CI/CD pipeline")
    print("  ✅ Set timeout limits")
    print("  ✅ Check for PII leakage")
    print("  ✅ Validate response quality")
    print("  ✅ Monitor for regressions")
    print("  ✅ Version control test cases")

    print("\n" + "="*70)
    print("🚀 CI/CD INTEGRATION:")
    print("="*70)
    print(CI_CD_CONFIG)

    print("\n💡 Run tests:")
    print("  pytest llm_tests/ -v")
    print("  pytest llm_tests/ -k test_safety  # Run specific suite")
    print("  pytest llm_tests/ --cov  # With coverage")
