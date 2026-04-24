"""
Prompt Evaluation & A/B Testing
================================
Compara prompts para seleccionar el mejor.
A/B testing sistemático para prompt engineering.

Requirements:
    pip install openai  # o tu LLM provider
"""

import random
from typing import List, Dict, Any
from dataclasses import dataclass

# ============================================================================
# MOCK LLM (para demo sin API keys)
# ============================================================================

def mock_llm(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Mock LLM para demo (en producción usar OpenAI/Anthropic/etc).
    """
    # Simulate different prompt qualities
    if "español" in prompt.lower() and "concisa" in prompt.lower():
        # Good prompt
        responses = [
            "París es la capital de Francia.",
            "La capital de Francia es París.",
        ]
    elif "español" in prompt.lower():
        # Medium prompt
        responses = [
            "La capital de Francia es París, una ciudad hermosa con rica historia.",
            "París. Es una ciudad muy bonita.",
        ]
    else:
        # Basic prompt
        responses = [
            "The capital of France is Paris, which is a beautiful city...",
            "Paris is the capital.",
        ]

    return random.choice(responses)


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

@dataclass
class EvalResult:
    """Resultado de evaluación."""
    prompt_id: str
    avg_length: float
    correct_language: float
    concise: float
    overall_score: float


class PromptEvaluator:
    """
    Evalúa prompts automáticamente.
    """

    def __init__(self, test_questions: List[str]):
        self.test_questions = test_questions

    def evaluate_prompt(self, prompt_template: str, prompt_id: str) -> EvalResult:
        """
        Evalúa un prompt en el test set.
        """
        responses = []

        for question in self.test_questions:
            prompt = prompt_template.format(question=question)
            response = mock_llm(prompt)
            responses.append(response)

        # Calculate metrics
        avg_length = sum(len(r.split()) for r in responses) / len(responses)

        # Check if Spanish
        spanish_words = ["es", "la", "de", "el", "una"]
        correct_language = sum(
            any(word in r.lower() for word in spanish_words)
            for r in responses
        ) / len(responses)

        # Check conciseness (< 15 words)
        concise = sum(len(r.split()) <= 15 for r in responses) / len(responses)

        # Overall score (weighted)
        overall = (
            correct_language * 0.5 +
            concise * 0.3 +
            (1 - min(avg_length / 20, 1.0)) * 0.2  # Penalize verbosity
        )

        return EvalResult(
            prompt_id=prompt_id,
            avg_length=avg_length,
            correct_language=correct_language,
            concise=concise,
            overall_score=overall
        )


# ============================================================================
# A/B TESTING
# ============================================================================

def ab_test_prompts(
    prompts: Dict[str, str],
    test_set: List[str],
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    A/B test multiple prompts.

    Args:
        prompts: {prompt_id: prompt_template}
        test_set: List of test questions
        metrics: Metrics to report

    Returns:
        Results dict with winner
    """
    evaluator = PromptEvaluator(test_set)
    results = {}

    print("🧪 Running A/B Test...\n")

    for prompt_id, prompt_template in prompts.items():
        print(f"📝 Testing Prompt {prompt_id}:")
        print(f"   {prompt_template[:80]}...")

        result = evaluator.evaluate_prompt(prompt_template, prompt_id)
        results[prompt_id] = result

        print(f"   ✅ Overall Score: {result.overall_score:.3f}\n")

    # Determine winner
    winner = max(results.items(), key=lambda x: x[1].overall_score)

    return {
        "results": results,
        "winner": winner[0],
        "winner_score": winner[1].overall_score
    }


# ============================================================================
# REAL LLM USAGE (Commented)
# ============================================================================

REAL_CODE = """
import openai
from typing import List

def evaluate_prompt_real(prompt_template: str, test_set: List[str]) -> float:
    '''
    Evalúa prompt con LLM real.
    '''
    scores = []

    for question in test_set:
        # 1. Generate response
        prompt = prompt_template.format(question=question)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        # 2. Evaluate with GPT-4 as judge
        eval_prompt = f'''
        Evalúa esta respuesta en escala 1-5:

        Pregunta: {question}
        Respuesta: {response}

        Criterios: claridad, concisión, corrección
        Responde solo el número.
        '''

        score = float(openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": eval_prompt}]
        ).choices[0].message.content.strip())

        scores.append(score)

    return sum(scores) / len(scores)


# A/B Testing con LLM real
prompts = {
    "A": "Responde: {question}",
    "B": "Eres un asistente útil. Responde en español de forma concisa: {question}"
}

test_set = [
    "¿Cuál es la capital de Francia?",
    "¿Qué es la fotosíntesis?",
    "¿Quién escribió Don Quijote?"
]

for prompt_id, template in prompts.items():
    score = evaluate_prompt_real(template, test_set)
    print(f"Prompt {prompt_id}: {score:.2f}/5.0")
"""


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_basic_ab_test():
    """A/B test básico."""
    print("="*70)
    print("DEMO 1: Basic A/B Test")
    print("="*70 + "\n")

    # Test set
    test_questions = [
        "¿Cuál es la capital de Francia?",
        "¿Cuál es la capital de España?",
        "¿Cuál es la capital de Italia?",
    ]

    # Prompts to test
    prompts = {
        "A_baseline": "Responde: {question}",
        "B_role": "Eres un asistente útil. Responde: {question}",
        "C_detailed": "Eres un asistente útil. Responde en español de forma concisa: {question}",
    }

    results = ab_test_prompts(prompts, test_questions)

    print("🏆 RESULTS:")
    print(f"   Winner: Prompt {results['winner']}")
    print(f"   Score:  {results['winner_score']:.3f}\n")

    # Detailed comparison
    print("📊 Detailed Comparison:")
    for prompt_id, result in results['results'].items():
        print(f"\n  {prompt_id}:")
        print(f"    Overall:     {result.overall_score:.3f}")
        print(f"    Language:    {result.correct_language:.1%}")
        print(f"    Concise:     {result.concise:.1%}")
        print(f"    Avg Length:  {result.avg_length:.1f} words")


def demo_iterative_improvement():
    """Mejora iterativa de prompts."""
    print("\n" + "="*70)
    print("DEMO 2: Iterative Prompt Improvement")
    print("="*70 + "\n")

    test_questions = [
        "¿Cuál es la capital de Francia?",
        "¿Quién escribió Don Quijote?",
    ]

    iterations = [
        ("v1_basic", "Responde: {question}"),
        ("v2_role", "Eres un experto. Responde: {question}"),
        ("v3_constraints", "Eres un experto. Responde en español, máximo 10 palabras: {question}"),
        ("v4_format", "Eres un experto. Responde en español de forma concisa y clara: {question}"),
    ]

    print("🔄 Iterative Improvement:\n")

    previous_score = 0
    for version, prompt in iterations:
        evaluator = PromptEvaluator(test_questions)
        result = evaluator.evaluate_prompt(prompt, version)

        improvement = result.overall_score - previous_score
        emoji = "📈" if improvement > 0 else "📉"

        print(f"{emoji} {version}: {result.overall_score:.3f} (Δ {improvement:+.3f})")
        previous_score = result.overall_score

    print("\n💡 Iterate until convergence or diminishing returns")


def demo_statistical_significance():
    """Test de significancia estadística."""
    print("\n" + "="*70)
    print("DEMO 3: Statistical Significance")
    print("="*70 + "\n")

    print("⚠️  Common Mistake: Declaring winner too early!\n")

    # Small test set
    print("🔸 Test with 3 questions:")
    print("   Prompt A: 0.85")
    print("   Prompt B: 0.87")
    print("   Difference: 0.02")
    print("   ❌ NOT statistically significant (too few samples)\n")

    # Large test set
    print("🔸 Test with 100 questions:")
    print("   Prompt A: 0.85 ± 0.03")
    print("   Prompt B: 0.87 ± 0.02")
    print("   p-value: 0.03")
    print("   ✅ Statistically significant (p < 0.05)\n")

    print("💡 Best Practices:")
    print("   • Use ≥50 test questions")
    print("   • Run multiple trials")
    print("   • Calculate confidence intervals")
    print("   • Use paired t-test")


def demo_multidimensional_evaluation():
    """Evaluación multi-dimensional."""
    print("\n" + "="*70)
    print("DEMO 4: Multi-Dimensional Evaluation")
    print("="*70 + "\n")

    print("📊 Don't just optimize for ONE metric!\n")

    print("Prompt A:")
    print("  Accuracy:  ████████░░ 80%")
    print("  Latency:   ██░░░░░░░░ 2.5s")
    print("  Cost:      ████████░░ $0.05")
    print("  Concise:   ███████░░░ 70%\n")

    print("Prompt B:")
    print("  Accuracy:  ██████░░░░ 60%")
    print("  Latency:   █████████░ 0.8s")
    print("  Cost:      ██░░░░░░░░ $0.01")
    print("  Concise:   █████████░ 90%\n")

    print("💡 Choose based on priorities:")
    print("   • Production app → Optimize latency + cost")
    print("   • Critical QA → Optimize accuracy")
    print("   • User-facing → Optimize conciseness")


def demo_prompt_versioning():
    """Versionado de prompts."""
    print("\n" + "="*70)
    print("DEMO 5: Prompt Versioning")
    print("="*70 + "\n")

    print("📝 PROMPT VERSIONING SYSTEM:\n")
    print("""
    prompts/
      ├── v1.0_baseline.txt
      │     "Responde: {question}"
      │     Score: 0.65
      │
      ├── v1.1_add_role.txt
      │     "Eres un experto. Responde: {question}"
      │     Score: 0.72 (↑ 10.8%)
      │
      ├── v2.0_add_constraints.txt
      │     "Eres un experto. Responde en español..."
      │     Score: 0.85 (↑ 18.1%)
      │     ⭐ PRODUCTION
      │
      └── v2.1_experimental.txt
            "Eres Claude, un asistente..."
            Score: 0.83 (↓ 2.4%)
            ❌ Rejected
    """)

    print("💡 Benefits:")
    print("   • Reproducibility")
    print("   • Rollback capability")
    print("   • Track improvements over time")


if __name__ == "__main__":
    print("\n🎯 PROMPT EVALUATION & A/B TESTING")
    print("📊 Data-driven prompt engineering\n")

    demo_basic_ab_test()
    demo_iterative_improvement()
    demo_statistical_significance()
    demo_multidimensional_evaluation()
    demo_prompt_versioning()

    print("\n" + "="*70)
    print("💡 BEST PRACTICES:")
    print("="*70)
    print("  ✅ Test with ≥50 diverse examples")
    print("  ✅ Use multiple evaluation metrics")
    print("  ✅ Check statistical significance")
    print("  ✅ Version control your prompts")
    print("  ✅ A/B test in production (slowly)")
    print("  ✅ Monitor for regressions")
    print("  ✅ Document why changes were made")

    print("\n" + "="*70)
    print("CÓDIGO REAL (con OpenAI):")
    print("="*70)
    print(REAL_CODE)
