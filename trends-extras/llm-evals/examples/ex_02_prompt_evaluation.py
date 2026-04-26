"""
Prompt Evaluation & A/B Testing
================================
Compare prompts to select the best one.
Systematic A/B testing for prompt engineering.

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
    Mock LLM for demo (in production use OpenAI/Anthropic/etc).
    """
    # Simulate different prompt qualities
    if "concise" in prompt.lower():
        # Good prompt
        responses = [
            "Paris is the capital of France.",
            "The capital of France is Paris.",
        ]
    elif "answer" in prompt.lower() or "question" in prompt.lower():
        # Medium prompt
        responses = [
            "The capital of France is Paris, a beautiful city with rich history.",
            "Paris. It's a very beautiful city.",
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
    """Evaluation result."""
    prompt_id: str
    avg_length: float
    correct_language: float
    concise: float
    overall_score: float


class PromptEvaluator:
    """
    Evaluate prompts automatically.
    """

    def __init__(self, test_questions: List[str]):
        self.test_questions = test_questions

    def evaluate_prompt(self, prompt_template: str, prompt_id: str) -> EvalResult:
        """
        Evaluate a prompt on the test set.
        """
        responses = []

        for question in self.test_questions:
            prompt = prompt_template.format(question=question)
            response = mock_llm(prompt)
            responses.append(response)

        # Calculate metrics
        avg_length = sum(len(r.split()) for r in responses) / len(responses)

        # Check if answer is non-empty
        correct_language = sum(
            len(r.strip()) > 0
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
    Evaluate prompt with real LLM.
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
        Evaluate this response on a scale of 1-5:

        Question: {question}
        Response: {response}

        Criteria: clarity, conciseness, correctness
        Reply with just the number.
        '''

        score = float(openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": eval_prompt}]
        ).choices[0].message.content.strip())

        scores.append(score)

    return sum(scores) / len(scores)


# A/B Testing con LLM real
prompts = {
    "A": "Answer: {question}",
    "B": "You are a helpful assistant. Answer concisely: {question}"
}

test_set = [
    "What is the capital of France?",
    "What is photosynthesis?",
    "Who wrote Don Quixote?"
]

for prompt_id, template in prompts.items():
    score = evaluate_prompt_real(template, test_set)
    print(f"Prompt {prompt_id}: {score:.2f}/5.0")
"""


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_basic_ab_test():
    """Basic A/B test."""
    print("="*70)
    print("DEMO 1: Basic A/B Test")
    print("="*70 + "\n")

    # Test set
    test_questions = [
        "What is the capital of France?",
        "What is the capital of Spain?",
        "What is the capital of Italy?",
    ]

    # Prompts to test
    prompts = {
        "A_baseline": "Answer: {question}",
        "B_role": "You are a helpful assistant. Answer: {question}",
        "C_detailed": "You are a helpful assistant. Answer concisely: {question}",
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
    """Iterative prompt improvement."""
    print("\n" + "="*70)
    print("DEMO 2: Iterative Prompt Improvement")
    print("="*70 + "\n")

    test_questions = [
        "What is the capital of France?",
        "Who wrote Don Quixote?",
    ]

    iterations = [
        ("v1_basic", "Answer: {question}"),
        ("v2_role", "You are an expert. Answer: {question}"),
        ("v3_constraints", "You are an expert. Answer in maximum 10 words: {question}"),
        ("v4_format", "You are an expert. Answer concisely and clearly: {question}"),
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
    """Statistical significance test."""
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
    """Multi-dimensional evaluation."""
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
    """Prompt versioning."""
    print("\n" + "="*70)
    print("DEMO 5: Prompt Versioning")
    print("="*70 + "\n")

    print("📝 PROMPT VERSIONING SYSTEM:\n")
    print("""
    prompts/
      ├── v1.0_baseline.txt
      │     "Respond, Response, Responds, Responded, Responder: {question}"
      │     Score: 0.65
      │
      ├── v1.1_add_role.txt
      │     "Eres un experto. Respond, Response, Responds, Responded, Responder: {question}"
      │     Score: 0.72 (↑ 10.8%)
      │
      ├── v2.0_add_constraints.txt
      │     "You are an expert. Answer concisely..."
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
    print("REAL CODE (with OpenAI):")
    print("="*70)
    print(REAL_CODE)
