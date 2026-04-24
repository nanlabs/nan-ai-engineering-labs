"""
Automatic Metrics for LLM Evaluation
=====================================
Calcula métricas automáticas: BLEU, ROUGE, BERTScore.
Útil para evaluar quality sin human evaluation costosa.

Requirements:
    pip install rouge-score sacrebleu bert-score nltk
"""

import re
from typing import List, Dict

# ============================================================================
# SIMPLIFIED METRIC IMPLEMENTATIONS (for demo)
# ============================================================================

def calculate_bleu_1(reference: str, candidate: str) -> float:
    """
    BLEU-1: Unigram (palabra) overlap entre reference y candidate.

    BLEU real usa n-gramas (1-4) y brevity penalty.
    Aquí simplificado para demostración.
    """
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()

    # Count matching words
    matches = sum(1 for word in cand_words if word in ref_words)

    # Precision
    precision = matches / len(cand_words) if cand_words else 0

    return precision


def calculate_rouge_l(reference: str, candidate: str) -> Dict[str, float]:
    """
    ROUGE-L: Longest Common Subsequence entre reference y candidate.

    Mide recall-oriented overlap (importante para summarization).
    """
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()

    # LCS (simplificado)
    def lcs_length(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    lcs_len = lcs_length(ref_words, cand_words)

    # Precision, Recall, F1
    precision = lcs_len / len(cand_words) if cand_words else 0
    recall = lcs_len / len(ref_words) if ref_words else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================================
# REAL LIBRARY USAGE (Commented)
# ============================================================================

REAL_METRICS_CODE = """
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score

# ============================================================================
# 1. BLEU (Bilingual Evaluation Understudy)
# ============================================================================

# BLEU mide n-gram overlap (1-4 gramas)
# Usado en machine translation, pero también text generation

references = ["The cat sat on the mat"]
candidate = "A cat was sitting on the mat"

bleu = BLEU()
score = bleu.corpus_score([candidate], [[ref] for ref in references])
print(f"BLEU: {score.score:.2f}")
# Output: ~40-60 típico para buena generación

# ============================================================================
# 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
# ============================================================================

# ROUGE mide recall (importante para summarization)
# ROUGE-1: unigrams, ROUGE-2: bigrams, ROUGE-L: longest common subsequence

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, candidate)

print(f"ROUGE-1: {scores['rouge1'].fmeasure:.3f}")
print(f"ROUGE-2: {scores['rouge2'].fmeasure:.3f}")
print(f"ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

# Interpretación:
# > 0.5: Excellent
# 0.3-0.5: Good
# < 0.3: Poor

# ============================================================================
# 3. BERTScore (Semantic Similarity)
# ============================================================================

# BERTScore usa embeddings para medir semantic similarity
# Mejor que BLEU/ROUGE para capturar paráfrasis

P, R, F1 = bert_score(
    [candidate],
    [reference],
    lang="en",
    model_type="distilbert-base-uncased"
)

print(f"BERTScore F1: {F1[0]:.3f}")

# Interpretación:
# > 0.90: Excellent
# 0.85-0.90: Good
# < 0.85: Needs improvement
"""


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_translation_metrics():
    """Evaluar traducción."""
    print("="*70)
    print("DEMO 1: Machine Translation Evaluation")
    print("="*70 + "\n")

    reference = "The cat sat on the mat"

    candidates = [
        ("A cat was sitting on the mat", "Good translation"),
        ("The cat on mat sat", "Poor grammar"),
        ("El gato se sentó", "Wrong language"),
    ]

    for candidate, description in candidates:
        print(f"🔹 {description}")
        print(f"   Reference:  {reference}")
        print(f"   Candidate:  {candidate}")

        bleu = calculate_bleu_1(reference, candidate)
        rouge = calculate_rouge_l(reference, candidate)

        print(f"   BLEU-1:    {bleu:.3f}")
        print(f"   ROUGE-L:   {rouge['f1']:.3f}\n")


def demo_summarization_metrics():
    """Evaluar summarization."""
    print("="*70)
    print("DEMO 2: Summarization Evaluation")
    print("="*70 + "\n")

    reference = "AI is transforming industries through automation and data analysis"

    candidates = [
        ("AI transforms industries via automation and analytics", "Good paraphrase"),
        ("Artificial intelligence is changing the world", "Too vague"),
        ("The cat sat on the mat", "Completely wrong"),
    ]

    for candidate, description in candidates:
        print(f"🔹 {description}")
        print(f"   Reference:  {reference}")
        print(f"   Summary:    {candidate}")

        rouge = calculate_rouge_l(reference, candidate)

        print(f"   ROUGE-L F1: {rouge['f1']:.3f}")
        print(f"   Precision:  {rouge['precision']:.3f}")
        print(f"   Recall:     {rouge['recall']:.3f}\n")


def demo_qa_metrics():
    """Evaluar QA."""
    print("="*70)
    print("DEMO 3: Question Answering Evaluation")
    print("="*70 + "\n")

    question = "What is the capital of France?"
    ground_truth = "Paris"

    candidates = [
        ("Paris", "Exact match"),
        ("The capital is Paris", "Correct but verbose"),
        ("paris", "Case difference"),
        ("Lyon", "Wrong answer"),
    ]

    for candidate, description in candidates:
        print(f"🔹 {description}")
        print(f"   Ground truth: {ground_truth}")
        print(f"   Prediction:   {candidate}")

        # Exact Match (case-insensitive)
        exact_match = ground_truth.lower().strip() == candidate.lower().strip()

        # F1 (token overlap)
        gt_tokens = set(ground_truth.lower().split())
        pred_tokens = set(candidate.lower().split())

        common = gt_tokens & pred_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        print(f"   Exact Match: {exact_match}")
        print(f"   F1 Score:    {f1:.3f}\n")


def demo_metric_comparison():
    """Comparar métricas."""
    print("="*70)
    print("DEMO 4: Metric Comparison")
    print("="*70 + "\n")

    reference = "Machine learning is a subset of artificial intelligence"

    # Paráfrasis perfecta semánticamente
    paraphrase = "ML is part of AI"

    # Overlap exacto pero sin sentido
    nonsense = "Machine learning artificial intelligence subset is a of"

    print("🔹 Paráfrasis semántica:")
    print(f"   Ref: {reference}")
    print(f"   Gen: {paraphrase}")
    bleu_p = calculate_bleu_1(reference, paraphrase)
    rouge_p = calculate_rouge_l(reference, paraphrase)
    print(f"   BLEU-1:  {bleu_p:.3f} (LOW - no exact word matches)")
    print(f"   ROUGE-L: {rouge_p['f1']:.3f}")
    print(f"   BERTScore would be HIGH (~0.90) - captures semantics\n")

    print("🔹 Same words, no sense:")
    print(f"   Ref: {reference}")
    print(f"   Gen: {nonsense}")
    bleu_n = calculate_bleu_1(reference, nonsense)
    rouge_n = calculate_rouge_l(reference, nonsense)
    print(f"   BLEU-1:  {bleu_n:.3f} (HIGH - matches words)")
    print(f"   ROUGE-L: {rouge_n['f1']:.3f}")
    print(f"   BERTScore would be LOW - no semantic sense\n")

    print("💡 LECCIÓN: Usar múltiples métricas!")
    print("   • BLEU/ROUGE: Surface overlap")
    print("   • BERTScore: Semantic similarity")
    print("   • Both together: Comprehensive evaluation")


def demo_metrics_cheatsheet():
    """Cheatsheet de métricas."""
    print("\n" + "="*70)
    print("CHEATSHEET: Choosing the Right Metric")
    print("="*70 + "\n")

    print("📊 METRIC COMPARISON:\n")
    print("╔══════════════╦══════════════╦══════════════╦══════════════╗")
    print("║   Metric     ║   Best For   ║   Strengths  ║  Weaknesses  ║")
    print("╠══════════════╬══════════════╬══════════════╬══════════════╣")
    print("║ BLEU         ║ Translation  ║ Fast, simple ║ Ignores      ║")
    print("║              ║              ║ deterministic║ semantics    ║")
    print("╠══════════════╬══════════════╬══════════════╬══════════════╣")
    print("║ ROUGE        ║ Summariz.    ║ Recall focus ║ Surface      ║")
    print("║              ║              ║ multiple refs║ level only   ║")
    print("╠══════════════╬══════════════╬══════════════╬══════════════╣")
    print("║ BERTScore    ║ Any text     ║ Semantic     ║ Slower,      ║")
    print("║              ║ generation   ║ paraphrases  ║ needs model  ║")
    print("╠══════════════╬══════════════╬══════════════╬══════════════╣")
    print("║ Exact Match  ║ QA, facts    ║ Strict       ║ Too strict,  ║")
    print("║              ║              ║ objective    ║ no partial   ║")
    print("╚══════════════╩══════════════╩══════════════╩══════════════╝")

    print("\n🎯 QUICK GUIDE:\n")
    print("  Use BLEU when:")
    print("    • Exact wording matters (translation)")
    print("    • Fast evaluation needed")
    print("    • Benchmark comparison\n")

    print("  Use ROUGE when:")
    print("    • Summarization")
    print("    • Recall is important")
    print("    • Multiple reference texts\n")

    print("  Use BERTScore when:")
    print("    • Paraphrases are valid")
    print("    • Semantic similarity matters")
    print("    • Creative generation\n")

    print("  Use Exact Match when:")
    print("    • QA tasks")
    print("    • Factual accuracy critical")
    print("    • Binary yes/no\n")


if __name__ == "__main__":
    print("\n🎯 AUTOMATIC METRICS FOR LLM EVALUATION")
    print("📊 Evaluar quality sin human judges\n")

    demo_translation_metrics()
    demo_summarization_metrics()
    demo_qa_metrics()
    demo_metric_comparison()
    demo_metrics_cheatsheet()

    print("\n" + "="*70)
    print("📚 PARA PRODUCCIÓN:")
    print("="*70)
    print(REAL_METRICS_CODE)

    print("\n💡 BEST PRACTICES:")
    print("  ✅ Use multiple metrics (not just one)")
    print("  ✅ Compare against baselines")
    print("  ✅ Correlate with human evaluation")
    print("  ✅ Report confidence intervals")
    print("  ✅ Consider domain-specific metrics")
