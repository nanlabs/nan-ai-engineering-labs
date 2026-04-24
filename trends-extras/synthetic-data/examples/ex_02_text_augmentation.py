"""
Text Data Augmentation with NLP Techniques
===========================================
Augmenta datasets de texto para training con back-translation, paraphrasing, y EDA.

Requirements:
    pip install transformers torch nltk sentence-transformers nlpaug
"""

from transformers import MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet
import random
import torch
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


# ============================================================================
# TECHNIQUE 1: BACK-TRANSLATION
# ============================================================================

class BackTranslator:
    """
    Back-translation: en → es → en para parafrasear.
    Preserva significado pero cambia wording.
    """

    def __init__(self):
        print("📥 Loading translation models...")

        # English to Spanish
        self.en_es_model_name = 'Helsinki-NLP/opus-mt-en-es'
        self.en_es_tokenizer = MarianTokenizer.from_pretrained(self.en_es_model_name)
        self.en_es_model = MarianMTModel.from_pretrained(self.en_es_model_name)

        # Spanish to English
        self.es_en_model_name = 'Helsinki-NLP/opus-mt-es-en'
        self.es_en_tokenizer = MarianTokenizer.from_pretrained(self.es_en_model_name)
        self.es_en_model = MarianMTModel.from_pretrained(self.es_en_model_name)

        print("   ✅ Models loaded\n")

    def translate(self, text: str, model, tokenizer) -> str:
        """Translate text."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def augment(self, text: str) -> str:
        """Back-translate: en → es → en."""
        # Translate to Spanish
        spanish = self.translate(text, self.en_es_model, self.en_es_tokenizer)

        # Translate back to English
        back_translated = self.translate(spanish, self.es_en_model, self.es_en_tokenizer)

        return back_translated


# ============================================================================
# TECHNIQUE 2: PARAPHRASING WITH T5
# ============================================================================

class T5Paraphraser:
    """
    T5-based paraphrasing.
    """

    def __init__(self):
        print("📥 Loading T5 paraphraser...")

        self.model_name = "t5-small"  # or "t5-base" for better quality
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        print("   ✅ Model loaded\n")

    def augment(self, text: str, num_paraphrases: int = 1) -> List[str]:
        """
        Generate paraphrases.
        """
        # T5 expects "paraphrase: {text}"
        input_text = f"paraphrase: {text} </s>"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Generate with beam search for diversity
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=num_paraphrases + 2,
            num_return_sequences=num_paraphrases,
            temperature=1.5,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        paraphrases = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return paraphrases


# ============================================================================
# TECHNIQUE 3: EDA (Easy Data Augmentation)
# ============================================================================

class EDA:
    """
    EDA operations:
    - Synonym Replacement (SR): Replace words with synonyms
    - Random Insertion (RI): Insert random synonyms
    - Random Swap (RS): Swap positions of words
    - Random Deletion (RD): Delete words randomly
    """

    @staticmethod
    def get_synonyms(word: str) -> List[str]:
        """Get synonyms from WordNet."""
        synonyms = set()

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)

        return list(synonyms)

    @staticmethod
    def synonym_replacement(sentence: str, n: int = 2) -> str:
        """Replace n words with synonyms."""
        words = sentence.split()
        random_word_indices = random.sample(range(len(words)), min(n, len(words)))

        for idx in random_word_indices:
            synonyms = EDA.get_synonyms(words[idx])
            if synonyms:
                words[idx] = random.choice(synonyms)

        return ' '.join(words)

    @staticmethod
    def random_insertion(sentence: str, n: int = 2) -> str:
        """Insert n random synonyms."""
        words = sentence.split()

        for _ in range(n):
            # Get random word and its synonym
            random_word = random.choice(words)
            synonyms = EDA.get_synonyms(random_word)

            if synonyms:
                random_synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, random_synonym)

        return ' '.join(words)

    @staticmethod
    def random_swap(sentence: str, n: int = 2) -> str:
        """Swap positions of n random words."""
        words = sentence.split()

        for _ in range(n):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    @staticmethod
    def random_deletion(sentence: str, p: float = 0.1) -> str:
        """Delete words with probability p."""
        words = sentence.split()

        if len(words) == 1:
            return sentence

        words = [word for word in words if random.random() > p]

        if not words:
            return random.choice(sentence.split())

        return ' '.join(words)


# ============================================================================
# SEMANTIC SIMILARITY VALIDATION
# ============================================================================

class SemanticValidator:
    """
    Valida que augmentations preserven el significado.
    """

    def __init__(self):
        print("📥 Loading sentence transformer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ Model loaded\n")

    def validate(self, original: str, augmented: str, threshold: float = 0.75) -> Dict:
        """
        Calcula similaridad semántica.

        Returns:
            dict with similarity score and validation result
        """
        # Encode sentences
        emb1 = self.model.encode(original, convert_to_tensor=True)
        emb2 = self.model.encode(augmented, convert_to_tensor=True)

        # Cosine similarity
        similarity = util.pytorch_cos_sim(emb1, emb2).item()

        is_valid = similarity >= threshold

        return {
            'similarity': similarity,
            'is_valid': is_valid,
            'threshold': threshold
        }


# ============================================================================
# DEMOS
# ============================================================================

def demo_back_translation():
    """Demo back-translation."""
    print("="*70)
    print("TECHNIQUE 1: Back-Translation")
    print("="*70 + "\n")

    bt = BackTranslator()
    validator = SemanticValidator()

    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we solve problems.",
        "Climate change requires immediate global action."
    ]

    for i, text in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Original:   {text}")

        augmented = bt.augment(text)
        print(f"Augmented:  {augmented}")

        validation = validator.validate(text, augmented)
        print(f"Similarity: {validation['similarity']:.3f} "
              f"({'✅ Valid' if validation['is_valid'] else '❌ Invalid'})\n")


def demo_paraphrasing():
    """Demo T5 paraphrasing."""
    print("="*70)
    print("TECHNIQUE 2: T5 Paraphrasing")
    print("="*70 + "\n")

    paraphraser = T5Paraphraser()
    validator = SemanticValidator()

    text = "Artificial intelligence is revolutionizing healthcare."

    print(f"Original: {text}\n")
    print("Paraphrases:")

    paraphrases = paraphraser.augment(text, num_paraphrases=3)

    for i, para in enumerate(paraphrases, 1):
        validation = validator.validate(text, para)
        print(f"{i}. {para}")
        print(f"   Similarity: {validation['similarity']:.3f} "
              f"({'✅' if validation['is_valid'] else '❌'})\n")


def demo_eda():
    """Demo EDA techniques."""
    print("="*70)
    print("TECHNIQUE 3: EDA (Easy Data Augmentation)")
    print("="*70 + "\n")

    validator = SemanticValidator()

    text = "The weather is beautiful today and I feel happy."

    print(f"Original: {text}\n")

    techniques = [
        ("Synonym Replacement", lambda: EDA.synonym_replacement(text, n=2)),
        ("Random Insertion", lambda: EDA.random_insertion(text, n=1)),
        ("Random Swap", lambda: EDA.random_swap(text, n=2)),
        ("Random Deletion", lambda: EDA.random_deletion(text, p=0.2))
    ]

    for name, func in techniques:
        augmented = func()
        validation = validator.validate(text, augmented)

        print(f"{name}:")
        print(f"  {augmented}")
        print(f"  Similarity: {validation['similarity']:.3f} "
              f"({'✅' if validation['is_valid'] else '❌'})\n")


def demo_augment_dataset():
    """Demo: augment entire dataset."""
    print("="*70)
    print("DEMO: Augment Complete Dataset")
    print("="*70 + "\n")

    # Sample sentiment analysis dataset
    dataset = [
        ("This movie was absolutely fantastic!", "positive"),
        ("Terrible service, would not recommend.", "negative"),
        ("The product quality exceeded my expectations.", "positive"),
        ("Disappointing experience overall.", "negative"),
    ]

    print(f"Original dataset size: {len(dataset)}\n")

    # Apply back-translation augmentation
    bt = BackTranslator()
    augmented_dataset = []

    print("Augmenting with back-translation...\n")

    for text, label in dataset:
        # Original
        augmented_dataset.append((text, label))

        # Augmented
        aug_text = bt.augment(text)
        augmented_dataset.append((aug_text, label))

        print(f"Original:  {text}")
        print(f"Augmented: {aug_text}")
        print(f"Label: {label}\n")

    print(f"Augmented dataset size: {len(augmented_dataset)}")
    print(f"Size increase: {len(augmented_dataset) / len(dataset):.1f}x\n")


def demo_best_practices():
    """Best practices."""
    print("="*70)
    print("BEST PRACTICES")
    print("="*70 + "\n")

    print("""
1. ✅ VALIDATE SEMANTIC SIMILARITY
   ────────────────────────────────────────
   • Always check augmented text preserves meaning
   • Use semantic similarity threshold (0.75-0.85)
   • Manually review samples

2. ✅ COMBINE MULTIPLE TECHNIQUES
   ────────────────────────────────────────
   • Back-translation for paraphrasing
   • EDA for diversity
   • Paraphrasing for quality
   • Mix techniques for 3-5x augmentation

3. ✅ MAINTAIN LABEL CONSISTENCY
   ────────────────────────────────────────
   • Augmentation should NOT change the label
   • For sentiment: "great" → "excellent" (OK)
   • But NOT "great" → "terrible" (breaks label)

4. ⚠️  AVOID OVER-AUGMENTATION
   ────────────────────────────────────────
   • Too much augmentation can hurt performance
   • Start with 2-3x augmentation
   • Monitor validation performance

5. ✅ STRATIFIED AUGMENTATION
   ────────────────────────────────────────
   • Augment minority classes more
   • Balance dataset with augmentation
   • E.g., 1000 positive + 100 negative → augment negative 10x

6. ✅ QUALITY > QUANTITY
   ────────────────────────────────────────
   • Better to have 100 high-quality augmented samples
   • Than 1000 low-quality ones
   • Always validate!
    """)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 TEXT DATA AUGMENTATION")
    print("="*70 + "\n")

    demo_back_translation()
    demo_paraphrasing()
    demo_eda()
    demo_augment_dataset()
    demo_best_practices()

    print("="*70)
    print("USE CASES")
    print("="*70)
    print("  • Low-resource tasks (limited training data)")
    print("  • Imbalanced datasets (augment minority class)")
    print("  • Improve model robustness")
    print("  • Data augmentation for fine-tuning")
    print("  • Test model sensitivity to paraphrasing")

    print("\n📚 Resources:")
    print("  • EDA Paper: https://arxiv.org/abs/1901.11196")
    print("  • Back-translation: https://arxiv.org/abs/1511.06709")
    print("  • NLP Aug library: https://github.com/makcedward/nlpaug")
