"""
Image Captioning with BLIP
===========================
Generate image descriptions using Salesforce BLIP.
BLIP (Bootstrapping Language-Image Pre-training) es SOTA para captioning.

Requirements:
    pip install transformers torch pillow
"""

from PIL import Image

# ============================================================================
# CONCEPTUAL DEMO
# ============================================================================

class MockBLIPDemo:
    """
    Demo conceptual de BLIP image captioning.
    """

    def __init__(self):
        print("🔄 Mock BLIP model loaded (conceptual demo)")
        print("For actual usage: use BlipForConditionalGeneration\n")

    def generate_caption(self, image_path: str, conditional_text: str = None) -> str:
        """
        Genera caption para la image.

        Args:
            image_path: Path to the image
            conditional_text: Conditional text (optional) like "A photo of"
        """
        print(f"📸 Image: {image_path}")

        if conditional_text:
            print(f"💬 Conditional: {conditional_text}")
            # Conditional captioning guides generation
            caption = f"{conditional_text} a golden retriever playing in a park"
        else:
            # Unconditional captioning (free description)
            # In production BLIP generates this automatically
            caption = "a dog playing with a ball in a sunny park"

        print(f"📝 Caption: {caption}\n")
        return caption


# ============================================================================
# REAL BLIP USAGE
# ============================================================================

REAL_BLIP_CODE = """
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. Load image
image = Image.open("photo.jpg")

# 3. Unconditional captioning (free description)
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Caption: {caption}")
# Output: "a dog playing with a ball in a park"

# 4. Conditional captioning (guide the description)
text = "A professional photo of"
inputs = processor(image, text=text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Conditional caption: {caption}")
# Output: "A professional photo of a golden retriever in autumn"
"""


# ============================================================================
# EXAMPLES OF USE
# ============================================================================

def demo_unconditional():
    """Caption without context."""
    print("="*70)
    print("DEMO 1: Unconditional Captioning")
    print("="*70)

    blip = MockBLIPDemo()

    # Different image types
    images = [
        "dog_park.jpg",
        "city_street.jpg",
        "mountain_view.jpg",
        "family_dinner.jpg"
    ]

    for img in images:
        caption = blip.generate_caption(img)


def demo_conditional():
    """Caption with context."""
    print("\n" + "="*70)
    print("DEMO 2: Conditional Captioning")
    print("="*70)

    blip = MockBLIPDemo()

    # Same image, different contexts
    image = "landscape.jpg"

    contexts = [
        "A professional photograph of",
        "An oil painting of",
        "A satellite image of",
        "A vintage photo of"
    ]

    for context in contexts:
        caption = blip.generate_caption(image, conditional_text=context)


def demo_comparison():
    """Compare BLIP vs human."""
    print("\n" + "="*70)
    print("DEMO 3: BLIP vs Human Captions")
    print("="*70)

    image = "beach_sunset.jpg"

    print(f"📸 Image: {image}\n")

    # Human caption
    human_caption = "A breathtaking sunset over the ocean with orange and purple hues"
    print(f"👤 Human: {human_caption}")

    # BLIP caption (mock)
    blip = MockBLIPDemo()
    blip_caption = "a sunset over the beach with colorful sky"
    print(f"🤖 BLIP:  {blip_caption}")

    print("\n💡 BLIP is more concise, humans are more descriptive/emotive")


def demo_multilingual():
    """Multilingual BLIP (with prompt)."""
    print("\n" + "="*70)
    print("DEMO 4: Multilingual Captioning (via prompting)")
    print("="*70)

    blip = MockBLIPDemo()

    image = "food_plate.jpg"

    # English
    print("🇬🇧 English:")
    blip.generate_caption(image, conditional_text="An image of")

    # Spanish (via conditional)
    print("🇪🇸 Spanish:")
    blip.generate_caption(image, conditional_text="A photograph of")

    print("💡 Use conditional text to guide the language")


# ============================================================================
# ADVANCED: Quality Control
# ============================================================================

def demo_quality_metrics():
    """Caption quality metrics."""
    print("\n" + "="*70)
    print("DEMO 5: Caption Quality Metrics")
    print("="*70)

    # Ground truth (human caption)
    ground_truth = "a golden retriever playing fetch in a sunny park"

    # Generated caption
    generated = "a dog playing with a ball in a park"

    print(f"Ground truth: {ground_truth}")
    print(f"Generated:    {generated}\n")

    # Metrics (simplified)
    def calculate_bleu_1(reference, candidate):
        """BLEU-1 simplified (unigram overlap)."""
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        overlap = len(ref_words & cand_words)
        return overlap / len(cand_words) if cand_words else 0

    bleu_1 = calculate_bleu_1(ground_truth, generated)

    print(f"📊 BLEU-1 Score: {bleu_1:.2f}")
    print(f"💡 BLEU-1 > 0.5 es acceptable para captioning")


if __name__ == "__main__":
    print("\n🎯 IMAGE CAPTIONING WITH BLIP")
    print("📝 Generates automatic image descriptions\n")

    demo_unconditional()
    demo_conditional()
    demo_comparison()
    demo_multilingual()
    demo_quality_metrics()

    print("\n" + "="*70)
    print("💡 USE CASES:")
    print("="*70)
    print("✅ Accessibility: Alt text for the visually impaired")
    print("✅ E-commerce: Auto-tag products")
    print("✅ Social Media: Suggest captions for posts")
    print("✅ Content Moderation: Detect inappropriate content")
    print("✅ Search: Index images by content")

    print("\n📚 Model: Salesforce/blip-image-captioning-base")
    print("📄 Paper: https://arxiv.org/abs/2201.12086")

    print("\n" + "="*70)
    print("REAL CODE (for production):")
    print("="*70)
    print(REAL_BLIP_CODE)
