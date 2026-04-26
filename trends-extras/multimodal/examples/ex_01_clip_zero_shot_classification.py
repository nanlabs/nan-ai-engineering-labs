"""
CLIP Zero-Shot Classification
==============================
Classify images without training using OpenAI CLIP.
CLIP aprende relaciones imagen-text de 400M pares.

Requirements:
    pip install transformers torch pillow
"""

from PIL import Image
import torch

# Note: In production uncomment real imports
# from transformers import CLIPProcessor, CLIPModel

# ============================================================================
# CONCEPTUAL DEMO (sin requerir GPU)
# ============================================================================

class MockCLIPDemo:
    """
    Demo conceptual de CLIP zero-shot classification.
    Shows the logic without requiring a real model.
    """

    def __init__(self):
        print("🔄 Mock CLIP model loaded (conceptual demo)")
        print("Para usage real: descomentar imports y usar model real\n")

    def classify_image(self, image_path: str, candidate_labels: list) -> dict:
        """
        Clasifica image contra labels sin training.
        """
        print(f"📸 Image: {image_path}")
        print(f"🏷️  Labels: {candidate_labels}")

        # En el model real:
        # 1. image_features = clip.encode_image(image) -> [512]
        # 2. text_features = clip.encode_text(prompts) -> [n_labels, 512]
        # 3. similarity = image_features @ text_features.T -> [n_labels]
        # 4. probs = softmax(similarity * temperature)

        # Mock probabilities (in production these are calculated by CLIP)
        import random
        random.seed(42)

        probs = [random.random() for _ in candidate_labels]
        total = sum(probs)
        probs = [p/total for p in probs]  # Normalize

        # Sort by probability
        results = sorted(
            zip(candidate_labels, probs),
            key=lambda x: x[1],
            reverse=True
        )

        print("\n📊 Results:")
        for label, prob in results:
            bar = "█" * int(prob * 50)
            print(f"  {label:20s} {prob:6.2%} {bar}")

        return {
            "predicted_label": results[0][0],
            "confidence": results[0][1],
            "all_scores": dict(results)
        }


# ============================================================================
# REAL CLIP USAGE (Commented)
# ============================================================================

REAL_CLIP_CODE = """
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1. Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Load image
image = Image.open("photo.jpg")

# 3. Define candidate labels
candidate_labels = ["a cat", "a dog", "a car", "a tree", "a building"]

# 4. Create prompts
prompts = [f"a photo of {label}" for label in candidate_labels]

# 5. Process inputs
inputs = processor(
    text=prompts,
    images=image,
    return_tensors="pt",
    padding=True
)

# 6. Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # [1, n_labels]
    probs = logits_per_image.softmax(dim=1)[0]  # [n_labels]

# 7. Get results
results = {label: prob.item() for label, prob in zip(candidate_labels, probs)}
predicted = max(results, key=results.get)

print(f"Predicted: {predicted} ({results[predicted]:.2%})")
"""


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_animals():
    """Clasificar animales."""
    print("="*70)
    print("DEMO 1: Animal Classification")
    print("="*70)

    clip = MockCLIPDemo()

    result = clip.classify_image(
        image_path="cat_photo.jpg",
        candidate_labels=["cat", "dog", "bird", "fish", "hamster"]
    )

    print(f"\n✅ Prediction: {result['predicted_label']} ({result['confidence']:.2%})")


def demo_vehicles():
    """Classify vehicles."""
    print("\n" + "="*70)
    print("DEMO 2: Vehicle Classification")
    print("="*70)

    clip = MockCLIPDemo()

    result = clip.classify_image(
        image_path="vehicle.jpg",
        candidate_labels=["car", "truck", "motorcycle", "bicycle", "airplane"]
    )

    print(f"\n✅ Prediction: {result['predicted_label']} ({result['confidence']:.2%})")


def demo_scenes():
    """Clasificar escenas."""
    print("\n" + "="*70)
    print("DEMO 3: Scene Classification")
    print("="*70)

    clip = MockCLIPDemo()

    result = clip.classify_image(
        image_path="scene.jpg",
        candidate_labels=[
            "beach sunset",
            "mountain landscape",
            "city street",
            "forest trail",
            "desert"
        ]
    )

    print(f"\n✅ Prediction: {result['predicted_label']} ({result['confidence']:.2%})")


def demo_emotions():
    """Clasificar emociones en rostros."""
    print("\n" + "="*70)
    print("DEMO 4: Emotion Classification (Zero-Shot!)")
    print("="*70)

    clip = MockCLIPDemo()

    result = clip.classify_image(
        image_path="face.jpg",
        candidate_labels=[
            "happy person",
            "sad person",
            "angry person",
            "surprised person",
            "neutral expression"
        ]
    )

    print(f"\n✅ Prediction: {result['predicted_label']} ({result['confidence']:.2%})")


if __name__ == "__main__":
    print("\n🎯 CLIP ZERO-SHOT IMAGE CLASSIFICATION")
    print("🔬 Sin training, solo con descripciones de text!\n")

    demo_animals()
    demo_vehicles()
    demo_scenes()
    demo_emotions()

    print("\n" + "="*70)
    print("💡 VENTAJAS DE CLIP ZERO-SHOT:")
    print("="*70)
    print("✅ No require dataset de training")
    print("✅ Funciona con cualquier label (incluso inventado)")
    print("✅ Multilingual (works in Spanish, English, etc.)")
    print("✅ Flexible: cambia labels sin reentrenar")
    print("\n📚 Model real: openai/clip-vit-base-patch32 (Hugging Face)")
    print("📄 Paper: https://arxiv.org/abs/2103.00020")

    print("\n" + "="*70)
    print("REAL CODE (for production):")
    print("="*70)
    print(REAL_CLIP_CODE)
