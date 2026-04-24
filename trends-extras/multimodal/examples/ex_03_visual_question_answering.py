"""
Visual Question Answering (VQA)
================================
Answer questions about images using BLIP-VQA.
Combines vision + reasoning to understand scenes.

Requirements:
    pip install transformers torch pillow
"""

from PIL import Image

# ============================================================================
# CONCEPTUAL DEMO
# ============================================================================

class MockVQADemo:
    """
    Demo conceptual de Visual Question Answering.
    """

    def __init__(self):
        print("🔄 Mock BLIP-VQA model loaded (conceptual demo)\n")

        # Simulated knowledge base (in production, the model "sees" the image)
        self.image_knowledge = {
            "dog_park.jpg": {
                "objects": ["dog", "ball", "grass", "trees"],
                "colors": {"dog": "golden", "ball": "red"},
                "actions": ["playing", "running"],
                "count": {"dogs": 1, "balls": 1},
                "scene": "park",
                "weather": "sunny"
            },
            "city_street.jpg": {
                "objects": ["car", "building", "people", "traffic light"],
                "colors": {"car": "blue", "building": "gray"},
                "count": {"cars": 3, "people": 5},
                "scene": "urban street",
                "weather": "cloudy"
            }
        }

    def answer_question(self, image_path: str, question: str) -> str:
        """
        Responde pregunta sobre la imagen.
        """
        print(f"📸 Imagen: {image_path}")
        print(f"❓ Pregunta: {question}")

        # Get image knowledge
        knowledge = self.image_knowledge.get(
            image_path,
            {"objects": ["unknown"], "colors": {}, "count": {}}
        )

        # Simple rule-based answering (in production this is the VQA model)
        question_lower = question.lower()

        if "what color" in question_lower:
            for obj, color in knowledge["colors"].items():
                if obj in question_lower:
                    answer = color
                    break
            else:
                answer = "unknown"

        elif "how many" in question_lower:
            for obj, count in knowledge["count"].items():
                if obj in question_lower:
                    answer = str(count)
                    break
            else:
                answer = "unknown"

        elif "what is" in question_lower or "what are" in question_lower:
            if knowledge["objects"]:
                answer = ", ".join(knowledge["objects"][:3])
            else:
                answer = "unknown"

        elif "where" in question_lower:
            answer = knowledge.get("scene", "unknown location")

        elif "weather" in question_lower:
            answer = knowledge.get("weather", "unknown")

        elif "action" in question_lower or "doing" in question_lower:
            answer = ", ".join(knowledge.get("actions", ["nothing"]))

        else:
            answer = "I cannot determine that from the image"

        print(f"✅ Respuesta: {answer}\n")
        return answer


# ============================================================================
# REAL BLIP-VQA USAGE
# ============================================================================

REAL_VQA_CODE = """
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# 1. Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# 2. Load image
image = Image.open("photo.jpg")

# 3. Ask question
question = "What color is the dog?"

# 4. Process inputs
inputs = processor(image, question, return_tensors="pt")

# 5. Generate answer
outputs = model.generate(**inputs, max_length=20)
answer = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Q: {question}")
print(f"A: {answer}")
# Output: "golden"
"""


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_object_questions():
    """Preguntas sobre objetos."""
    print("="*70)
    print("DEMO 1: Object Recognition Questions")
    print("="*70 + "\n")

    vqa = MockVQADemo()

    qa_pairs = [
        ("dog_park.jpg", "What is in the image?"),
        ("dog_park.jpg", "What color is the dog?"),
        ("city_street.jpg", "What objects are visible?"),
    ]

    for image, question in qa_pairs:
        vqa.answer_question(image, question)


def demo_counting_questions():
    """Preguntas de conteo."""
    print("="*70)
    print("DEMO 2: Counting Questions")
    print("="*70 + "\n")

    vqa = MockVQADemo()

    qa_pairs = [
        ("dog_park.jpg", "How many dogs are in the image?"),
        ("city_street.jpg", "How many cars are visible?"),
        ("city_street.jpg", "How many people are there?"),
    ]

    for image, question in qa_pairs:
        vqa.answer_question(image, question)


def demo_reasoning_questions():
    """Preguntas que requieren razonamiento."""
    print("="*70)
    print("DEMO 3: Reasoning Questions")
    print("="*70 + "\n")

    vqa = MockVQADemo()

    qa_pairs = [
        ("dog_park.jpg", "What is the dog doing?"),
        ("dog_park.jpg", "What is the weather like?"),
        ("city_street.jpg", "Where was this photo taken?"),
    ]

    for image, question in qa_pairs:
        vqa.answer_question(image, question)


def demo_multi_question_conversation():
    """Conversation with multiple questions."""
    print("="*70)
    print("DEMO 4: Multi-Question Conversation")
    print("="*70 + "\n")

    vqa = MockVQADemo()

    image = "dog_park.jpg"

    print(f"🖼️  Analizando: {image}\n")

    conversation = [
        "What is in the image?",
        "What color is the dog?",
        "How many dogs are there?",
        "What is the dog doing?",
        "What is the weather?",
    ]

    for i, question in enumerate(conversation, 1):
        print(f"Turn {i}:")
        vqa.answer_question(image, question)


def demo_yes_no_questions():
    """Yes/no questions."""
    print("="*70)
    print("DEMO 5: Yes/No Questions")
    print("="*70 + "\n")

    # In production, VQA can answer yes/no
    print("📸 Imagen: dog_park.jpg\n")

    questions_and_answers = [
        ("Is there a dog in the image?", "yes"),
        ("Is the dog indoors?", "no"),
        ("Is it daytime?", "yes"),
        ("Are there multiple dogs?", "no"),
    ]

    for question, answer in questions_and_answers:
        print(f"❓ {question}")
        print(f"✅ {answer}\n")


# ============================================================================
# ADVANCED: VQA Pipeline
# ============================================================================

def demo_vqa_pipeline():
    """Pipeline completo de VQA."""
    print("="*70)
    print("DEMO 6: Complete VQA Pipeline")
    print("="*70 + "\n")

    print("🔧 VQA Pipeline Steps:")
    print("  1. Image Encoding → Visual features (CNN/ViT)")
    print("  2. Question Encoding → Text features (BERT)")
    print("  3. Cross-Modal Fusion → Attention mechanism")
    print("  4. Answer Generation → Decoder\n")

    print("📊 Architecture:")
    print("""
    ┌─────────────┐       ┌─────────────┐
    │   Image     │       │  Question   │
    │  (RGB)      │       │   (Text)    │
    └──────┬──────┘       └──────┬──────┘
           │                     │
           ▼                     ▼
    ┌─────────────┐       ┌─────────────┐
    │   ViT       │       │   BERT      │
    │  Encoder    │       │  Encoder    │
    └──────┬──────┘       └──────┬──────┘
           │                     │
           └──────────┬──────────┘
                      │
                      ▼
           ┌──────────────────┐
           │  Cross-Attention │
           │     Fusion       │
           └─────────┬────────┘
                     │
                     ▼
           ┌──────────────────┐
           │     Answer       │
           │    Decoder       │
           └──────────────────┘
                     │
                     ▼
                  Answer
    """)


if __name__ == "__main__":
    print("\n🎯 VISUAL QUESTION ANSWERING (VQA)")
    print("🤔 Answers questions about images\n")

    demo_object_questions()
    demo_counting_questions()
    demo_reasoning_questions()
    demo_multi_question_conversation()
    demo_yes_no_questions()
    demo_vqa_pipeline()

    print("\n" + "="*70)
    print("💡 USE CASES:")
    print("="*70)
    print("✅ Accessibility: Answer questions about images")
    print("✅ E-commerce: 'Is this shirt blue?' sin tags manuales")
    print("✅ Content Moderation: 'Is there violence in this image?'")
    print("✅ Education: Interactive visual learning")
    print("✅ Healthcare: 'How many tumors are visible?'")
    print("✅ Robotics: Visual understanding para robots")

    print("\n📚 Modelo: Salesforce/blip-vqa-base")
    print("📄 Paper: https://arxiv.org/abs/2201.12086")
    print("📊 Dataset: VQAv2 (>1M questions on 200K images)")

    print("\n" + "="*70)
    print("REAL CODE (for production):")
    print("="*70)
    print(REAL_VQA_CODE)
