# Multimodal AI — Vision + Language Models

## 🎯 Objective

Work with multimodal Models that combine vision and language (CLIP, BLIP, GPT-4V, LLaVA): image captioning, visual question answering, text-to-image.

## 💡 What will you learn

- Vision-Language Models (VLMs): architecture and training
- CLIP for image-text similarity and zero-shot classification
- BLIP for image captioning and VQA
- GPT-4V and Gemini for visual understanding
- Text-to-image: Stable Diffusion, DALL-E, Midjourney
- Multi-modal embeddings and retrieval
- Fusion techniques (early vs late fusion)

## 📂 Content

### Examples

- **ex_01_clip_zero_shot_classification.py**: Classification of Images without training with CLIP
- **ex_02_image_captioning_blip.py**: Generate captions descriptions with BLIP
- **ex_03_visual_question_answering.py**: Responder questions about Images
- **ex_04_text_to_image_stable_diffusion.py**: Generate Images from prompts

## 🔑 Concepts Clave

### Vision-Language Models Architecture

```
┌─────────────┐         ┌─────────────┐
│  Image      │         │   Text      │
│  Encoder    │         │   Encoder   │
│  (ViT)      │         │   (BERT)    │
└──────┬──────┘         └──────┬──────┘
       │                       │
       └───────────┬───────────┘
                   │
            ┌──────▼──────┐
            │  Joint      │
            │  Embedding  │
            │  Space      │
            └─────────────┘
```

### Applications

**Image → Text:**

- Image captioning
- Visual question answering (VQA)
- Visual dialog
- Scene understanding

**Text → Image:**

- Text-to-image generation (Stable Diffusion)
- Image editing from instructions
- Style transfer with text prompts

**Image ↔ Text:**

- Image-text retrieval
- Zero-shot image classification
- Visual grounding (localize objects from text)

## 🎨 CLIP: Contrastive Learning

**Training**: Pairs (image, caption) with contrastive loss

```python
# Simplified CLIP concept
image_features = image_encoder(image)    # [batch, 512]
text_features = text_encoder(text)       # [batch, 512]

# Similarity matrix
similarity = image_features @ text_features.T  # [batch, batch]

# Maximize diagonal (correct pairs), minimize off-diagonal
```

**Zero-Shot Classification**:

```python
# without additional training
prompts = ["a photo of a cat", "a photo of a dog"]
text_features = clip.encode_text(prompts)
image_features = clip.encode_image(image)

# Classify por similarity
similarities = image_features @ text_features.T
predicted_class = similarities.argmax()
```

## 📊 Models Comparison

| Model                | Task                | Open Source | API | Best For                 |
| -------------------- | ------------------- | ----------- | --- | ------------------------ |
| **CLIP**             | Image-text matching | ✅          | -   | Zero-shot classification |
| **BLIP**             | Captioning, VQA     | ✅          | -   | Understanding images     |
| **KEY**              | Visual chat         | ✅          | -   | Open source VLM          |
| **GPT-4V**           | Multi-modal chat    | ❌          | ✅  | Production apps          |
| **Stable Diffusion** | Text-to-image       | ✅          | ✅  | Image generation         |
| **DALL-E 3**         | Text-to-image       | ❌          | ✅  | High quality images      |

## 💻 Code Pattern: VQA Pipeline

```python
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Process
image = Image.open("photo.jpg")
question = "What color is the car?"

inputs = processor(image, question, return_tensors="pt")
outputs = model.generate(**inputs)
answer = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Q: {question}")
print(f"A: {answer}")
```

## 🧪 Quick Exercise

1. **Setup**: `pip install transformers pillow torch`
1. **CLIP Zero-Shot**: Classify Image between 5 categories without training
1. **Image Captioning**: Generate captions for your photos
1. **VQA**: Ask 3 questions about an Image
1. **Compare**: Precision of CLIP vs BLIP in your Examples

## 📚 Health Resources

**Models & Papers:**

- [CLIP (OpenAI)](https://arxiv.org/abs/2103.00020)
- [BLIP (Salesforce)](https://arxiv.org/abs/2201.12086)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [Stable Diffusion](https://stability.ai/stable-diffusion)

**Frameworks:**

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/clip)
- [Diffusers Library](https://github.com/huggingface/diffusers)

**Datasets:**

- [COCO Captions](https://cocodataset.org/)
- [Visual Genome](https://visualgenome.org/)
- [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/)

**Tutorials:**

- [CLIP Tutorial](https://github.com/openai/CLIP)
- [Stable Diffusion Guide](https://stable-diffusion-art.com/)

## ✅ Learning Checklist

- [ ] CLIP for zero-shot classification
- [ ] Image captioning with BLIP
- [ ] Visual question answering
- [ ] Multi-modal embeddings for retrieval
- [ ] Text-to-image with Stable Diffusion
- [ ] Fine-tuning VLM in custom dataset
- [ ] Evaluation metrics (BLEU, CIDEr for captioning)

## 🎯 Real Impact

- **E-commerce**: Search by Image + text, auto-tagging
- **Accessibility**: Auto-captioning for people with visual impairments
- **Content Moderation**: Detect Inappropriate Content with VQA
- **Creative Tools**: Text-to-image for graphic design
- **Healthcare**: Medical image analysis with VLMs

## 🚀 Next Steps

Integrate with:

- **agents** to create visual agents (analyze image → reason → act)
- **guardrails** for filter Images generated inappropriately
- **llm-evals** to assess quality of captions/VQA responses

## Module objective

Build practical multimodal workflows that combine vision and language models for classification, captioning, question answering, and generation tasks.

## What you will achieve

- Run core multimodal pipelines end-to-end.
- Select appropriate models for image-text use cases.
- Evaluate response quality and failure patterns.
- Adapt examples to product-oriented scenarios.

## Internal structure

- `README.md`: concepts, trade-offs, and implementation guidance.
- `examples/`: CLIP, BLIP, VQA, and text-to-image demonstrations.
- `practices/`: guided multimodal exercises and performance analysis.

## Level path (L1-L4)

- L1: Execute baseline image-text tasks.
- L2: Tune prompts and preprocessing strategies.
- L3: Compare model outputs across real scenarios.
- L4: Design a robust multimodal mini-solution with evaluation.

## Recommended plan (by progress, not by weeks)

Start with deterministic tasks (classification/captioning), then add open-ended tasks (VQA/generation). Once outputs are stable, add evaluation and guardrails.

## Module completion criteria

- You can deploy at least two multimodal pipelines.
- You can explain model selection and observed trade-offs.
- You can provide a short quality review with representative examples.
