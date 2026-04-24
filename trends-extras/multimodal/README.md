# Multimodal AI — Vision + Language Models

## 🎯 Objetivo

Trabajar con modelos multimodales que combinan visión y lenguaje (CLIP, BLIP, GPT-4V, LLaVA): image captioning, visual question answering, text-to-image.

## 💡 Qué aprenderás

- Vision-Language Models (VLMs): arquitectura y training
- CLIP para image-text similarity y zero-shot classification
- BLIP para image captioning y VQA
- GPT-4V y Gemini para visual understanding
- Text-to-image: Stable Diffusion, DALL-E, Midjourney
- Multi-modal embeddings y retrieval
- Fusion techniques (early vs late fusion)

## 📂 Contenido

### Examples

- **ex_01_clip_zero_shot_classification.py**: Clasificación de imágenes sin training con CLIP
- **ex_02_image_captioning_blip.py**: Generar captions descriptivos con BLIP
- **ex_03_visual_question_answering.py**: Responder preguntas sobre imágenes
- **ex_04_text_to_image_stable_diffusion.py**: Generar imágenes desde prompts

## 🔑 Conceptos Clave

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

### Aplicaciones

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

**Training**: Pares (image, caption) con contrastive loss

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
# Sin training adicional
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
| **LLaVA**            | Visual chat         | ✅          | -   | Open source VLM          |
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

## 🧪 Ejercicio Rápido

1. **Setup**: `pip install transformers pillow torch`
1. **CLIP Zero-Shot**: Clasifica imagen entre 5 categorías sin training
1. **Image Captioning**: Genera caption para tus fotos
1. **VQA**: Haz 3 preguntas sobre una imagen
1. **Compara**: Precisión de CLIP vs BLIP en tus ejemplos

## 📚 Recursos Curados

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

## ✅ Checklist de Aprendizaje

- [ ] CLIP para zero-shot classification
- [ ] Image captioning con BLIP
- [ ] Visual question answering
- [ ] Multi-modal embeddings para retrieval
- [ ] Text-to-image con Stable Diffusion
- [ ] Fine-tuning VLM en custom dataset
- [ ] Evaluation metrics (BLEU, CIDEr para captioning)

## 🎯 Impacto Real

- **E-commerce**: Search por imagen + texto, auto-tagging
- **Accessibility**: Auto-captioning para personas con discapacidad visual
- **Content Moderation**: Detectar contenido inapropiado con VQA
- **Creative Tools**: Text-to-image para diseño gráfico
- **Healthcare**: Medical image analysis con VLMs

## 🚀 Próximos Pasos

Integra con:

- **agents** para crear agentes visuales (analyse image → reason → act)
- **guardrails** para filtrar imágenes generadas inapropiadas
- **llm-evals** para evaluar calidad de captions/VQA responses

## Module objective

Pendiente de completar este apartado.

## What you will achieve

Pendiente de completar este apartado.

## Internal structure

Pendiente de completar este apartado.

## Level path (L1-L4)

Pendiente de completar este apartado.

## Recommended plan (by progress, not by weeks)

Pendiente de completar este apartado.

## Module completion criteria

Pendiente de completar este apartado.
