"""
Text-to-Image with Stable Diffusion
====================================
Genera imágenes desde texto usando Stable Diffusion.
Modelo de difusión que crea imágenes de alta calidad.

Requirements:
    pip install diffusers transformers torch pillow accelerate
"""

# ============================================================================
# CONCEPTUAL DEMO
# ============================================================================

class MockStableDiffusionDemo:
    """
    Demo conceptual de Stable Diffusion text-to-image.
    """

    def __init__(self):
        print("🔄 Mock Stable Diffusion loaded (conceptual demo)")
        print("En producción: StableDiffusionPipeline.from_pretrained()\n")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512
    ) -> dict:
        """
        Genera imagen desde texto.

        Args:
            prompt: Descripción de la imagen deseada
            negative_prompt: Lo que NO quieres en la imagen
            num_inference_steps: Más pasos = mejor calidad (pero más lento)
            guidance_scale: Qué tan estricto seguir el prompt (7-15 típico)
            width, height: Dimensiones (múltiplos de 64)
        """
        print(f"🎨 Generando imagen...")
        print(f"📝 Prompt: {prompt}")

        if negative_prompt:
            print(f"🚫 Negative: {negative_prompt}")

        print(f"⚙️  Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        print(f"📐 Size: {width}x{height}")

        # En producción, aquí se ejecuta el diffusion process:
        # 1. Encode prompt → text embeddings
        # 2. Start from random noise
        # 3. Iteratively denoise (num_inference_steps veces)
        # 4. Decode latents → RGB image

        print(f"✅ Imagen generada (mock): output_{hash(prompt) % 1000}.png\n")

        return {
            "image_path": f"output_{hash(prompt) % 1000}.png",
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "size": (width, height)
            }
        }


# ============================================================================
# REAL STABLE DIFFUSION USAGE
# ============================================================================

REAL_SD_CODE = """
from diffusers import StableDiffusionPipeline
import torch

# 1. Load model (requiere ~5GB VRAM)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16  # Usar FP16 para ahorrar memoria
)
pipe = pipe.to("cuda")  # GPU (o "cpu" pero MUY lento)

# 2. Generate image
prompt = "A futuristic city at sunset with flying cars, cyberpunk style, highly detailed"
negative_prompt = "blurry, low quality, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,  # 20-100 típico (más = mejor calidad)
    guidance_scale=7.5,      # 7-15 típico (más = sigue más el prompt)
    width=512,
    height=512
).images[0]

# 3. Save image
image.save("output.png")
"""


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def demo_basic_generation():
    """Generación básica."""
    print("="*70)
    print("DEMO 1: Basic Image Generation")
    print("="*70 + "\n")

    sd = MockStableDiffusionDemo()

    prompts = [
        "A serene mountain landscape at sunrise, photorealistic",
        "A cute robot playing with a cat, digital art",
        "An astronaut riding a horse on Mars, cinematic lighting",
    ]

    for prompt in prompts:
        sd.generate_image(prompt)


def demo_negative_prompts():
    """Usando negative prompts."""
    print("="*70)
    print("DEMO 2: Negative Prompts")
    print("="*70 + "\n")

    sd = MockStableDiffusionDemo()

    prompt = "A portrait of a woman, professional photography"

    # Sin negative prompt
    print("🔵 Sin negative prompt:")
    sd.generate_image(prompt)

    # Con negative prompt
    print("🟢 Con negative prompt:")
    sd.generate_image(
        prompt=prompt,
        negative_prompt="blurry, distorted, ugly, low quality, cartoon"
    )

    print("💡 Negative prompts mejoran la calidad evitando defectos comunes")


def demo_style_variations():
    """Diferentes estilos artísticos."""
    print("\n" + "="*70)
    print("DEMO 3: Artistic Styles")
    print("="*70 + "\n")

    sd = MockStableDiffusionDemo()

    base = "A castle on a hill"

    styles = [
        (f"{base}, photorealistic, 4K", "Photorealistic"),
        (f"{base}, oil painting style", "Oil Painting"),
        (f"{base}, anime style, Studio Ghibli", "Anime"),
        (f"{base}, cyberpunk neon style", "Cyberpunk"),
        (f"{base}, watercolor painting", "Watercolor"),
    ]

    for prompt, style_name in styles:
        print(f"🎨 Style: {style_name}")
        sd.generate_image(prompt)


def demo_parameter_effects():
    """Efecto de parámetros."""
    print("="*70)
    print("DEMO 4: Parameter Effects")
    print("="*70 + "\n")

    sd = MockStableDiffusionDemo()

    prompt = "A magical forest with glowing mushrooms"

    # Guidance scale variations
    print("📊 Guidance Scale Effect:")
    print("  Low (5.0):  Más creativo, menos adherencia al prompt")
    sd.generate_image(prompt, guidance_scale=5.0)

    print("  Medium (7.5): Balance")
    sd.generate_image(prompt, guidance_scale=7.5)

    print("  High (15.0): Muy estricto al prompt, puede sobre-saturar")
    sd.generate_image(prompt, guidance_scale=15.0)

    # Steps variations
    print("\n🔢 Inference Steps Effect:")
    print("  20 steps: Rápido pero menor calidad")
    sd.generate_image(prompt, num_inference_steps=20)

    print("  50 steps: Balance (recomendado)")
    sd.generate_image(prompt, num_inference_steps=50)

    print("  100 steps: Mejor calidad pero 2x más lento")
    sd.generate_image(prompt, num_inference_steps=100)


def demo_aspect_ratios():
    """Diferentes aspect ratios."""
    print("\n" + "="*70)
    print("DEMO 5: Aspect Ratios")
    print("="*70 + "\n")

    sd = MockStableDiffusionDemo()

    prompt = "A panoramic view of a medieval city"

    ratios = [
        (512, 512, "Square (1:1)"),
        (768, 512, "Landscape (3:2)"),
        (512, 768, "Portrait (2:3)"),
        (896, 512, "Widescreen (16:9)"),
    ]

    for width, height, name in ratios:
        print(f"📐 {name}:")
        sd.generate_image(prompt, width=width, height=height)


def demo_prompting_techniques():
    """Técnicas de prompting."""
    print("="*70)
    print("DEMO 6: Advanced Prompting Techniques")
    print("="*70 + "\n")

    print("💡 PROMPTING BEST PRACTICES:\n")

    print("1️⃣ Especificidad:")
    print("  ❌ Bad:  'a dog'")
    print("  ✅ Good: 'a golden retriever puppy playing in autumn leaves'\n")

    print("2️⃣ Calidad descriptors:")
    print("  ✅ highly detailed, 4K, professional photography")
    print("  ✅ trending on artstation, award winning")
    print("  ✅ cinematic lighting, dramatic composition\n")

    print("3️⃣ Estilo artístico:")
    print("  ✅ by Greg Rutkowski (digital art)")
    print("  ✅ Studio Ghibli style (anime)")
    print("  ✅ in the style of Monet (impressionism)\n")

    print("4️⃣ Iluminación:")
    print("  ✅ golden hour lighting")
    print("  ✅ volumetric fog")
    print("  ✅ rim lighting, backlit\n")

    print("5️⃣ Composición:")
    print("  ✅ centered, symmetrical")
    print("  ✅ rule of thirds")
    print("  ✅ wide angle, close-up\n")


def demo_diffusion_process():
    """Explicar el proceso de difusión."""
    print("="*70)
    print("DEMO 7: How Diffusion Works")
    print("="*70 + "\n")

    print("🔬 DIFFUSION PROCESS:\n")
    print("Step 1: Start with random noise")
    print("  [████████████████] 100% noise\n")

    print("Step 2-10: Iteratively denoise")
    print("  [████████░░░░░░░░] Some shapes emerging...")
    print("  [██████░░░░░░░░░░] More structure...")
    print("  [████░░░░░░░░░░░░] Colors appearing...\n")

    print("Step 45-50: Final refinement")
    print("  [██░░░░░░░░░░░░░░] Details sharpening...")
    print("  [░░░░░░░░░░░░░░░░] Clean image!\n")

    print("💡 Cada step predice y remueve un poco de 'ruido'")
    print("💡 Guidance scale controla qué tan estricto sigue el prompt")


if __name__ == "__main__":
    print("\n🎯 TEXT-TO-IMAGE WITH STABLE DIFFUSION")
    print("🖼️  Genera imágenes desde descripciones de texto\n")

    demo_basic_generation()
    demo_negative_prompts()
    demo_style_variations()
    demo_parameter_effects()
    demo_aspect_ratios()
    demo_prompting_techniques()
    demo_diffusion_process()

    print("\n" + "="*70)
    print("💡 USE CASES:")
    print("="*70)
    print("✅ Content Creation: Arte para blogs, marketing")
    print("✅ Game Development: Concept art, textures")
    print("✅ Product Design: Visualizar ideas rápidamente")
    print("✅ Education: Ilustrar conceptos")
    print("✅ Entertainment: Crear personajes, escenas")

    print("\n⚠️  CONSIDERACIONES:")
    print("  • Requiere GPU potente (>6GB VRAM)")
    print("  • Copyright: Modelo entrenado con imágenes de internet")
    print("  • Content policy: No generar contenido ilegal/dañino")
    print("  • Watermarking: Algunas versiones añaden watermark")

    print("\n📚 Modelos:")
    print("  • stabilityai/stable-diffusion-2-1 (latest open)")
    print("  • runwayml/stable-diffusion-v1-5 (popular)")
    print("  • stabilityai/stable-diffusion-xl-base-1.0 (SDXL, mejor calidad)")

    print("\n📄 Paper: https://arxiv.org/abs/2112.10752")

    print("\n" + "="*70)
    print("CÓDIGO REAL (para producción):")
    print("="*70)
    print(REAL_SD_CODE)
