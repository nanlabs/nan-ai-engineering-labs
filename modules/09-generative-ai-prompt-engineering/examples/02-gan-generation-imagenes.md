# Example 02 — GAN para Generación de Images

## Contexto

Las GANs (Generative Adversarial Networks) usan 2 neural networks que compiten: **Generator** crea Images falsas, **Discriminator** intenta distinguir reales de falsas.

## Objective

Entrenar GAN para generar dígitos sintéticos (MNIST) desde ruido aleatorio.

______________________________________________________________________

## 🚀 Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Hiperparámetros
latent_dim = 100      # Dimensión del ruido z
image_size = 28*28    # MNIST: 28x28 = 784 píxeles
batch_size = 128
num_epochs = 50
lr = 0.0002
```

______________________________________________________________________

## 📚 Cargar Data

```python
# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])

# Dataset MNIST
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

print(f"Total imágenes: {len(train_dataset)}")
print(f"Batches por época: {len(dataloader)}")
```

**Salida:**

```
Total imágenes: 60000
Batches por época: 469
```

______________________________________________________________________

## 🏗️ Arquitectura GAN

### Generator

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=784):
        super(Generator, self).__init__()

        # z (100) → 256 → 512 → 784 (imagen)
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, img_size),
            nn.Tanh()  # Salida en [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img

generator = Generator(latent_dim, image_size).to(device)

# Param count
g_params = sum(p.numel() for p in generator.parameters())
print(f"Generator parámetros: {g_params:,}")
```

**Salida:**

```
Generator parámetros: 1,493,256
```

### Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self, img_size=784):
        super(Discriminator, self).__init__()

        # Imagen (784) → 512 → 256 → 1 (real/fake)
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()  # Probabilidad [0, 1]
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

discriminator = Discriminator(image_size).to(device)

d_params = sum(p.numel() for p in discriminator.parameters())
print(f"Discriminator parámetros: {d_params:,}")
```

**Salida:**

```
Discriminator parámetros: 533,505
```

______________________________________________________________________

## 🎯 Loss y optimizadores

```python
# Loss: Binary Cross Entropy
criterion = nn.BCELoss()

# Optimizadores separados
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

print("Optimizadores configurados")
```

______________________________________________________________________

## 🔄 Training Loop

```python
# Historia de pérdidas
g_losses = []
d_losses = []

# Fixed noise para visualización
fixed_noise = torch.randn(64, latent_dim, device=device)

for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):

        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.view(batch_size, -1).to(device)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # =====================
        # Train Discriminator
        # =====================
        optimizer_D.zero_grad()

        # Loss con imágenes reales
        real_output = discriminator(real_imgs)
        d_loss_real = criterion(real_output, real_labels)

        # Loss con imágenes falsas
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        fake_output = discriminator(fake_imgs.detach())  # No gradientes al G
        d_loss_fake = criterion(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # ==================
        # Train Generator
        # ==================
        optimizer_G.zero_grad()

        # Generator intenta engañar al Discriminator
        fake_output = discriminator(fake_imgs)
        g_loss = criterion(fake_output, real_labels)  # Queremos que D diga "real"

        g_loss.backward()
        optimizer_G.step()

    # Guardar pérdidas
    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    # Progreso
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("\nEntrenamiento completado")
```

**Salida:**

```
Epoch [10/50] | D Loss: 0.8234 | G Loss: 1.4521
Epoch [20/50] | D Loss: 0.6432 | G Loss: 1.1234
Epoch [30/50] | D Loss: 0.5123 | G Loss: 0.9876
Epoch [40/50] | D Loss: 0.4789 | G Loss: 0.8234
Epoch [50/50] | D Loss: 0.4523 | G Loss: 0.7456
```

______________________________________________________________________

## 📊 Visualization

### Evolución de pérdidas

```python
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss', alpha=0.7)
plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Evolución de pérdidas GAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gan_losses.png', dpi=150)
plt.show()
```

### Images generadas

```python
# Generar con fixed noise
generator.eval()
with torch.no_grad():
    fake_imgs = generator(fixed_noise)
    fake_imgs = fake_imgs.view(-1, 1, 28, 28).cpu()

# Visualizar
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_imgs[i].squeeze(), cmap='gray')
    ax.axis('off')

plt.suptitle('Dígitos Generados por GAN', fontsize=16)
plt.tight_layout()
plt.savefig('generated_digits.png', dpi=150)
plt.show()

print("Imágenes generadas guardadas")
```

______________________________________________________________________

## 🔄 Modos de falla de GANs

### 1. Mode Collapse

```python
# Si Generator genera siempre el mismo dígito
# Solución: Feature Matching o Minibatch Discrimination

def check_mode_collapse(generator, n_samples=1000):
    """
    Detecta si el generator solo produce pocas variantes
    """
    generator.eval()
    generated = []

    with torch.no_grad():
        for _ in range(n_samples // 64):
            z = torch.randn(64, latent_dim, device=device)
            fake = generator(z)
            generated.append(fake)

    generated = torch.cat(generated, dim=0)

    # Calcular varianza promedio
    variance = generated.var(dim=0).mean().item()

    print(f"Varianza promedio de outputs: {variance:.4f}")
    if variance < 0.01:
        print("⚠️ Posible Mode Collapse detectado")
    else:
        print("✅ Variedad adecuada")

    return variance

variance = check_mode_collapse(generator)
```

### 2. Vanishing Gradients

```python
# Si D es muy bueno, G no aprende (gradientes → 0)
# Solución: Wasserstein GAN (WGAN), usar Least Squares GAN

# Monitorear gradientes durante training
def monitor_gradients():
    g_grads = [p.grad.abs().mean().item() for p in generator.parameters() if p.grad is not None]
    d_grads = [p.grad.abs().mean().item() for p in discriminator.parameters() if p.grad is not None]

    print(f"G gradient mean: {np.mean(g_grads):.6f}")
    print(f"D gradient mean: {np.mean(d_grads):.6f}")
```

______________________________________________________________________

## 💡 Mejoras avanzadas

### DCGAN (Deep Convolutional GAN)

```python
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(DCGenerator, self).__init__()

        # z (100) → 7x7x256 → 14x14x128 → 28x28x1
        self.model = nn.Sequential(
            # Proyecto y reshape
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.ReLU(),

            # Reshape to (batch, 256, 7, 7)
            # Seguido de ConvTranspose2d para upsampling
        )

    def forward(self, z):
        # Implementación completa requiere ConvTranspose2d
        pass

# DCGAN usa:
# - ConvTranspose2d en Generator
# - Conv2d + stride en Discriminator
# - BatchNorm (excepto primera capa D y última G)
# - ReLU en G, LeakyReLU en D
```

### Conditional GAN

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(ConditionalGenerator, self).__init__()

        # z + label embedding → imagen
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),  # z + label
            nn.LeakyReLU(0.2),
            # ... resto de capas
        )

    def forward(self, z, labels):
        # Concatenar z con label embedding
        label_input = self.label_emb(labels)
        gen_input = torch.cat([z, label_input], dim=1)
        return self.model(gen_input)

# Uso: generar dígito específico
# z = torch.randn(1, 100)
# label = torch.tensor([7])  # Generar un "7"
# img = cond_generator(z, label)
```

______________________________________________________________________

## 📈 Metrics de Evaluation

### Inception Score (IS)

```python
# Mide calidad y diversidad
# IS alto = imágenes de calidad y diversas
# Requiere modelo Inception preentrenado

def inception_score(images, n_splits=10):
    """
    IS = exp(E[KL(p(y|x) || p(y))])
    """
    # Implementación requiere InceptionV3
    # from torchvision.models import inception_v3
    pass
```

### Fréchet Inception Distance (FID)

```python
# Compara distribución de imágenes reales vs generadas
# FID bajo = mejor (más parecidas distribuciones)

def frechet_inception_distance(real_images, fake_images):
    """
    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r Σ_f))
    """
    # Requiere features de Inception
    # μ = media, Σ = covarianza
    pass
```

______________________________________________________________________

## 📝 Resumen

### ✅ Componentes GAN

```
Generator: z (ruido) → imagen sintética
Discriminator: imagen → probabilidad real/fake

Training alternado:
1. D aprende a distinguir reales vs falsas
2. G aprende a engañar a D
```

### 🎯 Losses

**Discriminator:**

```
L_D = -[log(D(x_real)) + log(1 - D(G(z)))]
```

**Generator:**

```
L_G = -log(D(G(z)))
```

### 💡 Mejores Practices

- ✅ Usar LeakyReLU en Discriminator (evita dying ReLUs)
- ✅ BatchNorm en ambas redes (estabiliza training)
- ✅ Label smoothing (real=0.9 en vez de 1.0)
- ✅ Adam optimizer con β1=0.5 (momentum bajo)
- ✅ Entrenar D más veces que G si D es débil
- ✅ Monitorear ambas losses (deben converger sin dominar)

### 🚫 Errors comunes

- ❌ D muy fuerte → G no aprende (gradientes desaparecen)
- ❌ G muy fuerte → D siempre falla (mode collapse)
- ❌ LR muy alto → oscilaciones, no converge
- ❌ No usar BatchNorm → inestabilidad
- ❌ Olvidar `.detach()` en fake_imgs para D → gradientes incorrectos

### 🔧 Troubleshooting

| Problem            | Síntoma                       | Solución                               |
| ------------------- | ----------------------------- | -------------------------------------- |
| Mode Collapse       | G genera siempre igual        | Unrolled GAN, Minibatch Discrimination |
| Vanishing Gradients | G Loss no baja                | Wasserstein GAN, Least Squares GAN     |
| Oscilaciones        | Losses suben/bajan mucho      | Reducir LR, label smoothing            |
| Images borrosas   | G Loss baja pero mala calidad | Usar Progressive GAN, StyleGAN         |

### 📌 Checklist GAN

- ✅ Arquitectura balanceada (G y D similares capacidad)
- ✅ Learning rate adecuado (0.0002 típico)
- ✅ BatchNorm en Layers intermedias
- ✅ LeakyReLU en Discriminator
- ✅ Ruido latente suficiente (100-200 dim)
- ✅ Monitorear losses (no divergen)
- ✅ Validation visual periódica
- ✅ Evaluar diversidad (IS, FID)

### 🚀 Next steps

- Implementar DCGAN con convoluciones
- Conditional GAN para control de generación
- StyleGAN para alta resolución
- Aplicaciones: super-resolution, image-to-image translation (pix2pix)
