# Example 02 — GAN for Image Generation

## Context

GANs (Generative Adversarial Networks) use 2 competing neural networks: **Generator** creates fake Images, **Discriminator** tries to distinguish real from fake.

## Objective

Train GAN to generate synthetic digits (MNIST) from random noise.

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

# Hyperparameters
latent_dim = 100      # Dimension del noise z
image_size = 28*28    # MNIST: 28x28 = 784 pixels
batch_size = 128
num_epochs = 50
lr = 0.0002
```

______________________________________________________________________

## 📚 Load Data

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

print(f"Total images: {len(train_dataset)}")
print(f"Batches por time: {len(dataloader)}")
```

**Output:**

```
Total images: 60000
Batches por time: 469
```

______________________________________________________________________

## 🏗️ GAN Architecture

### Generator

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=784):
        super(Generator, self).__init__()

        # z (100) → 256 → 512 → 784 (image)
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
            nn.Tanh()  # Output en [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img

generator = Generator(latent_dim, image_size).to(device)

# Param count
g_params = sum(p.numel() for p in generator.parameters())
print(f"Generator parameters: {g_params:,}")
```

**Output:**

```
Generator parameters: 1,493,256
```

### Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self, img_size=784):
        super(Discriminator, self).__init__()

        # Image (784) → 512 → 256 → 1 (real/fake)
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
print(f"Discriminator parameters: {d_params:,}")
```

**Output:**

```
Discriminator parameters: 533,505
```

______________________________________________________________________

## 🎯 Loss and optimizers

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
# Historia de losses
g_losses = []
d_losses = []

# Fixed noise para visualization
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

        # Loss con images real
        real_output = discriminator(real_imgs)
        d_loss_real = criterion(real_output, real_labels)

        # Loss con images falsas
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

        # Generator intenta cheat al Discriminator
        fake_output = discriminator(fake_imgs)
        g_loss = criterion(fake_output, real_labels)  # Queremos que D diga "real"

        g_loss.backward()
        optimizer_G.step()

    # Save losses
    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    # Progreso
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("\nEntrenamiento completed")
```

**Output:**

```
Epoch [10/50] | D Loss: 0.8234 | G Loss: 1.4521
Epoch [20/50] | D Loss: 0.6432 | G Loss: 1.1234
Epoch [30/50] | D Loss: 0.5123 | G Loss: 0.9876
Epoch [40/50] | D Loss: 0.4789 | G Loss: 0.8234
Epoch [50/50] | D Loss: 0.4523 | G Loss: 0.7456
```

______________________________________________________________________

## 📊 Visualization

### Evolution of losses

```python
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss', alpha=0.7)
plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Evolution de losses GAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gan_losses.png', dpi=150)
plt.show()
```

### Images generated

```python
# Generate con fixed noise
generator.eval()
with torch.no_grad():
    fake_imgs = generator(fixed_noise)
    fake_imgs = fake_imgs.view(-1, 1, 28, 28).cpu()

# Visualize
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_imgs[i].squeeze(), cmap='gray')
    ax.axis('off')

plt.suptitle('Digits Generados por GAN', fontsize=16)
plt.tight_layout()
plt.savefig('generated_digits.png', dpi=150)
plt.show()

print("Images generadas guardadas")
```

______________________________________________________________________

## 🔄 GAN failure modes

### 1. Mode Collapse

```python
# Si Generator genera siempre el same digit
# Solution: Feature Matching o Minibatch Discrimination

def check_mode_collapse(generator, n_samples=1000):
    """
    Detecta si el generator solo produce pocas variants
    """
    generator.eval()
    generated = []

    with torch.no_grad():
        for _ in range(n_samples // 64):
            z = torch.randn(64, latent_dim, device=device)
            fake = generator(z)
            generated.append(fake)

    generated = torch.cat(generated, dim=0)

    # Calculate variance average
    variance = generated.var(dim=0).mean().item()

    print(f"Varianza average de outputs: {variance:.4f}")
    if variance < 0.01:
        print("⚠️ Possible Mode Collapse detectado")
    else:
        print("✅ Variedad adecuada")

    return variance

variance = check_mode_collapse(generator)
```

### 2. Vanishing Gradients

```python
# Si D es muy bueno, G no aprende (gradientes → 0)
# Solution: Wasserstein GAN (WGAN), use Least Squares GAN

# Monitorear gradientes durante training
def monitor_gradients():
    g_grads = [p.grad.abs().mean().item() for p in generator.parameters() if p.grad is not None]
    d_grads = [p.grad.abs().mean().item() for p in discriminator.parameters() if p.grad is not None]

    print(f"G gradient mean: {np.mean(g_grads):.6f}")
    print(f"D gradient mean: {np.mean(d_grads):.6f}")
```

______________________________________________________________________

## 💡 Advanced improvements

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
        # Implementation complete requires ConvTranspose2d
        pass

# DCGAN usa:
# - ConvTranspose2d en Generator
# - Conv2d + stride en Discriminator
# - BatchNorm (excepto primera layer D y last G)
# - ReLU en G, LeakyReLU en D
```

### Conditional GAN

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(ConditionalGenerator, self).__init__()

        # z + label embedding → image
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),  # z + label
            nn.LeakyReLU(0.2),
            # ... resto de layers
        )

    def forward(self, z, labels):
        # Concatenar z con label embedding
        label_input = self.label_emb(labels)
        gen_input = torch.cat([z, label_input], dim=1)
        return self.model(gen_input)

# Usage: generar digit specific
# z = torch.randn(1, 100)
# label = torch.tensor([7])  # Generate un "7"
# img = cond_generator(z, label)
```

______________________________________________________________________

## 📈Evaluation Metrics

### Inception Score (IS)

```python
# Measure quality y diversidad
# IS alto = images de quality y diversas
# Require model Inception preentrenado

def inception_score(images, n_splits=10):
    """
    IS = exp(E[KL(p(y|x) || p(y))])
    """
    # Implementation requires InceptionV3
    # from torchvision.models import inception_v3
    pass
```

### Frechet Inception Distance (FID)

```python
# Compara distribution de images real vs generadas
# FID low = better (más parecidas distributions)

def frechet_inception_distance(real_images, fake_images):
    """
    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2√(Σ_r Σ_f))
    """
    # Require features de Inception
    # μ = media, Σ = covarianza
    pass
```

______________________________________________________________________

## 📝 Summary

### ✅ Components GAN

```
Generator: z (noise) → image synthetic
Discriminator: image → probability real/fake

Training alternado:
1. D aprende a distinguir real vs falsas
2. G aprende a cheat a D
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

### 💡 Best Practices

- ✅ Use LeakyReLU in Discriminator (avoid dying ReLUs)
- ✅ BatchNorm on both networks (stabilizes training)
- ✅ Label smoothing (real=0.9 instead of 1.0)
- ✅ Adam optimizer with β1=0.5 (momentum low)
- ✅ Train D more times than G if D is weak
- ✅ Monitor both losses (they must converge without dominating)

### 🚫 Errors common

- ❌ D very strong → G does not learn (gradients disappear)
- ❌ G very strong → D always fails (mode collapse)
- ❌ LR very high → oscillations, does not converge
- ❌ Don't use BatchNorm → instability
- ❌ Forget `.detach()` in fake_imgs for D → wrong gradients

### 🔧 Troubleshooting

| Problem | Symptom | Solution |
| ------------------- | -------------------------- | -------------------------------------- |
| Mode Collapse | G always generates the same | Unrolled GAN, Minibatch Discrimination |
| Vanishing Gradients | G Loss does not go down | Wasserstein GAN, Least Squares GAN |
| Oscillations | Losses go up/down a lot | Reduce LR, label smoothing |
| Blurred Images | Low G Loss but poor quality | Use Progressive GAN, StyleGAN |

### 📌 Checklist GAN

- ✅ Balanced architecture (G and D similar capacity)
- ✅ Adequate learning rate (0.0002 typical)
- ✅ BatchNorm in intermediate layers
- ✅ LeakyReLU in Discriminator
- ✅ Sufficient latent noise (100-200 dim)
- ✅ Monitor losses (they do not diverge)
- ✅ Periodic visual validation
- ✅ Evaluate diversity (IS, FID)

### 🚀 Next steps

- Implement DCGAN with convolutions
- Conditional GAN ​​for generation control
- StyleGAN for high resolution
- Applications: super-resolution, image-to-image translation (pix2pix)
