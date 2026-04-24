# Práctica 01 — Generative Adversarial Networks (GANs)

## 🎯 Objetivos

- Implementar GAN básica
- Entrenar generador y discriminador
- Generar imágenes sintéticas
- Identificar mode collapse

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: GAN Simple

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Inicializar
latent_dim = 100
gen = Generator(latent_dim)
disc = Discriminator()

criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr=0.0002)
opt_disc = optim.Adam(disc.parameters(), lr=0.0002)

print(f"Generator params: {sum(p.numel() for p in gen.parameters())}")
print(f"Discriminator params: {sum(p.numel() for p in disc.parameters())}")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: DCGAN

**Enunciado:**
Implementa Deep Convolutional GAN:

- Usa Conv2d y ConvTranspose2d
- BatchNorm en ambos networks
- Entrena en MNIST o CIFAR-10

### Ejercicio 2.2: Conditional GAN

**Enunciado:**
Controla generación con labels:

- Concatena label con latent vector
- Genera dígitos específicos
- Visualiza interpolación

### Ejercicio 2.3: Mode Collapse Detection

**Enunciado:**
Detecta mode collapse:

- Mide diversidad de samples
- Compara con dataset real
- Implementa mitigation strategies

### Ejercicio 2.4: WGAN

**Enunciado:**
Implementa Wasserstein GAN:

- Earth Mover's Distance
- Weight clipping
- Compara estabilidad con GAN vanilla

### Ejercicio 2.5: Image-to-Image con Pix2Pix

**Enunciado:**
Traducción de imágenes:

- U-Net generator
- PatchGAN discriminator
- L1 loss + adversarial loss

______________________________________________________________________

## ✅ Checklist

- [ ] Implementar GAN vanilla
- [ ] DCGAN para imágenes
- [ ] Conditional generation
- [ ] Detectar mode collapse
- [ ] WGAN para estabilidad

______________________________________________________________________

## 📚 Recursos

- [GAN Lab](https://poloclub.github.io/ganlab/)
- [PyTorch GAN Examples](https://github.com/eriklindernoren/PyTorch-GAN)
