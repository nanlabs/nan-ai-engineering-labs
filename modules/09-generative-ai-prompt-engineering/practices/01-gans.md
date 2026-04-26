# Practice 01 — Generative Adversarial Networks (GANs)

## 🎯 Objectives

- Basic GAN implementation
- Train generator and discriminator
- Generate synthetic images
- Identify mode collapse

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: GAN Simple

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

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: DCGAN

**Statement:**
Implement Deep Convolutional GAN:

- Use Conv2d and ConvTranspose2d
- BatchNorm on both networks
- Train on MNIST or CIFAR-10

### Exercise 2.2: Conditional GAN

**Statement:**
Control generation with labels:

- Concatenate label with latent vector
- Generate specific digits
- Display interpolation

### Exercise 2.3: Mode Collapse Detection

**Statement:**
Detect mode collapse:

- Measure sample diversity
- Compare with real dataset
- Implement mitigation strategies

### Exercise 2.4: WGAN

**Statement:**
Implement Wasserstein GAN:

- Earth Mover's Distance
- Weight clipping
- Compare stability with vanilla GAN

### Exercise 2.5: Image-to-Image with Pix2Pix

**Statement:**
Images Translation:

- U-Net generator
- PatchGAN discriminator
- L1 loss + adversarial loss

______________________________________________________________________

## ✅ Checklist

- [ ] Implement GAN vanilla
- [ ] DCGAN for Images
- [ ] Conditional generation
- [ ] Detect mode collapse
- [ ] WGAN for stability

______________________________________________________________________

## 📚 Resources

- [GAN Lab](https://poloclub.github.io/ganlab/)
- [PyTorch GAN Examples](https://github.com/eriklindernoren/PyTorch-GAN)
