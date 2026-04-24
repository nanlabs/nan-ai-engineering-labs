# Práctica 02 — Variational Autoencoders y Diffusion

## 🎯 Objetivos

- Implementar VAE
- Entender reparametrization trick
- Explorar latent space
- Introducción a diffusion models

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: VAE Básico

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

model = VAE()
print(model)
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Latent Space Interpolation

**Enunciado:**
Interpola entre dos imágenes:

- Encode ambas a latent vectors
- Interpola linealmente
- Decode interpolaciones
- Visualiza transición suave

### Ejercicio 2.2: β-VAE

**Enunciado:**
Implementa β-VAE para disentanglement:

```python
loss = BCE + beta * KLD
```

Varía beta (1, 4, 10) y compara.

### Ejercicio 2.3: Conditional VAE

**Enunciado:**
Genera condicionalmente:

- Concatena label con input/latent
- Controla generación por clase
- Visualiza latent space por clase

### Ejercicio 2.4: Diffusion Model Básico

**Enunciado:**
Implementa forward diffusion:

- Añade ruido gaussiano progresivamente
- T pasos hasta ruido puro
- Visualiza proceso

### Ejercicio 2.5: Denoising

**Enunciado:**
Entrena denoiser simple:

- U-Net que predice ruido
- Entrena en múltiples timesteps
- Sampling iterativo

______________________________________________________________________

## ✅ Checklist

- [ ] Implementar VAE con reparameterization
- [ ] VAE loss (BCE + KLD)
- [ ] Latent space exploration
- [ ] β-VAE para disentanglement
- [ ] Diffusion basics

______________________________________________________________________

## 📚 Recursos

- [VAE Tutorial](https://arxiv.org/abs/1906.02691)
- [Diffusion Models Explained](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
