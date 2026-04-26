# Practice 02 — Variational Autoencoders and Diffusion

## 🎯 Objectives

- Implement VAE
- Entender reparametrization trick
- Explorar latent space
- Introduction a diffusion models

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Basic VAE

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

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Latent Space Interpolation

**Statement:**
Interpola entre dos Images:

- Encode ambas a latent vectors
- Interpola linealmente
- Decode interpolaciones
- Visualize smooth transition

### Exercise 2.2: β-VAE

**Statement:**
Implement β-VAE for disentanglement:

```python
loss = BCE + beta * KLD
```

Vary beta (1, 4, 10) and compare.

### Exercise 2.3: Conditional VAE

**Statement:**
Genera condicionalmente:

- Concatenate label with input/latent
- Control generation by class
- Visualize latent space by clause, class

### Exercise 2.4: Basic Diffusion Model

**Statement:**
Implement forward diffusion:

- Add Gaussian noise progressively
- T steps hasta noise puro
- Visualize process

### Exercise 2.5: Denoising

**Statement:**
Train denoiser simple:

- U-Net predicting noise
- Train in multiple timesteps
- Sampling iterativo

______________________________________________________________________

## ✅ Checklist

- [ ] Implement VAE with reparameterization
- [ ] VAE loss (BCE + KLD)
- [ ] Latent space exploration
- [ ] β-VAE for disentanglement
- [ ] Diffusion basics

______________________________________________________________________

## 📚 Resources

- [VAE Tutorial](https://arxiv.org/abs/1906.02691)
- [Diffusion Models Explained](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
