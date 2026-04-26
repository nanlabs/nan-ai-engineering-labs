# Example 01 — Simple Neural network with PyTorch (MNIST)

## Context

You will build your first Neural network from scratch using PyTorch to classify handwritten digits (MNIST). You will learn basic architecture, forward pass, backpropagation and Training.

## Objective

Classify Images of digits (0-9) from the MNIST dataset.

______________________________________________________________________

## 🚀 Step 1: Setup and imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Configurar device (GPU si this disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando device: {device}")

# Semilla para reproducibilidad
torch.manual_seed(42)
```

**Output:**

```
Usando device: cuda  (o cpu si no hay GPU)
```

______________________________________________________________________

## 📥 Step 2: Load and explore Data

### 2.1 Download MNIST

```python
# Transformaciones: convertir a tensor y normalizar
transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte PIL Image a Tensor [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # Media y std de MNIST
])

# Descargar datasets
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Train set: {len(train_dataset)} images")
print(f"Test set: {len(test_dataset)} images")
```

**Output:**

```
Train set: 60000 images
Test set: 10000 images
```

### 2.2 Explore Data

```python
# Visualize examples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    image, label = train_dataset[i]
    # Desnormalizar para visualization
    image = image.squeeze().numpy() * 0.3081 + 0.1307
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Inspeccionar forma
sample_image, sample_label = train_dataset[0]
print(f"\nForma de image: {sample_image.shape}")  # [1, 28, 28]
print(f"Forma de label: {sample_label}")
```

**Output:**

```
Forma de image: torch.Size([1, 28, 28])  👈 1 channel (grayscale), 28x28 pixels
Forma de label: 5
```

### 2.3 DataLoaders

```python
# Batch size
batch_size = 64

# DataLoaders (automatizan batching y shuffling)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True  # Mezclar each epoch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"Number de batches en train: {len(train_loader)}")
print(f"Number de batches en test: {len(test_loader)}")
```

______________________________________________________________________

## 🏗️ Step 3: Define architecture of the Neural network

```python
class SimpleNN(nn.Module):
    """
    Neural network simple (Fully Connected / MLP)
    Arquitectura: 784 -> 128 -> 64 -> 10
    """
    def __init__(self):
        super(SimpleNN, self).__init__()

        # Layers
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 784, Output: 128
        self.fc2 = nn.Linear(128, 64)        # Input: 128, Output: 64
        self.fc3 = nn.Linear(64, 10)         # Input: 64, Output: 10 (classes)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass
        x: [batch_size, 1, 28, 28]
        """
        # Flatten: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(x.size(0), -1)

        # Layer 1: Linear + ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # Layer 2: Linear + ReLU
        x = self.fc2(x)
        x = self.relu(x)

        # Layer 3: Linear (sin activation, esto se have en loss function)
        x = self.fc3(x)

        return x  # Logits (sin softmax aún)

# Create model
model = SimpleNN().to(device)
print(model)

# Contar parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal de parameters: {total_params:,}")
print(f"Parameters entrenables: {trainable_params:,}")
```

**Output:**

```
SimpleNN(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=10, bias=True)
  (relu): ReLU()
)

Total de parameters: 109,386
Parameters entrenables: 109,386
```

**Parameter calculation:**

- fc1: 784 × 128 + 128 (bias) = 100,480
- fc2: 128 × 64 + 64 = 8,256
- fc3: 64 × 10 + 10 = 650
- **Total:** 109,386

______________________________________________________________________

## 🔧 Step 4: Define loss function and optimizer

```python
# Loss function: CrossEntropyLoss (combina LogSoftmax + NLLLoss)
criterion = nn.CrossEntropyLoss()

# Optimizador: Adam (adaptative learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
```

______________________________________________________________________

## 🏋️ Step 5: Training Function

```python
def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train una time
    """
    model.train()  # Modo training (activa dropout, batch norm, etc.)
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # Mover a device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass y optimization
        optimizer.zero_grad()  # Limpiar gradientes previous
        loss.backward()        # Calculate gradientes (backpropagation)
        optimizer.step()       # Actualizar pesos

        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # Clause, Class con mayor probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Promedios
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc
```

______________________________________________________________________

## 🧪 Step 6: Evaluation Function

```python
def evaluate(model, loader, criterion, device):
    """
    Evaluate en validation/test set
    """
    model.eval()  # Modo evaluation (desactiva dropout, batch norm)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No calculator gradientes (ahorra memoria)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc
```

______________________________________________________________________

## 🚂 Step 7: Training Loop

```python
# Hyperparameters
num_epochs = 10

# Tracking de metrics
train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("=== INICIO ENTRENAMIENTO ===\n")

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # Log
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
    print()

print("=== ENTRENAMIENTO COMPLETADO ===")
```

**Output expected:**

```
=== INICIO ENTRENAMIENTO ===

Epoch [1/10]
  Train Loss: 0.3156, Train Acc: 90.72%
  Test Loss:  0.1542, Test Acc:  95.24%

Epoch [2/10]
  Train Loss: 0.1398, Train Acc: 95.84%
  Test Loss:  0.1124, Test Acc:  96.62%

...

Epoch [10/10]
  Train Loss: 0.0234, Train Acc: 99.32%
  Test Loss:  0.0856, Test Acc:  97.89%

=== ENTRENAMIENTO COMPLETADO ===
```

______________________________________________________________________

## 📊 Step 8: Visualize Training curves

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss', color='blue')
axes[0].plot(range(1, num_epochs+1), test_losses, marker='s', label='Test Loss', color='red')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss vs Epoch')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(range(1, num_epochs+1), train_accs, marker='o', label='Train Acc', color='blue')
axes[1].plot(range(1, num_epochs+1), test_accs, marker='s', label='Test Acc', color='red')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy vs Epoch')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Observations:**

- Train accuracy increases faster than test (expected)
- Test loss starts to increase after era 7-8 → sign of slight overfitting
- Final test accuracy ~98% (excellent for baseline)

______________________________________________________________________

## 🔍 Step 9: Inference and Visualization of Predictions

```python
# Obtener batch de test
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Visualize primeros 10
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    # Mover a CPU y desnormalizar
    image = images[i].cpu().squeeze().numpy() * 0.3081 + 0.1307
    true_label = labels[i].item()
    pred_label = predicted[i].item()

    # Color: verde si correcto, rojo si incorrecto
    color = 'green' if true_label == pred_label else 'red'

    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

______________________________________________________________________

## 💾 Paso 10: Save Model

```python
# Save pesos del model
torch.save(model.state_dict(), 'mnist_simple_nn.pth')
print("Model guardado: mnist_simple_nn.pth")

# Load model (en production)
model_loaded = SimpleNN().to(device)
model_loaded.load_state_dict(torch.load('mnist_simple_nn.pth'))
model_loaded.eval()
print("Model cargado exitosamente")
```

______________________________________________________________________

## 📝 Executive summary

### ✅ Results

- **Train accuracy:** 99.32%
- **Test accuracy:** 97.89%
- **Parameters:** 109,386

### 🏗️ Architecture

```
Input (28x28 = 784)
  ↓
Dense(784 → 128) + ReLU
  ↓
Dense(128 → 64) + ReLU
  ↓
Dense(64 → 10)
  ↓
Output (10 classes)
```

### 🎯 Hyperparameters

- **Batch size:** 64
- **Learning rate:** 0.001
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Epochs:** 10

______________________________________________________________________

## 🎓 Lessons learned

### ✅ PyTorch key concepts

1. **`nn.Module`:** Clause, Base class for Models

   - `__init__()`: Define Layers
   - `forward()`: Defines data flow

1. **Device management:**

   - `.to(device)`: Mover Model/Data a GPU/CPU
   - Always move Data before forward pass

1. **Loss functions:**

   - `CrossEntropyLoss`: For multiclass Classification
   - Combine softmax + negative log likelihood

1. **Optimizers:**

- `Adam`: Adjust learning rate automatically by parameter
   - `optimizer.zero_grad()`: Clear previous gradients
   - `loss.backward()`: Calculate gradients (backpropagation)
   - `optimizer.step()`: Update weights

1. **Model Modes:**

   - `model.train()`: Activate dropout, batch norm
   - `model.eval()`: Disable dropout, batch norm
   - `torch.no_grad()`: No calculator gradients (saves memory)

### 🚫 Common errors avoided

- ❌ Forget `optimizer.zero_grad()` → accumulated gradients
- ❌ Do not move Data to device → device mismatch error
- ❌ Don't use `model.eval()` in Evaluation → active dropout
- ❌ Calculate gradients in Evaluation → memory overflow

### 📊 Performance analysis

- **overfitting slighte, level:** Train acc (99.3%) > Test acc (97.9%)
- **Mitigation:** Add dropout, increase dataset, early stopping

### 💡 Possible improvements

1. **Regularization:** Add Dropout entre Layers
1. **Data augmentation:** Rotations, translations
1. **Architecture:** CNN instead of FC (better for Images)
1. **Learning rate schedule:** Reduce lr during Training
1. **Early stopping:** Stop when test loss increases

### 🔧 Next steps

- Example 02: Implement dropout and early stopping
- Module 6: CNNs to improve performance in MNIST
