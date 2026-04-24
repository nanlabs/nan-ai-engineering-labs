# Ejemplo 01 — Red Neuronal Simple con PyTorch (MNIST)

## Contexto

Construirás tu primera red neuronal desde cero usando PyTorch para clasificar dígitos escritos a mano (MNIST). Aprenderás arquitectura básica, forward pass, backpropagation y entrenamiento.

## Objective

Clasificar imágenes de dígitos (0-9) del dataset MNIST.

______________________________________________________________________

## 🚀 Paso 1: Setup e importaciones

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Configurar device (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando device: {device}")

# Semilla para reproducibilidad
torch.manual_seed(42)
```

**Salida:**

```
Usando device: cuda  (o cpu si no hay GPU)
```

______________________________________________________________________

## 📥 Paso 2: Cargar y explorar datos

### 2.1 Descargar MNIST

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

print(f"Train set: {len(train_dataset)} imágenes")
print(f"Test set: {len(test_dataset)} imágenes")
```

**Salida:**

```
Train set: 60000 imágenes
Test set: 10000 imágenes
```

### 2.2 Explorar datos

```python
# Visualizar ejemplos
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    image, label = train_dataset[i]
    # Desnormalizar para visualización
    image = image.squeeze().numpy() * 0.3081 + 0.1307
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Inspeccionar forma
sample_image, sample_label = train_dataset[0]
print(f"\nForma de imagen: {sample_image.shape}")  # [1, 28, 28]
print(f"Forma de label: {sample_label}")
```

**Salida:**

```
Forma de imagen: torch.Size([1, 28, 28])  👈 1 canal (grayscale), 28x28 píxeles
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
    shuffle=True  # Mezclar cada epoch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"Número de batches en train: {len(train_loader)}")
print(f"Número de batches en test: {len(test_loader)}")
```

______________________________________________________________________

## 🏗️ Paso 3: Definir arquitectura de la red neuronal

```python
class SimpleNN(nn.Module):
    """
    Red neuronal simple (Fully Connected / MLP)
    Arquitectura: 784 -> 128 -> 64 -> 10
    """
    def __init__(self):
        super(SimpleNN, self).__init__()

        # Capas
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 784, Output: 128
        self.fc2 = nn.Linear(128, 64)        # Input: 128, Output: 64
        self.fc3 = nn.Linear(64, 10)         # Input: 64, Output: 10 (clases)

        # Activación
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass
        x: [batch_size, 1, 28, 28]
        """
        # Flatten: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(x.size(0), -1)

        # Capa 1: Linear + ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # Capa 2: Linear + ReLU
        x = self.fc2(x)
        x = self.relu(x)

        # Capa 3: Linear (sin activación, esto se hace en loss function)
        x = self.fc3(x)

        return x  # Logits (sin softmax aún)

# Crear modelo
model = SimpleNN().to(device)
print(model)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal de parámetros: {total_params:,}")
print(f"Parámetros entrenables: {trainable_params:,}")
```

**Salida:**

```
SimpleNN(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=10, bias=True)
  (relu): ReLU()
)

Total de parámetros: 109,386
Parámetros entrenables: 109,386
```

**Cálculo de parámetros:**

- fc1: 784 × 128 + 128 (bias) = 100,480
- fc2: 128 × 64 + 64 = 8,256
- fc3: 64 × 10 + 10 = 650
- **Total:** 109,386

______________________________________________________________________

## 🔧 Paso 4: Definir loss function y optimizador

```python
# Loss function: CrossEntropyLoss (combina LogSoftmax + NLLLoss)
criterion = nn.CrossEntropyLoss()

# Optimizador: Adam (adaptative learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
```

______________________________________________________________________

## 🏋️ Paso 5: Función de entrenamiento

```python
def train_epoch(model, loader, criterion, optimizer, device):
    """
    Entrenar una época
    """
    model.train()  # Modo entrenamiento (activa dropout, batch norm, etc.)
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # Mover a device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass y optimización
        optimizer.zero_grad()  # Limpiar gradientes previos
        loss.backward()        # Calcular gradientes (backpropagation)
        optimizer.step()       # Actualizar pesos

        # Métricas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # Clase con mayor probabilidad
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Promedios
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc
```

______________________________________________________________________

## 🧪 Paso 6: Función de evaluación

```python
def evaluate(model, loader, criterion, device):
    """
    Evaluar en validation/test set
    """
    model.eval()  # Modo evaluación (desactiva dropout, batch norm)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No calcular gradientes (ahorra memoria)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Métricas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc
```

______________________________________________________________________

## 🚂 Paso 7: Loop de entrenamiento

```python
# Hiperparámetros
num_epochs = 10

# Tracking de métricas
train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("=== INICIO ENTRENAMIENTO ===\n")

for epoch in range(num_epochs):
    # Entrenar
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Evaluar
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

**Salida esperada:**

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

## 📊 Paso 8: Visualizar curvas de entrenamiento

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

**Observaciones:**

- Train accuracy aumenta más rápido que test (esperado)
- Test loss empieza a aumentar después de época 7-8 → señal de overfitting leve
- Test accuracy final ~98% (excelente para baseline)

______________________________________________________________________

## 🔍 Paso 9: Inferencia y visualización de predicciones

```python
# Obtener batch de test
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Predecir
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Visualizar primeros 10
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

## 💾 Paso 10: Guardar modelo

```python
# Guardar pesos del modelo
torch.save(model.state_dict(), 'mnist_simple_nn.pth')
print("Modelo guardado: mnist_simple_nn.pth")

# Cargar modelo (en producción)
model_loaded = SimpleNN().to(device)
model_loaded.load_state_dict(torch.load('mnist_simple_nn.pth'))
model_loaded.eval()
print("Modelo cargado exitosamente")
```

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Resultados

- **Train Accuracy:** 99.32%
- **Test Accuracy:** 97.89%
- **Parámetros:** 109,386

### 🏗️ Arquitectura

```
Input (28x28 = 784)
  ↓
Dense(784 → 128) + ReLU
  ↓
Dense(128 → 64) + ReLU
  ↓
Dense(64 → 10)
  ↓
Output (10 clases)
```

### 🎯 Hiperparámetros

- **Batch size:** 64
- **Learning rate:** 0.001
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Epochs:** 10

______________________________________________________________________

## 🎓 Lecciones aprendidas

### ✅ Conceptos clave de PyTorch

1. **`nn.Module`:** Clase base para modelos

   - `__init__()`: Definir capas
   - `forward()`: Definir flujo de datos

1. **Device management:**

   - `.to(device)`: Mover modelo/datos a GPU/CPU
   - Siempre mover datos antes de forward pass

1. **Loss functions:**

   - `CrossEntropyLoss`: Para clasificación multiclase
   - Combina softmax + negative log likelihood

1. **Optimizadores:**

   - `Adam`: Ajusta learning rate automáticamente por parámetro
   - `optimizer.zero_grad()`: Limpiar gradientes previos
   - `loss.backward()`: Calcular gradientes (backpropagation)
   - `optimizer.step()`: Actualizar pesos

1. **Modos del modelo:**

   - `model.train()`: Activa dropout, batch norm
   - `model.eval()`: Desactiva dropout, batch norm
   - `torch.no_grad()`: No calcular gradientes (ahorra memoria)

### 🚫 Errores comunes evitados

- ❌ Olvidar `optimizer.zero_grad()` → gradientes acumulados
- ❌ No mover datos a device → error de device mismatch
- ❌ No usar `model.eval()` en evaluación → dropout activo
- ❌ Calcular gradientes en evaluación → memory overflow

### 📊 Análisis de performance

- **Overfitting leve:** Train acc (99.3%) > Test acc (97.9%)
- **Mitigación:** Agregar dropout, aumentar dataset, early stopping

### 💡 Mejoras posibles

1. **Regularización:** Agregar Dropout entre capas
1. **Data augmentation:** Rotaciones, traslaciones
1. **Arquitectura:** CNN en lugar de FC (mejor para imágenes)
1. **Learning rate schedule:** Reducir lr durante entrenamiento
1. **Early stopping:** Detener cuando test loss aumenta

### 🔧 Próximos pasos

- Ejemplo 02: Implementar dropout y early stopping
- Módulo 6: CNNs para mejorar performance en MNIST
