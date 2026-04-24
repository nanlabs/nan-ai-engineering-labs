# Práctica 02 — CNNs y Transfer Learning

## 🎯 Objetivos

- Construir CNNs para clasificación de imágenes
- Aplicar data augmentation
- Usar transfer learning con modelos preentrenados
- Fine-tuning de redes profundas

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: CNN Simple para MNIST

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# CNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()
print(model)
```

**✅ Solución - Entrenar:**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Test
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Data Augmentation

**Enunciado:**
Aplica transformaciones en training:

- RandomHorizontalFlip
- RandomRotation(10)
- RandomCrop con padding

Compara accuracy con vs sin augmentation.

### Ejercicio 2.2: Transfer Learning con ResNet

**Enunciado:**
Usa `torchvision.models.resnet18(pretrained=True)`:

1. Congela capas convolucionales
1. Reemplaza última capa FC (num_classes=10)
1. Entrena solo la última capa

### Ejercicio 2.3: Fine-Tuning

**Enunciado:**
Después de ejercicio anterior:

1. Descongela todas las capas
1. Usa learning rate bajo (1e-5)
1. Entrena todo el modelo

### Ejercicio 2.4: Visualización de Activaciones

**Enunciado:**
Extrae y visualiza feature maps:

- Después de conv1
- Después de conv2
- Para una imagen de test

### Ejercicio 2.5: Grad-CAM

**Enunciado:**
Implementa Grad-CAM simplificado:

- Identifica qué regiones influyen en predicción
- Superpone heatmap sobre imagen original

______________________________________________________________________

## ✅ Checklist

- [ ] Construir CNNs con Conv2d, MaxPool2d
- [ ] Calcular dimensiones output de conv layers
- [ ] Aplicar data augmentation
- [ ] Usar modelos preentrenados (transfer learning)
- [ ] Congelar/descongelar capas
- [ ] Fine-tuning con differential learning rates
- [ ] Visualizar feature maps

______________________________________________________________________

## 📚 Recursos

- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
