# Ejemplo 02 — Transfer Learning con ResNet (ImageNet → Custom Dataset)

## Contexto

Entrenar CNNs desde cero requiere millones de imágenes y días de cómputo. **Transfer Learning** permite aprovechar modelos preentrenados (ImageNet: 1.2M imágenes, 1000 clases) y adaptarlos a tu problema específico.

## Objective

Clasificar imágenes de perros vs gatos usando ResNet18 preentrenado, con fine-tuning.

______________________________________________________________________

## 🚀 Paso 1: Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

torch.manual_seed(42)
```

______________________________________________________________________

## 📥 Paso 2: Preparar dataset custom

### 2.1 Estructura de directorios esperada

```
data/
  train/
    dogs/
      dog001.jpg
      dog002.jpg
      ...
    cats/
      cat001.jpg
      cat002.jpg
      ...
  val/
    dogs/
      ...
    cats/
      ...
```

### 2.2 Custom Dataset class

```python
class DogsVsCatsDataset(Dataset):
    """
    Dataset custom para Dogs vs Cats
    """
    def __init__(self, root_dir, transform=None):
        """
        root_dir: ruta a 'train/' o 'val/'
        transform: transformaciones de imágenes
        """
        self.transform = transform
        self.images = []
        self.labels = []

        # Clase 0: dogs, Clase 1: cats
        for class_idx, class_name in enumerate(['dogs', 'cats']):
            class_dir = os.path.join(root_dir, class_name)
            image_files = glob(os.path.join(class_dir, '*.jpg'))

            self.images.extend(image_files)
            self.labels.extend([class_idx] * len(image_files))

        print(f"Loaded {len(self.images)} images from {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Nota: Si no tienes el dataset, puedes descargarlo de:
# https://www.kaggle.com/c/dogs-vs-cats
```

### 2.3 Transformaciones (importante para transfer learning)

```python
# ImageNet stats (obligatorio para modelos preentrenados en ImageNet)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Train: con data augmentation
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),           # Resize to slightly larger
    transforms.RandomCrop(224, 224),          # Crop to 224x224 (ResNet input size)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)  # 👈 CRÍTICO para transfer learning
])

# Val: sin augmentation
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Crear datasets
train_dataset = DogsVsCatsDataset('data/train', transform=transform_train)
val_dataset = DogsVsCatsDataset('data/val', transform=transform_val)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

**Salida esperada:**

```
Loaded 20000 images from data/train
Loaded 5000 images from data/val
Train batches: 625
Val batches: 157
```

______________________________________________________________________

## 🏗️ Paso 3: Cargar modelo preentrenado

### 3.1 ResNet18 preentrenado

```python
# Cargar ResNet18 con pesos de ImageNet
model = models.resnet18(pretrained=True)  # Descarga pesos automáticamente
print("ResNet18 cargado con pesos de ImageNet")

# Inspeccionar arquitectura
print(f"\n{model}")
```

**Arquitectura de ResNet18:**

```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
  (bn1): BatchNorm2d(64)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
  (layer1): Sequential(...)
  (layer2): Sequential(...)
  (layer3): Sequential(...)
  (layer4): Sequential(...)
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000)  👈 1000 clases ImageNet
)
```

### 3.2 Modificar última capa para nuestro problema

```python
# Reemplazar FC layer para 2 clases (dogs vs cats)
num_features = model.fc.in_features  # 512 para ResNet18
model.fc = nn.Linear(num_features, 2)  # 2 clases: dogs, cats

model = model.to(device)
print(f"\nÚltima capa reemplazada: Linear(512 → 2)")
```

______________________________________________________________________

## 🔧 Paso 4: Estrategias de Transfer Learning

### Estrategia 1: Feature Extraction (congelar capas previas)

```python
# Congelar todas las capas EXCEPTO la última FC
for param in model.parameters():
    param.requires_grad = False  # Congelar

# Descongelar solo la última capa
model.fc.requires_grad_(True)

# Verificar
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nEstrategia: Feature Extraction")
print(f"Parámetros entrenables: {trainable_params:,} de {total_params:,} ({100*trainable_params/total_params:.2f}%)")
```

**Salida:**

```
Estrategia: Feature Extraction
Parámetros entrenables: 1,026 de 11,689,512 (0.01%)  👈 Solo última capa
```

**Cuándo usar:** Dataset pequeño (\<5k imágenes), clases similares a ImageNet

### Estrategia 2: Fine-Tuning (descongelar progresivamente)

```python
# Descongelar todas las capas
for param in model.parameters():
    param.requires_grad = True

# Usar diferentes learning rates (LR más bajo para capas tempranas)
optimizer = optim.SGD([
    {'params': model.layer1.parameters(), 'lr': 1e-5},  # Capas tempranas: LR bajo
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}  # Última capa: LR alto
], momentum=0.9, weight_decay=5e-4)

print("\nEstrategia: Fine-Tuning con differential learning rates")
```

**Cuándo usar:** Dataset mediano-grande (>10k imágenes), clases diferentes a ImageNet

**Para este ejemplo, usaremos Feature Extraction (más simple):**

```python
# Volver a Feature Extraction
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)

# Optimizer solo para FC layer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

______________________________________________________________________

## 🏋️ Paso 5: Entrenamiento

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total

# Entrenar
num_epochs = 10
train_losses, train_accs = [], []
val_losses, val_accs = [], []

print("\n=== TRANSFER LEARNING: FEATURE EXTRACTION ===\n")

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
```

**Salida esperada:**

```
=== TRANSFER LEARNING: FEATURE EXTRACTION ===

Epoch [1/10] | Train Loss: 0.2145, Train Acc: 91.23% | Val Loss: 0.1234, Val Acc: 95.67%
Epoch [2/10] | Train Loss: 0.1234, Train Acc: 94.56% | Val Loss: 0.0987, Val Acc: 96.34%
...
Epoch [10/10] | Train Loss: 0.0456, Train Acc: 98.12% | Val Loss: 0.0678, Val Acc: 97.89%

👉 97.89% de accuracy con solo 10 épocas usando un modelo entrenado en ImageNet!
```

______________________________________________________________________

## 📊 Paso 6: Comparar con entrenamiento desde cero

### 6.1 Entrenar ResNet18 desde cero (sin pesos preentrenados)

```python
# Crear modelo sin pretrain
model_scratch = models.resnet18(pretrained=False)
model_scratch.fc = nn.Linear(512, 2)
model_scratch = model_scratch.to(device)

# Entrenar TODAS las capas
criterion_scratch = nn.CrossEntropyLoss()
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.001, weight_decay=5e-4)

print("\n=== ENTRENAMIENTO DESDE CERO ===\n")

scratch_train_accs = []
scratch_val_accs = []

for epoch in range(10):
    train_loss, train_acc = train_epoch(model_scratch, train_loader, criterion_scratch, optimizer_scratch, device)
    val_loss, val_acc = evaluate(model_scratch, val_loader, criterion_scratch, device)

    scratch_train_accs.append(train_acc)
    scratch_val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/10] | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
```

**Salida esperada:**

```
=== ENTRENAMIENTO DESDE CERO ===

Epoch [1/10] | Train Loss: 0.6834, Train Acc: 56.12% | Val Loss: 0.6523, Val Acc: 62.34%
Epoch [2/10] | Train Loss: 0.5123, Train Acc: 74.56% | Val Loss: 0.4821, Val Acc: 78.21%
...
Epoch [10/10] | Train Loss: 0.2134, Train Acc: 91.23% | Val Loss: 0.2987, Val Acc: 88.45%

👉 Solo 88.45% (vs 97.89% con transfer learning)
```

### 6.2 Comparación gráfica

```python
fig, ax = plt.subplots(figsize=(10, 6))

epochs = range(1, 11)
ax.plot(epochs, val_accs, marker='o', label='Transfer Learning (Feature Extraction)', linewidth=2)
ax.plot(epochs, scratch_val_accs, marker='s', label='Desde Cero', linewidth=2)

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Transfer Learning vs Entrenamiento Desde Cero')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

print("\n=== COMPARACIÓN FINAL ===")
print(f"Transfer Learning: {val_accs[-1]:.2f}%")
print(f"Desde Cero:        {scratch_val_accs[-1]:.2f}%")
print(f"Mejora:            +{val_accs[-1] - scratch_val_accs[-1]:.2f}%")
```

**Salida:**

```
=== COMPARACIÓN FINAL ===
Transfer Learning: 97.89%
Desde Cero:        88.45%
Mejora:            +9.44%  👈 ¡Significativo!
```

______________________________________________________________________

## 🚀 Paso 7: Fine-Tuning (opcional)

### 7.1 Descongelar capas progresivamente

```python
print("\n=== FINE-TUNING: Descongelar últimas 2 capas ===\n")

# Descongelar layer4 y fc
for param in model.layer4.parameters():
    param.requires_grad = True

# Optimizer con differential LR
optimizer_finetune = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # LR bajo para layer4
    {'params': model.fc.parameters(), 'lr': 1e-3}       # LR alto para fc
], weight_decay=5e-4)

# Entrenar 5 épocas más
finetune_val_accs = []

for epoch in range(5):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_finetune, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    finetune_val_accs.append(val_acc)

    print(f"Fine-tune Epoch [{epoch+1}/5] | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print(f"\nMejora con fine-tuning: {val_accs[-1]:.2f}% → {finetune_val_accs[-1]:.2f}%")
```

**Salida esperada:**

```
=== FINE-TUNING: Descongelar últimas 2 capas ===

Fine-tune Epoch [1/5] | Train Loss: 0.0389, Train Acc: 98.67% | Val Loss: 0.0512, Val Acc: 98.34%
Fine-tune Epoch [2/5] | Train Loss: 0.0267, Train Acc: 99.12% | Val Loss: 0.0487, Val Acc: 98.56%
...
Fine-tune Epoch [5/5] | Train Loss: 0.0156, Train Acc: 99.56% | Val Loss: 0.0434, Val Acc: 98.89%

Mejora con fine-tuning: 97.89% → 98.89%  👈 +1% adicional
```

______________________________________________________________________

## 🔍 Paso 8: Visualizar features aprendidas

### 8.1 Activations de capa intermedia

```python
# Hook para capturar activaciones
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Registrar hook en layer4
model.layer4.register_forward_hook(get_activation('layer4'))

# Procesar una imagen
model.eval()
sample_image, sample_label = val_dataset[0]
sample_image_batch = sample_image.unsqueeze(0).to(device)  # [1, 3, 224, 224]

with torch.no_grad():
    output = model(sample_image_batch)

# Visualizar activaciones (primeros 16 canales)
layer4_act = activations['layer4'].squeeze()  # [512, 7, 7]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.ravel()

for i in range(16):
    axes[i].imshow(layer4_act[i].cpu().numpy(), cmap='viridis')
    axes[i].axis('off')
    axes[i].set_title(f'Channel {i}')

plt.suptitle('Activaciones de layer4 (ResNet18)')
plt.tight_layout()
plt.show()
```

**Interpretación:** Cada canal detecta different features (bordes, texturas, partes del objeto).

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Comparación de estrategias

| Estrategia             | Parámetros Entrenables | Val Acc @ 10 epochs | Tiempo entrenamiento | Cuándo usar                                      |
| ---------------------- | ---------------------- | ------------------- | -------------------- | ------------------------------------------------ |
| **Feature Extraction** | 1,026 (0.01%)          | **97.89%**          | ~5 min (GPU)         | Dataset pequeño, clases similares a ImageNet     |
| **Fine-Tuning**        | 11,689,512 (100%)      | **98.89%**          | ~25 min (GPU)        | Dataset mediano-grande, clases diferentes        |
| **Desde Cero**         | 11,689,512 (100%)      | 88.45%              | ~30 min (GPU)        | Dataset muy grande (>100k), tarea muy específica |

### 🎯 Ventajas del Transfer Learning

1. **Menos datos:** 97.89% con 20k imágenes vs 88.45% desde cero
1. **Entrenamiento rápido:** Feature extraction 5x más rápido
1. **Mejor generalización:** Features de ImageNet son universales
1. **Menor overfitting:** Capas convolucionales ya optimizadas

______________________________________________________________________

## 🎓 Lecciones aprendidas

### ✅ Transfer Learning: Guía práctica

**1. Elegir arquitectura:**

- **ResNet (18, 34, 50):** Balance entre accuracy y velocidad
- **EfficientNet (B0-B7):** State-of-the-art, más eficiente
- **MobileNet:** Para edge devices (celulares, IoT)
- **VGG:** Obsoleto (muchos parámetros, lento)

**2. Cuándo usar cada estrategia:**

| Dataset Size  | Similitud a ImageNet | Estrategia Recomendada                 |
| ------------- | -------------------- | -------------------------------------- |
| \<1k imágenes | Alta                 | Feature Extraction                     |
| \<1k imágenes | Baja                 | Data Augmentation + Feature Extraction |
| 1k-10k        | Alta                 | Feature Extraction → Fine-Tuning       |
| 1k-10k        | Baja                 | Fine-Tuning todas las capas            |
| >10k          | Cualquiera           | Fine-Tuning                            |
| >100k         | Baja                 | Considerar entrenar desde cero         |

**3. Preprocessing CRÍTICO:**

- ✅ Usar **ImageNet stats** (mean/std) para normalizar
- ✅ Resize a **224×224** (mayoría de modelos)
- ❌ NO uses tus propias stats → rompe transfer learning

**4. Learning rates:**

- Feature Extraction: LR = 1e-3 (solo FC layer)
- Fine-Tuning: Differential LR
  - Capas tempranas: 1e-5 (cambios pequeños)
  - Capas tardías: 1e-4
  - FC layer: 1e-3

**5. Descongelar progresivamente:**

```python
# Paso 1: Entrenar solo FC (5-10 épocas)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)

# Paso 2: Descongelar layer4 (5 épocas)
model.layer4.requires_grad_(True)

# Paso 3: Descongelar layer3 (5 épocas)
model.layer3.requires_grad_(True)

# ... hasta descongelar todo
```

### 💡 Mejoras adicionales

1. **Mixed Precision Training:** FP16 para entrenar más rápido
1. **Gradient Accumulation:** Simular batch sizes grandes
1. **Test-Time Augmentation (TTA):** Predecir en múltiples versiones augmentadas
1. **Ensemble:** Combinar múltiples modelos (ResNet + EfficientNet)
1. **Progressive Resizing:** Entrenar con imágenes pequeñas primero, luego grandes

### 🚫 Errores comunes

- ❌ **No usar ImageNet normalization:** Features no alinean con pesos pretrained
- ❌ **LR muy alto en fine-tuning:** Destruye features aprendidasen ImageNet
- ❌ **Descongelar todo desde el inicio:** Capas tempranas aprenden muy rápido y rompen features
- ❌ **No hacer data augmentation:** Overfitting con datasets pequeños

______________________________________________________________________

## 🔧 Código de producción

```python
# Cargar modelo preentrenado y modificar FC
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Feature Extraction: congelar capas previas
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)

# Transformaciones (CRÍTICO: usar ImageNet stats)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Entrenar
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Después de convergencia: fine-tuning
model.layer4.requires_grad_(True)
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# Guardar
torch.save(model.state_dict(), 'model_finetunned.pth')
```

### 📌 Checklist Transfer Learning

- ✅ Usar modelo preentrenado (ResNet, EfficientNet)
- ✅ Modificar última capa para tu número de clases
- ✅ Preprocessing con ImageNet stats
- ✅ Empezar con Feature Extraction (congelar capas)
- ✅ usar LR apropiado (1e-3 para FC)
- ✅ Data augmentation agresiva
- ✅ Monitorear overfitting (train/val gap)
- ✅ Si dataset >10k: hacer fine-tuning
- ✅ Fine-tuning con differential LR
- ✅ Descongelar progresivamente (de atrás hacia adelante)
