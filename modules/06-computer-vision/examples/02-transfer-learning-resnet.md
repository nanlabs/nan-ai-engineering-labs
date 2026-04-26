# Example 02 — Transfer Learning with ResNet (ImageNet → Custom Dataset)

## Context

Train CNNs from scratch requires millions of Images and days of computation. **Transfer Learning** allows you to take advantage of pre-trained Models (ImageNet: 1.2M Images, 1000 classes) and adapt them to your specific Problem.

## Objective

Classify images of dogs vs cats using pre-trained ResNet18, with fine-tuning.

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

## 📥 Step 2: Prepare custom dataset

### 2.1 Structure of expected directors

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
        transform: transformations de images
        """
        self.transform = transform
        self.images = []
        self.labels = []

        # Clause, Class 0: dogs, Clause, Class 1: cats
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

### 2.3 Transformations (important for transfer learning)

```python
# ImageNet stats (obligatorio para models pretrained en ImageNet)
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
    transforms.Normalize(imagenet_mean, imagenet_std)  # 👈 CRITICAL para transfer learning
])

# Val: sin augmentation
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Create datasets
train_dataset = DogsVsCatsDataset('data/train', transform=transform_train)
val_dataset = DogsVsCatsDataset('data/val', transform=transform_val)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

**Output expected:**

```
Loaded 20000 images from data/train
Loaded 5000 images from data/val
Train batches: 625
Val batches: 157
```

______________________________________________________________________

## 🏗️ Step 3: Pre-trained Load Model

### 3.1 Pretrained ResNet18

```python
# Load ResNet18 con pesos de ImageNet
model = models.resnet18(pretrained=True)  # Descarga pesos automatically
print("ResNet18 cargado con pesos de ImageNet")

# Inspeccionar architecture
print(f"\n{model}")
```

**ResNet18 architecture:**

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
  (fc): Linear(in_features=512, out_features=1000)  👈 1000 classes ImageNet
)
```

### 3.2 Modify last Layer for our Problem

```python
# Reemplazar FC layer para 2 classes (dogs vs cats)
num_features = model.fc.in_features  # 512 para ResNet18
model.fc = nn.Linear(num_features, 2)  # 2 classes: dogs, cats

model = model.to(device)
print(f"\nÚltima layer reemplazada: Linear(512 → 2)")
```

______________________________________________________________________

## 🔧 Step 4: Transfer Learning Strategies

### Strategy 1: Feature Extraction (freeze previous Layers)

```python
# Congelar all las layers EXCEPTO la last FC
for param in model.parameters():
    param.requires_grad = False  # Congelar

# Descongelar solo la last layer
model.fc.requires_grad_(True)

# Verificar
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nEstrategia: Feature Extraction")
print(f"Parameters entrenables: {trainable_params:,} de {total_params:,} ({100*trainable_params/total_params:.2f}%)")
```

**Output:**

```
Estrategia: Feature Extraction
Parameters entrenables: 1,026 de 11,689,512 (0.01%)  👈 Solo last layer
```

**When use:** Small dataset (\<5k Images), classes similar to ImageNet

### Strategy 2: Fine-Tuning (defrost progressively)

```python
# Descongelar all las layers
for param in model.parameters():
    param.requires_grad = True

# Wear different learning rates (LR más low para layers tempranas)
optimizer = optim.SGD([
    {'params': model.layer1.parameters(), 'lr': 1e-5},  # Layers tempranas: LR low
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}  # Last layer: LR alto
], momentum=0.9, weight_decay=5e-4)

print("\nEstrategia: Fine-Tuning con differential learning rates")
```

**When use:** Medium-large dataset (>10k Images), classes different to ImageNet

**For this Example, we will use Feature Extraction (simpler):**

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

## 🏋️ Paso 5: Training

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

# Train
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

**Output expected:**

```
=== TRANSFER LEARNING: FEATURE EXTRACTION ===

Epoch [1/10] | Train Loss: 0.2145, Train Acc: 91.23% | Val Loss: 0.1234, Val Acc: 95.67%
Epoch [2/10] | Train Loss: 0.1234, Train Acc: 94.56% | Val Loss: 0.0987, Val Acc: 96.34%
...
Epoch [10/10] | Train Loss: 0.0456, Train Acc: 98.12% | Val Loss: 0.0678, Val Acc: 97.89%

👉 97.89% de accuracy con solo 10 eras using un model trained en ImageNet!
```

______________________________________________________________________

## 📊 Step 6: Compare with Training from scratch

### 6.1 Train ResNet18 from scratch (no pre-trained weights)

```python
# Create model sin pretrain
model_scratch = models.resnet18(pretrained=False)
model_scratch.fc = nn.Linear(512, 2)
model_scratch = model_scratch.to(device)

# Train TODAS las layers
criterion_scratch = nn.CrossEntropyLoss()
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.001, weight_decay=5e-4)

print("\n=== TRAINING DESDE CERO ===\n")

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

**Output expected:**

```
=== TRAINING DESDE CERO ===

Epoch [1/10] | Train Loss: 0.6834, Train Acc: 56.12% | Val Loss: 0.6523, Val Acc: 62.34%
Epoch [2/10] | Train Loss: 0.5123, Train Acc: 74.56% | Val Loss: 0.4821, Val Acc: 78.21%
...
Epoch [10/10] | Train Loss: 0.2134, Train Acc: 91.23% | Val Loss: 0.2987, Val Acc: 88.45%

👉 Solo 88.45% (vs 97.89% con transfer learning)
```

### 6.2 Graphical comparison

```python
fig, ax = plt.subplots(figsize=(10, 6))

epochs = range(1, 11)
ax.plot(epochs, val_accs, marker='o', label='Transfer Learning (Feature Extraction)', linewidth=2)
ax.plot(epochs, scratch_val_accs, marker='s', label='Desde Cero', linewidth=2)

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Transfer Learning vs Training Desde Cero')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

print("\n=== COMPARISON FINAL ===")
print(f"Transfer Learning: {val_accs[-1]:.2f}%")
print(f"Desde Cero:        {scratch_val_accs[-1]:.2f}%")
print(f"Improvement:            +{val_accs[-1] - scratch_val_accs[-1]:.2f}%")
```

**Output:**

```
=== COMPARISON FINAL ===
Transfer Learning: 97.89%
Desde Cero:        88.45%
Improvement:            +9.44%  👈 ¡Significativo!
```

______________________________________________________________________

## 🚀 Step 7: Fine-Tuning (optional)

### 7.1 Defrost Layers progressively

```python
print("\n=== FINE-TUNING: Descongelar latest 2 layers ===\n")

# Descongelar layer4 y fc
for param in model.layer4.parameters():
    param.requires_grad = True

# Optimizer con differential LR
optimizer_finetune = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # LR low para layer4
    {'params': model.fc.parameters(), 'lr': 1e-3}       # LR alto para fc
], weight_decay=5e-4)

# Train 5 eras más
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

**Output expected:**

```
=== FINE-TUNING: Descongelar latest 2 layers ===

Fine-tune Epoch [1/5] | Train Loss: 0.0389, Train Acc: 98.67% | Val Loss: 0.0512, Val Acc: 98.34%
Fine-tune Epoch [2/5] | Train Loss: 0.0267, Train Acc: 99.12% | Val Loss: 0.0487, Val Acc: 98.56%
...
Fine-tune Epoch [5/5] | Train Loss: 0.0156, Train Acc: 99.56% | Val Loss: 0.0434, Val Acc: 98.89%

Improvement con fine-tuning: 97.89% → 98.89%  👈 +1% adicional
```

______________________________________________________________________

## 🔍 Paso 8: Visualize features learned

### 8.1 Intermediate Layer Activations

```python
# Hook para capturar activaciones
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Registrar hook en layer4
model.layer4.register_forward_hook(get_activation('layer4'))

# Procesar una image
model.eval()
sample_image, sample_label = val_dataset[0]
sample_image_batch = sample_image.unsqueeze(0).to(device)  # [1, 3, 224, 224]

with torch.no_grad():
    output = model(sample_image_batch)

# Visualize activaciones (primeros 16 channels)
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

**Interpretation:** Each Channel detects different features (edges, textures, parts of the object).

______________________________________________________________________

## 📝 Executive summary

### ✅ Strategy comparison

| Strategy | Trainable Parameters | Val Acc @ 10 epochs | Training Time | When to use |
| ---------------------- | ---------------------- | ------------------- | -------------------- | ------------------------------------------------ |
| **Feature Extraction** | 1,026 (0.01%) | **97.89%** | ~5 min (GPU) | Small dataset, classes similar to ImageNet |
| **Fine-Tuning** | 11,689,512 (100%) | **98.89%** | ~25 min (GPU) | Medium-large dataset, different classes |
| **From Scratch** | 11,689,512 (100%) | 88.45% | ~30 min (GPU) | Very large dataset (>100k), very specific task |

### 🎯 Advantages of Transfer Learning

1. **Less Data:** 97.89% with 20k Images vs 88.45% from zero
1. **Training fast:** Feature extraction 5x faster
1. **Best generalization:** ImageNet features are universal
1. **Minor overfitting:** Convolutional layers already optimized

______________________________________________________________________

## 🎓 Lessons learned

### ✅ Transfer Learning: Practical Guide

**1. Choose architecture:**

- **ResNet (18, 34, 50):** Balance between accuracy and speed
- **EfficientNet (B0-B7):** State-of-the-art, most efficient
- **MobileNet:** For edge devices (cellular, IoT)
- **VGG:** Deprecated (many parameters, slow)

**2. When use each strategy:**

| Dataset Size | Similarity to ImageNet | Recommended Strategy |
| ------------- | -------------------- | -------------------------------------- |
| \<1k Images | High | Feature Extraction |
| \<1k Images | Low | Data Augmentation + Feature Extraction |
| 1k-10k | High | Feature Extraction → Fine-Tuning |
| 1k-10k | Low | Fine-Tuning all Layers |
| >10k | Any | Fine-Tuning |
| >100k | Low | Consider training from scratch |

**3. CRITICAL Preprocessing:**

- ✅ Use **ImageNet stats** (mean/std) to normalize
- ✅ Resize to **224×224** (most Models)
- ❌ DO NOT use your own stats → breaks transfer learning

**4. Learning rates:**

- Feature Extraction: LR = 1e-3 (solo FC layer)
- Fine-Tuning: Differential LR
- Early layers: 1e-5 (small changes)
- Late layers: 1e-4
  - FC layer: 1e-3

**5. Defrost progressively:**

```python
# Paso 1: Train solo FC (5-10 eras)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)

# Paso 2: Descongelar layer4 (5 eras)
model.layer4.requires_grad_(True)

# Paso 3: Descongelar layer3 (5 eras)
model.layer3.requires_grad_(True)

# ... hasta descongelar todo
```

### 💡 Additional improvements

1. **Mixed Precision Training:** FP16 to train faster
1. **gradient Accumulation:** Similar batch sizes large
1. **Test-Time Augmentation (TTA):** Predict in multiple augmented versions
1. **Ensemble:** Combine multiple Models (ResNet + EfficientNet)
1. **Progressive Resizing:** Train with small Images first, then large

### 🚫 Errors common

- ❌ **Do not use ImageNet normalization:** Features do not align with pretrained weights
- ❌ **LR very high in fine-tuning:** Destroys features learned in ImageNet
- ❌ **Unfreeze everything from the beginning:** Early layers learn very quickly and break features
- ❌ **Do not do data augmentation:** overfitting with small datasets

______________________________________________________________________

## 🔧 Production code

```python
# Load model preentrenado y modificar FC
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Feature Extraction: congelar layers previas
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)

# Transformaciones (CRITICAL: use ImageNet stats)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Train
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# After de convergencia: fine-tuning
model.layer4.requires_grad_(True)
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# Save
torch.save(model.state_dict(), 'model_finetunned.pth')
```

### 📌 Checklist Transfer Learning

- ✅ Use pre-trained Model (ResNet, EfficientNet)
- ✅ Modify last Layer for your number of classes
- ✅ Preprocessing with ImageNet stats
- ✅ Start with Feature Extraction (freeze Layers)
- ✅ use appropriate LR (1e-3 for FC)
- ✅ Aggressive data augmentation
- ✅ Monitor overfitting (train/val gap)
- ✅ If dataset >10k: do fine-tuning
- ✅ Fine-tuning with LR differential
- ✅ Defrost progressively (from back to front)
