# Práctica 01 — Clasificación de Imágenes con CNNs

## 🎯 Objetivos

- Construir CNNs para datasets de imágenes
- Aplicar data augmentation efectivo
- Comparar arquitecturas (custom vs preentrenadas)
- Optimizar performance

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: Custom CNN para CIFAR-10

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

print(f"Train samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: ResNet vs VGG

**Enunciado:**
Compara ResNet18 y VGG16 en CIFAR-10:

- Accuracy
- Training time
- Number of parameters

### Ejercicio 2.2: Image Segmentation

**Enunciado:**
Implementa U-Net simple para segmentación:

- Encoder-decoder architecture
- Skip connections
- Evalúa con IoU metric

### Ejercicio 2.3: Object Detection

**Enunciado:**
Usa YOLO pre-entrenado para detección:

- Detecta objetos en imágenes
- Dibuja bounding boxes
- Calcula confidence scores

### Ejercicio 2.4: Style Transfer

**Enunciado:**
Implementa neural style transfer:

- Content image + style image
- Optimiza imagen para combinar ambos
- Visualiza proceso

### Ejercicio 2.5: Data Augmentation Ablation

**Enunciado:**
Prueba diferentes augmentations:

- Sin augmentation
- Solo flips
- Flips + rotations
- Flips + rotations + color jitter

Compara accuracies.

______________________________________________________________________

## ✅ Checklist

- [ ] Construir CNNs custom
- [ ] Aplicar data augmentation
- [ ] Usar modelos preentrenados
- [ ] Implementar segmentación
- [ ] Evaluar con métricas apropiadas

______________________________________________________________________

## 📚 Recursos

- [PyTorch Vision](https://pytorch.org/vision/stable/index.html)
- [Papers With Code - Computer Vision](https://paperswithcode.com/area/computer-vision)
