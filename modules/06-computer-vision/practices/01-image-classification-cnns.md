# Practice 01 — Image Classification with CNNs

## 🎯 Objectives

- Build CNNs for Images datasets
- Apply data augmentation efectivo
- Compare arquitecturas (custom vs preentrenadas)
- Optimizar performance

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Custom CNN for CIFAR-10

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

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: ResNet vs VGG

**Statement:**
Compare ResNet18 and VGG16 on CIFAR-10:

- accuracy
- Training time
- Number of parameters

### Exercise 2.2: Image Segmentation

**Statement:**
Implement simple U-Net for segmentation:

- Encoder-decoder architecture
- Skip connections
- Evaluate with IoU metric

### Exercise 2.3: Object Detection

**Statement:**
Use pre-trained YOLO for detection:

- Detect objects in Images
- Dibuja bounding boxes
- Calculate confidence scores

### Exercise 2.4: Style Transfer

**Statement:**
Implement neural style transfer:

- Content image + style image
- Optimize Image to combine both
- Visualize process

### Exercise 2.5: Data Augmentation Ablation

**Statement:**
Testing different augmentations:

- Sin augmentation
- Solo flips
- Flips + rotations
- Flips + rotations + color jitter

Compare accuracies.

______________________________________________________________________

## ✅ Checklist

- [ ] Build CNNs custom
- [ ] Apply data augmentation
- [ ] Wear Models pretrained
- [ ] Implement segmentation
- [ ] Evaluate yourself with appropriate Metrics

______________________________________________________________________

## 📚 Resources

- [PyTorch Vision](https://pytorch.org/vision/stable/index.html)
- [Papers With Code - Computer Vision](https://paperswithcode.com/area/computer-vision)
