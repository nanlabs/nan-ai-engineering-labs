# Theory — Computer Vision

## Why this module matters

Computer Vision allows machines to "see" and understand the visual world. From medical diagnosis to autonomous vehicles, facial recognition and content moderation, CV is essential in modern AI applications.

______________________________________________________________________

## 1. What is Computer Vision?

**Computer Vision (CV)** is the area of ​​AI that allows systems to interpret and analyze **Images and videos** to detect patterns, objects, actions or events.

### Real applications

- **Health:** Detection of tumors in X-rays.
- **Retail:** Checkout without cashiers (Amazon Go).
- **Automotive:** Autonomous vehicles.
- **Security:** Facial recognition, Anomaly detection.
- **Agriculture:** Crop monitoring with drones.
- **Social networks:** Augmented reality filters.

📹 **Videos recommended:**

1. [Computer Vision Explained - IBM](https://www.youtube.com/watch?v=OcycT1Jwsns) - 10 min
1. [CS231n Lecture 1 - Stanford](https://www.youtube.com/watch?v=vT1JzLTH4G4) - 1 hour (full cursor available)

______________________________________________________________________

## 2. Digital Image Fundamentals

### Mathematical representation

- **Image = matrix of Pixels.**
- **Grayscale:** 2D matrix (height × width). Each value: intensity (0-255).
- **RGB:** 3D matrix (height × width × 3 Channels). Each Channel: red, green, blue.

**Example:**

```
Image RGB de 224×224 = tensor de forma (224, 224, 3)
```

### Resolution

- Higher resolution = more detail but higher computational cost.
- Trade-off between quality and efficiency.

### Common formats

- **JPEG:** lossy compression (photos).
- **PNG:** lossless compression (graphics, transparency).
- **TIFF:** without compression (medical images).

📹 **Videos recommended:**

1. [Digital Images Explained - Computerphile](https://www.youtube.com/watch?v=15aqFQQVBWU) - 10 min

______________________________________________________________________

## 3. Image Preprocessing

### Resize and Rescale

**Resize:** Change dimensions (ex: 1024×768 → 224×224).
**Rescale:** Normalize Pixel values.

**Common techniques:**

- Divide by 255: values ​​of [0, 255] → [0, 1].

- Normalize with ImageNet mean and std:

  ```python
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  ```

### Data Augmentation

Generate variations of Images to increase dataset and improve generalization.

**Common transformations:**

- Rotation (eg: ±15°).
- Flip horizontal/vertical.
- Zoom in/out.
- Brightness/contrast changes.
- Random cuts (random crops).
- Add noise.

**Caution:** Apply only to train set, not to test.

📹 **Videos recommended:**

1. [Image Preprocessing - Python Crash Course](https://www.youtube.com/watch?v=qH0dktiZmcg) - 20 min
1. [Data Augmentation - Krish Naik](https://www.youtube.com/watch?v=mTVf7BN7S8w) - 25 min

📚 **Resources written:**

- [Keras ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [Albumentations Library](https://albumentations.ai/docs/) - advanced augmentations

______________________________________________________________________

## 4. Convolutional Neural Networks (CNNs)

### Why CNNs?

MLPs (redes fully-connected) about Images:

- Too many parameters (Image 224×224×3 = 150,000 entries).
- They do not exploit spatial structure.

CNNs explode:

- **Locality:** Nearby pixels are related.
- **Translation invariance:** a cat remains a cat no matter where it is in the Image.

### Convolution operation

Apply **Filter/kernel** by sliding it over the Image.

**Example:**
Edge detection filter:

```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

### Components of a CNN

#### 1. Convolutional layers

- They learn Filters automatically.
- Parameters: number of Filters, kernel size (3×3, 5×5).

#### 2. Activation Functions

- ReLU after each convolutional Layer.

#### 3. Pooling

Reduce spatial dimensionality while maintaining relevant information.

**Max Pooling:** Take maximum value from each region.
**Average Pooling:** Take average.

**Example:**

```
Input 4×4 → Max Pooling 2×2 → Output 2×2
```

#### 4. Layers fully-connected (Dense)

At the end of the network for Classification.

#### 5. softmax

Output layer for multiclass Classification.

### Typical architecture

```
Input (224×224×3) →
  Conv (32 filters, 3×3) + ReLU →
  MaxPooling (2×2) →
  Conv (64 filters, 3×3) + ReLU →
  MaxPooling (2×2) →
  Conv (128 filters, 3×3) + ReLU →
  MaxPooling (2×2) →
  Flatten →
  Dense (128 neurons, ReLU) →
  Dropout (0.5) →
  Dense (10 classes, Softmax)
```

📹 **Videos recommended (FUNDAMENTAL):**

1. [CNNs Explained - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - part of their DL series
1. [Convolutional Neural Networks - Stanford CS231n](https://www.youtube.com/watch?v=bNb2fEVKeEo) - 1 hour
1. [CNNs Visualized - Computerphile](https://www.youtube.com/watch?v=py5byOOHZM8) - 15 min

📚 **Resources written:**

- [CS231n Convolutional Networks](http://cs231n.github.io/convolutional-networks/)
- [A Comprehensive Guide to CNNs](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

______________________________________________________________________

## 5. Classic CNN architectures

### LeNet-5 (1998)

First successful CNN (digit recognition).

### AlexNet (2012)

ImageNet won, popularized deep learning in vision.

### VGGNet (2014)

Very deep network (16-19 Layers), Small Filters (3×3).

### ResNet (2015)

**Innovation:** Skip connections (residual connections).

- It allows training very deep networks (50, 101, 152 Layers).
- Solve problem of vanishing gradients.

### Inception (GoogLeNet)

Multiple sizes of Filters in parallel.

### EfficientNet

Efficient scaling of width, depth and resolution.

📹 **Videos recommended:**

1. [CNN Architectures - Andrew Ng](https://www.youtube.com/watch?v=dXB-KQYkzNU) - 15 min
1. [ResNet Explained - Yannic Kilcher](https://www.youtube.com/watch?v=0tBPSxioIZE) - 20 min

______________________________________________________________________

## 6. Transfer Learning

**Concept:** Use a pre-trained network on a massive dataset (ImageNet) and adapt it to your Problem.

### Why does it work?

Initial layers learn generic features (edges, textures).
Final layers learn task-specific features.

### Strategies

#### 1. Feature Extraction

- Freeze the convolutional layers there.
- Train only the final Layers (Classifier).
- **Usage:** You have little Data and your task is similar to ImageNet.

#### 2. Fine-tuning

- Freeze first Layers.
- Train latest Convolutional Layers + Classifier.
- **Usage:** You have more Data or your task differs from ImageNet.

### Popular pre-trained models

- ResNet50, VGG16, EfficientNet, MobileNet (for edge devices).

📹 **Videos recommended:**

1. [Transfer Learning - Andrew Ng](https://www.youtube.com/watch?v=FQM13HkEfBk) - 10 min
1. [Transfer Learning in Practice - Krish Naik](https://www.youtube.com/watch?v=BqqfQnyjmgg) - 30 min

📚 **Resources written:**

- [Transfer Learning Guide - TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

______________________________________________________________________

## 7. Common tasks in Computer Vision

### Image Classification

Assign a tag to the entire Image.
**Example:** Cat vs Dog.

### Object Detection

Detect multiple objects in Image + location (bounding boxes).
**Models:** YOLO, Faster R-CNN, RetinaNet.

### Image Segmentation

Classify each pixel.

- **Semantic Segmentation:** tag clause, class (all cars = "car").
- **Instance Segmentation:** differentiate instances (auto1, auto2).
  **Models:** U-Net, Mask R-CNN.

### Pose Estimation

Detect position of human joints (keypoints).

### Image Generation

Generate realistic Images.
**Models:** GANs, Diffusion Models (Stable Diffusion).

📹 **Videos recommended:**

1. [Object Detection Overview - Paperspace](https://www.youtube.com/watch?v=O3b8lVF93jU) - 20 min
1. [YOLO Explained - Papers with Code](https://www.youtube.com/watch?v=9s_FpMpdYW8) - 15 min

______________________________________________________________________

## 8. Evaluation Metrics

### Classification

- **accuracy:** % of correct Predictions.
- **Precision, recall, f1:** when there is an imbalance of classes.
- **Top-k accuracy:** if the correct class is in the top-k Predictions.

### Object Detection

- **IoU (Intersection over Union):** overlap between predicted and actual bounding box.
- **mAP (mean Average Precision):** Standard metric in detection.

### Segmentation

- **Says Coefficient / f1-Score per pixel.**
- **IoU per clause, class.**

📹 **Videos recommended:**

1. [mAP Explained - Papers with Code](https://www.youtube.com/watch?v=FppOzcDvaDI) - 12 min

______________________________________________________________________

## 9. Common errors and solutions

### Imbalanced dataset

**Symptom:** Model always predicts the majority class.
**Solution:**

- Clause weights, class (class weights).
- Oversampling of clause, minority class.
- Focal Loss (penalizes easy Examples).

### overfitting

**Symptom:** High accuracy in train, low in test.
**Solution:**

- Data augmentation.
- Dropout.
- Regularization L2.
- Early stopping.
- Use more Data.

### Data Leakage

**Example:** Use test images in data augmentation or Normalization.
**Prevention:** Apply transformations only on train.

### Model muy lento

**Solution:**

- Reduce input resolution.
- Use more efficient architecture (MobileNet, EfficientNet).
- Quantization (int8 instead of float32).
- Pruning (pruning less important weights).

______________________________________________________________________

## 10. Buenas Practices

- ✅ Start with a simple baseline (Model pre-trained with transfer learning).
- ✅ Use GPU for Training (free Google Colab).
- ✅ Split Data reproducibly (set random seed).
- ✅ Apply augmentation only on train.
- ✅ Visualize Model Predictions (especially Errors).
- ✅ Analyze Confusion matrix to detect problematic classes.
- ✅ Save Model checkpoints during Training.
- ✅ Monitor loss and accuracy in train vs validation.
- ✅ Do not adjust multiple Hyperparameters at the same time (change one, measure impact).

📚 **General resources:**

- [Stanford CS231n (Course)](http://cs231n.stanford.edu/) - cursor complete with videos, Notes, assignments
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) - free practical course
- [Deep Learning for Computer Vision (Book)](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)

______________________________________________________________________

## Final comprehension checklist

Before moving to the next Module, you should be able to:

- ✅ Explain why CNNs outperform MLPs in vision tasks.
- ✅ Describe Convolution and pooling operations clearly.
- ✅ Implement a simple CNN with PyTorch or TensorFlow.
- ✅ Apply transfer learning using pre-trained Model.
- ✅ Decide when to use data augmentation and what transformations to apply.
- ✅ Detect overfitting in Training curves.
- ✅ Analyze Model Errors with Confusion matrix and Visualization.
- ✅ Differentiate Classification, object detection and segmentation.
- ✅ Manage unbalanced datasets with appropriate techniques.

If you answered "yes" to all, you are ready for advanced architectures and specialized CV applications.
