# Theory — Computer Vision

## Why this module matters

Computer Vision permite a las máquinas "ver" y comprender el mundo visual. Desde diagnóstico médico hasta vehículos autónomos, pasando por reconocimiento facial y moderación de Content, CV es fundamental en aplicaciones modernas de IA.

______________________________________________________________________

## 1. ¿Qué es Computer Vision?

**Computer Vision (CV)** es el área de IA que permite a los sistemas interpretar y analizar **Images y videos** para detectar patrones, objetos, acciones o eventos.

### Aplicaciones reales

- **Salud:** Detección de tumores en rayos X.
- **Retail:** Checkout sin cajeros (Amazon Go).
- **Automotriz:** Vehículos autónomos.
- **Seguridad:** Reconocimiento facial, detección de Anomalies.
- **Agricultura:** Monitoreo de cultivos con drones.
- **Redes sociales:** Filters de realidad aumentada.

📹 **Videos recomendados:**

1. [Computer Vision Explained - IBM](https://www.youtube.com/watch?v=OcycT1Jwsns) - 10 min
1. [CS231n Lecture 1 - Stanford](https://www.youtube.com/watch?v=vT1JzLTH4G4) - 1 hora (curso completo disponible)

______________________________________________________________________

## 2. Fundamentos de Image digital

### Representación matemática

- **Image = matriz de Pixels.**
- **Escala de grises:** matriz 2D (altura × ancho). Cada valor: intensidad (0-255).
- **RGB:** matriz 3D (altura × ancho × 3 Channels). Cada Channel: rojo, verde, azul.

**Example:**

```
Imagen RGB de 224×224 = tensor de forma (224, 224, 3)
```

### Resolución

- Mayor resolución = más detalle pero mayor costo computacional.
- Trade-off entre calidad y eficiencia.

### Formatos comunes

- **JPEG:** compresión con pérdida (fotos).
- **PNG:** compresión sin pérdida (gráficos, transparencia).
- **TIFF:** sin compresión (Images médicas).

📹 **Videos recomendados:**

1. [Digital Images Explained - Computerphile](https://www.youtube.com/watch?v=15aqFQQVBWU) - 10 min

______________________________________________________________________

## 3. Preprocesamiento de Images

### Resize y Rescale

**Resize:** Cambiar dimensiones (ej: 1024×768 → 224×224).
**Rescale:** Normalizar valores de Pixels.

**Técnicas comunes:**

- Dividir por 255: valores de [0, 255] → [0, 1].

- Normalizar con media y std de ImageNet:

  ```python
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  ```

### Data Augmentation

Generar variaciones de Images para aumentar dataset y mejorar generalización.

**Transformaciones comunes:**

- Rotación (ej: ±15°).
- Flip horizontal/vertical.
- Zoom in/out.
- Cambios de brillo/contraste.
- Recortes aleatorios (random crops).
- Añadir ruido.

**Cuidado:** Aplicar solo a train set, no a test.

📹 **Videos recomendados:**

1. [Image Preprocessing - Python Crash Course](https://www.youtube.com/watch?v=qH0dktiZmcg) - 20 min
1. [Data Augmentation - Krish Naik](https://www.youtube.com/watch?v=mTVf7BN7S8w) - 25 min

📚 **Resources escritos:**

- [Keras ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [Albumentations Library](https://albumentations.ai/docs/) - augmentations avanzadas

______________________________________________________________________

## 4. Convolutional Neural Networks (CNNs)

### ¿Por qué CNNs?

MLPs (redes fully-connected) sobre Images:

- Demasiados parámetros (Image 224×224×3 = 150,000 entradas).
- No explotan Structure espacial.

CNNs explotan:

- **Localidad:** Pixels cercanos están relacionados.
- **Invarianza a traslación:** un gato sigue siendo gato sin importar dónde esté en la Image.

### Operación de Convolution

Aplicar **Filter/kernel** deslizándolo sobre la Image.

**Example:**
Filter de detección de bordes:

```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

### Componentes de una CNN

#### 1. Layers convolucionales

- Aprenden Filters automáticamente.
- Parámetros: número de Filters, tamaño de kernel (3×3, 5×5).

#### 2. Functions de Activation

- ReLU después de cada Layer convolucional.

#### 3. Pooling

Reducir dimensionalidad espacial manteniendo información relevante.

**Max Pooling:** Tomar valor máximo de cada región.
**Average Pooling:** Tomar promedio.

**Example:**

```
Entrada 4×4 → Max Pooling 2×2 → Salida 2×2
```

#### 4. Layers fully-connected (Dense)

Al final de la red para Classification.

#### 5. Softmax

Layer de salida para Classification multiclase.

### Arquitectura típica

```
Entrada (224×224×3) →
  Conv (32 filtros, 3×3) + ReLU →
  MaxPooling (2×2) →
  Conv (64 filtros, 3×3) + ReLU →
  MaxPooling (2×2) →
  Conv (128 filtros, 3×3) + ReLU →
  MaxPooling (2×2) →
  Flatten →
  Dense (128 neuronas, ReLU) →
  Dropout (0.5) →
  Dense (10 clases, Softmax)
```

📹 **Videos recomendados (FUNDAMENTALES):**

1. [CNNs Explained - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - parte de su serie de DL
1. [Convolutional Neural Networks - Stanford CS231n](https://www.youtube.com/watch?v=bNb2fEVKeEo) - 1 hora
1. [CNNs Visualized - Computerphile](https://www.youtube.com/watch?v=py5byOOHZM8) - 15 min

📚 **Resources escritos:**

- [CS231n Convolutional Networks](http://cs231n.github.io/convolutional-networks/)
- [A Comprehensive Guide to CNNs](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

______________________________________________________________________

## 5. Arquitecturas clásicas de CNNs

### LeNet-5 (1998)

Primera CNN exitosa (reconocimiento de dígitos).

### AlexNet (2012)

Ganó ImageNet, popularizó deep learning en visión.

### VGGNet (2014)

Red muy profunda (16-19 Layers), Filters pequeños (3×3).

### ResNet (2015)

**Innovación:** Skip connections (conexiones residuales).

- Permite entrenar redes muy profundas (50, 101, 152 Layers).
- Resuelve Problem de vanishing gradients.

### Inception (GoogLeNet)

Múltiples tamaños de Filters en paralelo.

### EfficientNet

Escalado eficiente de ancho, profundidad y resolución.

📹 **Videos recomendados:**

1. [CNN Architectures - Andrew Ng](https://www.youtube.com/watch?v=dXB-KQYkzNU) - 15 min
1. [ResNet Explained - Yannic Kilcher](https://www.youtube.com/watch?v=0tBPSxioIZE) - 20 min

______________________________________________________________________

## 6. Transfer Learning

**Concept:** Usar red pre-entrenada en dataset masivo (ImageNet) y adaptarla a tu Problem.

### ¿Por qué funciona?

Layers iniciales aprenden features genéricas (bordes, texturas).
Layers finales aprenden features específicas de la tarea.

### Estrategias

#### 1. Feature Extraction

- Congelar todas las Layers convolucionales.
- Entrenar solo las Layers finales (Classifier).
- **Usage:** Tienes pocos Data y tu tarea es similar a ImageNet.

#### 2. Fine-tuning

- Congelar primeras Layers.
- Entrenar últimas Layers convolucionales + Classifier.
- **Usage:** Tienes más Data o tu tarea difiere de ImageNet.

### Models pre-entrenados populares

- ResNet50, VGG16, EfficientNet, MobileNet (para edge devices).

📹 **Videos recomendados:**

1. [Transfer Learning - Andrew Ng](https://www.youtube.com/watch?v=FQM13HkEfBk) - 10 min
1. [Transfer Learning in Practice - Krish Naik](https://www.youtube.com/watch?v=BqqfQnyjmgg) - 30 min

📚 **Resources escritos:**

- [Transfer Learning Guide - TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

______________________________________________________________________

## 7. Tareas comunes en Computer Vision

### Classification de Image

Asignar una etiqueta a toda la Image.
**Example:** Gato vs Perro.

### Object Detection

Detectar múltiples objetos en Image + ubicación (bounding boxes).
**Models:** YOLO, Faster R-CNN, RetinaNet.

### Image Segmentation

Clasificar cada píxel.

- **Semantic Segmentation:** etiquetar clase (todos los autos = "auto").
- **Instance Segmentation:** diferenciar instancias (auto1, auto2).
  **Models:** U-Net, Mask R-CNN.

### Pose Estimation

Detectar posición de articulaciones humanas (keypoints).

### Image Generation

Generar Images realistas.
**Models:** GANs, Diffusion Models (Stable Diffusion).

📹 **Videos recomendados:**

1. [Object Detection Overview - Paperspace](https://www.youtube.com/watch?v=O3b8lVF93jU) - 20 min
1. [YOLO Explained - Papers with Code](https://www.youtube.com/watch?v=9s_FpMpdYW8) - 15 min

______________________________________________________________________

## 8. Metrics de Evaluation

### Classification

- **accuracy:** % de Predictions correctas.
- **Precision, recall, f1:** cuando hay desbalance de clases.
- **Top-k accuracy:** si la clase correcta está en las top-k Predictions.

### Object Detection

- **IoU (Intersection over Union):** overlap entre bounding box predicho y real.
- **mAP (mean Average Precision):** Metric estándar en detection.

### Segmentación

- **Dice Coefficient / f1-Score por píxel.**
- **IoU por clase.**

📹 **Videos recomendados:**

1. [mAP Explained - Papers with Code](https://www.youtube.com/watch?v=FppOzcDvaDI) - 12 min

______________________________________________________________________

## 9. Errors comunes y soluciones

### Dataset desbalanceado

**Síntoma:** Model predice siempre la clase mayoritaria.
**Solución:**

- Pesos de clase (class weights).
- Oversampling de clase minoritaria.
- Focal Loss (penaliza Examples fáciles).

### overfitting

**Síntoma:** Alta accuracy en train, baja en test.
**Solución:**

- Data augmentation.
- Dropout.
- Regularization L2.
- Early stopping.
- Usar más Data.

### Data Leakage

**Example:** Usar test images en data augmentation o Normalization.
**Prevención:** Aplicar transformaciones solo en train.

### Model muy lento

**Solución:**

- Reducir resolución de entrada.
- Usar arquitectura más eficiente (MobileNet, EfficientNet).
- Cuantización (int8 en lugar de float32).
- Pruning (podar pesos menos importantes).

______________________________________________________________________

## 10. Buenas Practices

- ✅ Empezar con baseline simple (Model pre-entrenado con transfer learning).
- ✅ Usar GPU para Training (Google Colab gratuito).
- ✅ Dividir Data de forma reproducible (fijar random seed).
- ✅ Aplicar augmentation solo en train.
- ✅ Visualizar Predictions del Model (especialmente Errors).
- ✅ Analizar Confusion matrix para detectar clases problemáticas.
- ✅ Guardar checkpoints del Model durante Training.
- ✅ Monitorear loss y accuracy en train vs validation.
- ✅ No ajustar múltiples Hyperparameters a la vez (cambiar uno, medir impacto).

📚 **Resources generales:**

- [Stanford CS231n (Course)](http://cs231n.stanford.edu/) - curso completo con videos, Notes, assignments
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) - curso práctico gratuito
- [Deep Learning for Computer Vision (Book)](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente Module, deberías poder:

- ✅ Explicar por qué CNNs superan a MLPs en tareas de visión.
- ✅ Describir operaciones de Convolution y pooling con claridad.
- ✅ Implementar una CNN simple con PyTorch o TensorFlow.
- ✅ Aplicar transfer learning usando Model pre-entrenado.
- ✅ Decidir cuándo usar data augmentation y qué transformaciones aplicar.
- ✅ Detectar overfitting en curvas de Training.
- ✅ Analizar Errors del Model con Confusion matrix y Visualization.
- ✅ Diferenciar Classification, object detection y segmentación.
- ✅ Manejar datasets desbalanceados con técnicas apropiadas.

Si respondiste "sí" a todas, estás listo para architecturas avanzadas y aplicaciones especializadas de CV.
