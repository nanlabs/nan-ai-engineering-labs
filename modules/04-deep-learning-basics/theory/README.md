# Theory — Deep Learning Basics

## Why this module matters

Deep Learning ha revolucionado IA en la última década, permitiendo avances en visión por computadora, procesamiento de lenguaje natural y sistemas generativos. Entender los fundamentos de neural networks te permite trabajar con arquitecturas modernas, fine-tunear Models y diagnosticar Problems de Training.

______________________________________________________________________

## 1. ¿Qué es Deep Learning?

**Deep Learning** es un subconjunto de Machine Learning basado en **neural networks artificiales con múltiples Layers** ("profundas") que aprenden representaciones jerárquicas de los Data.

### ¿Por qué "deep"?

Las Layers profundas permiten al Model aprender features de bajo nivel (bordes, texturas) en Layers iniciales y features de alto nivel (objetos, Concepts) en Layers posteriores.

### Diferencia con ML tradicional

- **ML tradicional:** requiere feature engineering manual.
- **Deep Learning:** aprende features automáticamente a partir de Data raw.

📹 **Videos recomendados:**

1. [But what is a neural network? - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - 19 min (FUNDAMENTAL)
1. [Deep Learning Crash Course - freeCodeCamp](https://www.youtube.com/watch?v=VyWAvY2CF9c) - 30 min
1. [Deep Learning Specialization - Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning) - curso completo certificable

______________________________________________________________________

## 2. Arquitectura de una Neural network

### Neuron artificial (perceptron)

Unidad básica que realiza:

1. **Suma ponderada:** `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
1. **Function de Activation:** `a = σ(z)`

Donde:

- `x`: entradas (features)
- `w`: pesos (parámetros aprendidos)
- `b`: sesgo/bias
- `σ`: Function de Activation

### Layers de una red

- **Layer de entrada:** recibe los Data raw.
- **Layers ocultas:** procesan y transforman información.
- **Layer de salida:** produce la Prediction final.

### Arquitectura típica

```
Entrada (784 píxeles) →
  Capa Oculta 1 (128 neuronas, ReLU) →
  Capa Oculta 2 (64 neuronas, ReLU) →
  Salida (10 clases, Softmax)
```

______________________________________________________________________

## 3. Functions de Activation

Las Functions de Activation introducen **no linealidad**, permitiendo a la red aprender patrones complejos.

### ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)
```

- **Usage:** Layers ocultas (default moderno).
- **Ventaja:** simple, eficiente, mitiga vanishing gradients.
- **Desventaja:** "dying ReLU" (Neurons muertas con salida siempre 0).

### Sigmoid

```
f(x) = 1 / (1 + e^(-x))
```

- **Usage:** Classification binaria (Layer de salida).
- **Rango:** (0, 1) → interpretable como probabilidad.
- **Desventaja:** vanishing gradients en Layers profundas.

### Tanh

```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

- **Rango:** (-1, 1).
- **Usage:** alternativa a sigmoid en Layers ocultas (menos común hoy).

### Softmax

```
f(xᵢ) = e^xᵢ / Σ(e^xⱼ)
```

- **Usage:** Classification multiclase (Layer de salida).
- **Propiedad:** salidas suman 1 (distribución de probabilidad).

📹 **Videos recomendados:**

1. [Activation Functions Explained - StatQuest](https://www.youtube.com/watch?v=NkOv_k7r6no) - 13 min
1. [ReLU vs Sigmoid - Krish Naik](https://www.youtube.com/watch?v=4w3h6aPXKcQ) - 15 min

______________________________________________________________________

## 4. Forward Propagation

**Forward propagation** es el proceso de calcular la Prediction pasando Data a través de la red, Layer por Layer.

### Pasos

1. Multiplicar entradas por pesos: `Z = W·X + b`
1. Aplicar Activation: `A = σ(Z)`
1. Repetir para cada Layer hasta llegar a la salida.

📹 **Videos recomendados:**

1. [Forward Propagation - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - ya incluido arriba

______________________________________________________________________

## 5. Functions de pérdida (Loss Functions)

Miden **qué tan malo** es el Model. El Objective del Training es **minimizar la pérdida**.

### Mean Squared error (MSE) - Regresión

```
MSE = (1/n) Σ (y_true - y_pred)²
```

### Binary Cross-Entropy - Classification binaria

```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### Categorical Cross-Entropy - Classification multiclase

```
CCE = -Σ yᵢ·log(ŷᵢ)
```

📹 **Videos recomendados:**

1. [Loss Functions - StatQuest](https://www.youtube.com/watch?v=Skc8nqJirJg) - 10 min
1. [Cross Entropy Explained - Aurélien Géron](https://www.youtube.com/watch?v=j_NJVqE8e9Y) - 15 min

______________________________________________________________________

## 6. Backpropagation

**Backpropagation** calcula los gradientes (derivadas) de la pérdida respecto a cada peso, propagando el error desde la salida hacia la entrada.

### Algorithm

1. Calcular pérdida en la salida.
1. Calcular gradiente de la pérdida respecto a la salida.
1. Propagar ese gradiente hacia atrás usando la regla de la cadena.
1. Actualizar pesos: `w = w - learning_rate * gradient`

📹 **Videos recomendados (CRÍTICOS):**

1. [Backpropagation Calculus - 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8) - 10 min
1. [Backpropagation Explained - StatQuest](https://www.youtube.com/watch?v=IN2XmBhILt4) - 15 min

______________________________________________________________________

## 7. Optimizadores

Algorithms que actualizan los pesos basándose en gradientes.

### Stochastic gradient Descent (SGD)

```
w = w - lr * gradient
```

- Simple y efectivo.
- Puede converger lentamente.

### SGD with Momentum

Acelera convergencia acumulando velocidad en direcciones consistentes.

### Adam (Adaptive Moment Estimation)

Combinamomentum con tasa de Learning adaptativa por parámetro.

- **Default recomendado** para la mayoría de casos.
- Hyperparameters típicos: `lr=0.001`, `beta1=0.9`, `beta2=0.999`.

📹 **Videos recomendados:**

1. [Gradient Descent - StatQuest](https://www.youtube.com/watch?v=sDv4f4s2SB8) - 9 min
1. [Optimizers Explained - Andrej Karpathy](https://www.youtube.com/watch?v=IHZwWFHWa-w) - 46 min

📚 **Resources escritos:**

- [An Overview of Gradient Descent Optimization](https://ruder.io/optimizing-gradient-descent/)
- [Adam Paper (original)](https://arxiv.org/abs/1412.6980)

______________________________________________________________________

## 8. Hyperparameters clave

- **Learning rate:** tamaño del paso en actualización de pesos.

  - Muy alto: divergencia.
  - Muy bajo: convergencia lenta.
  - Típico: 0.001 (Adam), 0.01 (SGD).

- **Batch size:** número de Examples procesados antes de actualizar pesos.

  - Batch completo: estable pero lento.
  - Batch pequeño: rápido pero ruidoso.
  - Típico: 32, 64, 128.

- **Epochs:** número de veces que el Model ve todo el dataset.

  - Monitorear para evitar overfitting.

- **Número de Layers y Neurons:**

  - Más profundidad → más capacidad pero más overfitting.
  - Empezar simple y agregar complejidad si es necesario.

______________________________________________________________________

## 9. Regularization y overfitting

### Dropout

Durante Training, "apaga" aleatoriamente un % de Neurons en cada iteración.

- Fuerza a la red a no depender de Neurons específicas.
- Típico: dropout=0.2 a 0.5.

### Early Stopping

Detener Training cuando la pérdida de Validation deja de mejorar.

### Regularization L2 (Weight Decay)

Penaliza pesos grandes agregando término a la Function de pérdida.

### Data Augmentation

Generar variaciones de Data de Training (rotaciones, zoom, ruido).

📹 **Videos recomendados:**

1. [Regularization - Andrew Ng](https://www.youtube.com/watch?v=6g0t3Phly2M) - 10 min
1. [Dropout Explained - Krish Naik](https://www.youtube.com/watch?v=LN3qH5OI_lM) - 12 min

______________________________________________________________________

## 10. Frameworks de Deep Learning

### PyTorch

- Pythonic, flexible.
- Preferido en investigación.
- Dynamic computation graph.

### TensorFlow / Keras

- Keras: API de alto nivel (fácil).
- TensorFlow: producción robusta.
- Static computation graph (TF 1.x), eager execution (TF 2.x).

### JAX

- Alternativa moderna, enfoque funcional.

📹 **Videos recomendados:**

1. [PyTorch Tutorial - freeCodeCamp](https://www.youtube.com/watch?v=V_xro1bcAuA) - 10 horas (completo)
1. [TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk) - 7 horas
1. [PyTorch vs TensorFlow - Krish Naik](https://www.youtube.com/watch?v=sVm2FPi_WHQ) - 20 min

📚 **Resources escritos:**

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow / Keras Guides](https://www.tensorflow.org/guide)
- [Fast.ai Course](https://course.fast.ai/) - curso práctico gratuito

______________________________________________________________________

## 11. Buenas Practices de Training

- ✅ Normalizar/estandarizar Data de entrada.
- ✅ Empezar con arquitectura simple (baseline).
- ✅ Monitorear curvas de loss y accuracy (train vs validation).
- ✅ Usar early stopping para prevenir overfitting.
- ✅ Guardar checkpoints del Model durante Training.
- ✅ Probar múltiples learning rates (lr schedule, lr finder).
- ✅ Visualizar Predictions del Model en casos difíciles.
- ✅ Usar GPU para acelerar Training (Google Colab gratuito).

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente Module, deberías poder:

- ✅ Explicar arquitectura de una Neural network (Layers, Neurons, activaciones).
- ✅ Diferenciar forward propagation vs backpropagation con claridad.
- ✅ Elegir Function de Activation apropiada según Layer y Problem.
- ✅ Seleccionar Function de pérdida correcta (MSE, BCE, CCE).
- ✅ Implementar y entrenar una Neural network simple con PyTorch o TensorFlow.
- ✅ Interpretar curvas de Training y detectar overfitting/underfitting.
- ✅ Aplicar técnicas de Regularization (dropout, early stopping).
- ✅ Ajustar Hyperparameters (learning rate, batch size, epochs).
- ✅ Usar GPU para acelerar Training.

Si respondiste "sí" a todas, estás listo para arquitecturas avanzadas (CNNs, RNNs, Transformers).
