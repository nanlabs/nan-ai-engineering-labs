# Theory — Deep Learning Basics

## Why this module matters

Deep Learning ha revolucionado IA en la última década, permitiendo avances en visión por computadora, procesamiento de lenguaje natural y sistemas generativos. Entender los fundamentos de redes neuronales te permite trabajar con arquitecturas modernas, fine-tunear modelos y diagnosticar problemas de entrenamiento.

______________________________________________________________________

## 1. ¿Qué es Deep Learning?

**Deep Learning** es un subconjunto de Machine Learning basado en **redes neuronales artificiales con múltiples capas** ("profundas") que aprenden representaciones jerárquicas de los datos.

### ¿Por qué "deep"?

Las capas profundas permiten al modelo aprender features de bajo nivel (bordes, texturas) en capas iniciales y features de alto nivel (objetos, conceptos) en capas posteriores.

### Diferencia con ML tradicional

- **ML tradicional:** requiere feature engineering manual.
- **Deep Learning:** aprende features automáticamente a partir de datos raw.

📹 **Videos recomendados:**

1. [But what is a neural network? - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - 19 min (FUNDAMENTAL)
1. [Deep Learning Crash Course - freeCodeCamp](https://www.youtube.com/watch?v=VyWAvY2CF9c) - 30 min
1. [Deep Learning Specialization - Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning) - curso completo certificable

______________________________________________________________________

## 2. Arquitectura de una red neuronal

### Neurona artificial (perceptron)

Unidad básica que realiza:

1. **Suma ponderada:** `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
1. **Función de activación:** `a = σ(z)`

Donde:

- `x`: entradas (features)
- `w`: pesos (parámetros aprendidos)
- `b`: sesgo/bias
- `σ`: función de activación

### Capas de una red

- **Capa de entrada:** recibe los datos raw.
- **Capas ocultas:** procesan y transforman información.
- **Capa de salida:** produce la predicción final.

### Arquitectura típica

```
Entrada (784 píxeles) →
  Capa Oculta 1 (128 neuronas, ReLU) →
  Capa Oculta 2 (64 neuronas, ReLU) →
  Salida (10 clases, Softmax)
```

______________________________________________________________________

## 3. Funciones de activación

Las funciones de activación introducen **no linealidad**, permitiendo a la red aprender patrones complejos.

### ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)
```

- **Uso:** capas ocultas (default moderno).
- **Ventaja:** simple, eficiente, mitiga vanishing gradients.
- **Desventaja:** "dying ReLU" (neuronas muertas con salida siempre 0).

### Sigmoid

```
f(x) = 1 / (1 + e^(-x))
```

- **Uso:** clasificación binaria (capa de salida).
- **Rango:** (0, 1) → interpretable como probabilidad.
- **Desventaja:** vanishing gradients en capas profundas.

### Tanh

```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

- **Rango:** (-1, 1).
- **Uso:** alternativa a sigmoid en capas ocultas (menos común hoy).

### Softmax

```
f(xᵢ) = e^xᵢ / Σ(e^xⱼ)
```

- **Uso:** clasificación multiclase (capa de salida).
- **Propiedad:** salidas suman 1 (distribución de probabilidad).

📹 **Videos recomendados:**

1. [Activation Functions Explained - StatQuest](https://www.youtube.com/watch?v=NkOv_k7r6no) - 13 min
1. [ReLU vs Sigmoid - Krish Naik](https://www.youtube.com/watch?v=4w3h6aPXKcQ) - 15 min

______________________________________________________________________

## 4. Forward Propagation

**Forward propagation** es el proceso de calcular la predicción pasando datos a través de la red, capa por capa.

### Pasos

1. Multiplicar entradas por pesos: `Z = W·X + b`
1. Aplicar activación: `A = σ(Z)`
1. Repetir para cada capa hasta llegar a la salida.

📹 **Videos recomendados:**

1. [Forward Propagation - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - ya incluido arriba

______________________________________________________________________

## 5. Funciones de pérdida (Loss Functions)

Miden **qué tan malo** es el modelo. El objetivo del entrenamiento es **minimizar la pérdida**.

### Mean Squared Error (MSE) - Regresión

```
MSE = (1/n) Σ (y_true - y_pred)²
```

### Binary Cross-Entropy - Clasificación binaria

```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### Categorical Cross-Entropy - Clasificación multiclase

```
CCE = -Σ yᵢ·log(ŷᵢ)
```

📹 **Videos recomendados:**

1. [Loss Functions - StatQuest](https://www.youtube.com/watch?v=Skc8nqJirJg) - 10 min
1. [Cross Entropy Explained - Aurélien Géron](https://www.youtube.com/watch?v=j_NJVqE8e9Y) - 15 min

______________________________________________________________________

## 6. Backpropagation

**Backpropagation** calcula los gradientes (derivadas) de la pérdida respecto a cada peso, propagando el error desde la salida hacia la entrada.

### Algoritmo

1. Calcular pérdida en la salida.
1. Calcular gradiente de la pérdida respecto a la salida.
1. Propagar ese gradiente hacia atrás usando la regla de la cadena.
1. Actualizar pesos: `w = w - learning_rate * gradient`

📹 **Videos recomendados (CRÍTICOS):**

1. [Backpropagation Calculus - 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8) - 10 min
1. [Backpropagation Explained - StatQuest](https://www.youtube.com/watch?v=IN2XmBhILt4) - 15 min

______________________________________________________________________

## 7. Optimizadores

Algoritmos que actualizan los pesos basándose en gradientes.

### Stochastic Gradient Descent (SGD)

```
w = w - lr * gradient
```

- Simple y efectivo.
- Puede converger lentamente.

### SGD with Momentum

Acelera convergencia acumulando velocidad en direcciones consistentes.

### Adam (Adaptive Moment Estimation)

Combinamomentum con tasa de aprendizaje adaptativa por parámetro.

- **Default recomendado** para la mayoría de casos.
- Hiperparámetros típicos: `lr=0.001`, `beta1=0.9`, `beta2=0.999`.

📹 **Videos recomendados:**

1. [Gradient Descent - StatQuest](https://www.youtube.com/watch?v=sDv4f4s2SB8) - 9 min
1. [Optimizers Explained - Andrej Karpathy](https://www.youtube.com/watch?v=IHZwWFHWa-w) - 46 min

📚 **Recursos escritos:**

- [An Overview of Gradient Descent Optimization](https://ruder.io/optimizing-gradient-descent/)
- [Adam Paper (original)](https://arxiv.org/abs/1412.6980)

______________________________________________________________________

## 8. Hiperparámetros clave

- **Learning rate:** tamaño del paso en actualización de pesos.

  - Muy alto: divergencia.
  - Muy bajo: convergencia lenta.
  - Típico: 0.001 (Adam), 0.01 (SGD).

- **Batch size:** número de ejemplos procesados antes de actualizar pesos.

  - Batch completo: estable pero lento.
  - Batch pequeño: rápido pero ruidoso.
  - Típico: 32, 64, 128.

- **Epochs:** número de veces que el modelo ve todo el dataset.

  - Monitorear para evitar overfitting.

- **Número de capas y neuronas:**

  - Más profundidad → más capacidad pero más overfitting.
  - Empezar simple y agregar complejidad si es necesario.

______________________________________________________________________

## 9. Regularización y Overfitting

### Dropout

Durante entrenamiento, "apaga" aleatoriamente un % de neuronas en cada iteración.

- Fuerza a la red a no depender de neuronas específicas.
- Típico: dropout=0.2 a 0.5.

### Early Stopping

Detener entrenamiento cuando la pérdida de validación deja de mejorar.

### Regularización L2 (Weight Decay)

Penaliza pesos grandes agregando término a la función de pérdida.

### Data Augmentation

Generar variaciones de datos de entrenamiento (rotaciones, zoom, ruido).

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

📚 **Recursos escritos:**

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow / Keras Guides](https://www.tensorflow.org/guide)
- [Fast.ai Course](https://course.fast.ai/) - curso práctico gratuito

______________________________________________________________________

## 11. Buenas prácticas de entrenamiento

- ✅ Normalizar/estandarizar datos de entrada.
- ✅ Empezar con arquitectura simple (baseline).
- ✅ Monitorear curvas de loss y accuracy (train vs validation).
- ✅ Usar early stopping para prevenir overfitting.
- ✅ Guardar checkpoints del modelo durante entrenamiento.
- ✅ Probar múltiples learning rates (lr schedule, lr finder).
- ✅ Visualizar predicciones del modelo en casos difíciles.
- ✅ Usar GPU para acelerar entrenamiento (Google Colab gratuito).

______________________________________________________________________

## Final comprehension checklist

Antes de pasar al siguiente módulo, deberías poder:

- ✅ Explicar arquitectura de una red neuronal (capas, neuronas, activaciones).
- ✅ Diferenciar forward propagation vs backpropagation con claridad.
- ✅ Elegir función de activación apropiada según capa y problema.
- ✅ Seleccionar función de pérdida correcta (MSE, BCE, CCE).
- ✅ Implementar y entrenar una red neuronal simple con PyTorch o TensorFlow.
- ✅ Interpretar curvas de entrenamiento y detectar overfitting/underfitting.
- ✅ Aplicar técnicas de regularización (dropout, early stopping).
- ✅ Ajustar hiperparámetros (learning rate, batch size, epochs).
- ✅ Usar GPU para acelerar entrenamiento.

Si respondiste "sí" a todas, estás listo para arquitecturas avanzadas (CNNs, RNNs, Transformers).
