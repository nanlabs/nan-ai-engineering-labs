# Theory — Deep Learning Basics

## Why this module matters

Deep Learning has revolutionized AI in the last decade, enabling advances in computer vision, natural language processing, and generative systems. Understanding the fundamentals of neural networks allows you to work with modern architectures, fine-tune Models and diagnose Training Problems.

______________________________________________________________________

## 1. What is Deep Learning?

**Deep Learning** is a subset of Machine Learning based on **artificial neural networks with multiple Layers** ("deep") that learn hierarchical representations of Data.

### Why "deep"?

Deep Layers allow the Model to learn low-level features (edges, textures) in initial Layers and high-level features (objects, Concepts) in later Layers.

### Difference with traditional ML

- **ML traditional:** requires feature engineering manual.
- **Deep Learning:** learns features automatically from raw Data.

📹 **Videos recommended:**

1. [But what is a neural network? - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - 19 min (FUNDAMENTAL)
1. [Deep Learning Crash Course - freeCodeCamp](https://www.youtube.com/watch?v=VyWAvY2CF9c) - 30 min
1. [Deep Learning Specialization - Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning) - certifiable complete cursor

______________________________________________________________________

## 2. Architecture of a Neural network

### Neuron artificial (perceptron)

Basic unit that performs:

1. **Weighted sum:** `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
1. **Activation Function:** `a = σ(z)`

Where:

- `x`: inputs (features)
- `w`: weights (learned parameters)
- `b`: bias
- `σ`: Activation Function

### Layers of a network

- **Input layer:** receives the raw Data.
- **Hidden layers:** process and transform information.
- **Output layer:** produces the final Prediction.

### Typical architecture

```
Entrada (784 pixels) →
  Layer Oculta 1 (128 neurons, ReLU) →
  Layer Oculta 2 (64 neurons, ReLU) →
  Output (10 classes, Softmax)
```

______________________________________________________________________

## 3. Activation Functions

Activation Functions introduce **nonlinearity**, allowing the network to learn complex patterns.

### ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)
```

- **Usage:** Hidden layers (modern default).
- **Advantage:** simple, efficient, mitigates vanishing gradients.
- **Disadvantage:** "dying ReLU" (Dead Neurons with output always 0).

### Sigmoid

```
f(x) = 1 / (1 + e^(-x))
```

- **Usage:** Binary classification (Output Layer).
- **Range:** (0, 1) → interpretable as probability.
- **Disadvantage:** Vanishing gradients in deep Layers.

### Tanh

```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

- **Range:** (-1, 1).
- **Usage:** alternative to sigmoid in hidden Layers (less common today).

### softmax

```
f(xᵢ) = e^xᵢ / Σ(e^xⱼ)
```

- **Usage:** Multiclass Classification (Output Layer).
- **Property:** outputs add up to 1 (probability distribution).

📹 **Videos recommended:**

1. [Activation Functions Explained - StatQuest](https://www.youtube.com/watch?v=NkOv_k7r6no) - 13 min
1. [ReLU vs Sigmoid - Krish Naik](https://www.youtube.com/watch?v=4w3h6aPXKcQ) - 15 min

______________________________________________________________________

## 4. Forward Propagation

**Forward propagation** is the process of calculating the Prediction by passing Data through the network, Layer by Layer.

### Steps

1. Multiply entries by weights: `Z = W·X + b`
1. Apply Activation: `A = σ(Z)`
1. Repeat for each Layer until you reach the output.

📹 **Videos recommended:**

1. [Forward Propagation - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - already included above

______________________________________________________________________

## 5. Loss Functions

They measure **how bad** the Model is. The Objective of Training is to **minimize loss**.

### Mean Squared error (MSE) - Regression

```
MSE = (1/n) Σ (y_true - y_pred)²
```

### Binary Cross-Entropy - Binary Classification

```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### Categorical Cross-Entropy - Multiclass Classification

```
CCE = -Σ yᵢ·log(ŷᵢ)
```

📹 **Videos recommended:**

1. [Loss Functions - StatQuest](https://www.youtube.com/watch?v=Skc8nqJirJg) - 10 min
1. [Cross Entropy Explained - Aurélien Geron](https://www.youtube.com/watch?v=j_NJVqE8e9Y) - 15 min

______________________________________________________________________

## 6. Backpropagation

**Backpropagation** calculates the gradients (derivatives) of the loss with respect to each weight, propagating the error from the output to the input.

### Algorithm

1. Calculate loss in output.
1. Calculate gradient of the loss with respect to the output.
1. Propagate that gradient backwards using the chain rule.
1. Update weights: `w = w - learning_rate * gradient`

📹 **Videos recommended (CRITICS):**

1. [Backpropagation Calculus - 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8) - 10 min
1. [Backpropagation Explained - StatQuest](https://www.youtube.com/watch?v=IN2XmBhILt4) - 15 min

______________________________________________________________________

## 7. Optimizers

Algorithms that update weights based on gradients.

### Stochastic gradient Descent (SGD)

```
w = w - lr * gradient
```

- Simple and effective.
- It may converge slowly.

### SGD with Momentum

Accelerates convergence by accumulating speed in consistent directions.

### Adam (Adaptive Moment Estimation)

Combine momentum with adaptive learning rate per parameter.

- **Default recommended** for most cases.
- Typical hyperparameters: `lr=0.001`, `beta1=0.9`, `beta2=0.999`.

📹 **Videos recommended:**

1. [Gradient Descent - StatQuest](https://www.youtube.com/watch?v=sDv4f4s2SB8) - 9 min
1. [Optimizers Explained - Andrej Karpathy](https://www.youtube.com/watch?v=IHZwWFHWa-w) - 46 min

📚 **Resources written:**

- [An Overview of Gradient Descent Optimization](https://ruder.io/optimizing-gradient-descent/)
- [Adam Paper (original)](https://arxiv.org/abs/1412.6980)

______________________________________________________________________

## 8. Key hyperparameters

- **Learning rate:** step size in weight update.

  - Very high: divergence.
  - Very low: slow convergence.
- Typical: 0.001 (Adam), 0.01 (SGD).

- **Batch size:** number of Examples processed before updating weights.

  - Batch complete: stable but slow.
- Small batch: fast but noisy.
- Typical: 32, 64, 128.

- **Epochs:** number of times the Model sees the entire dataset.

  - Monitor to avoid overfitting.

- **Number of Layers and Neurons:**

- More depth → more capacity but more overfitting.
  - Start simple and add complexity if necessary.

______________________________________________________________________

## 9. Regularization and overfitting

### Dropout

During Training, it randomly "turns off" a % of Neurons in each iteration.

- Forces the network to not depend on specific Neurons.
- Typical: dropout=0.2 to 0.5.

### Early Stopping

Stop Training when Validation loss stops improving.

### Regularization L2 (Weight Decay)

Penalizes large weights by adding a term to the Loss Function.

### Data Augmentation

Generate Training Data variations (rotations, zoom, noise).

📹 **Videos recommended:**

1. [Regularization - Andrew Ng](https://www.youtube.com/watch?v=6g0t3Phly2M) - 10 min
1. [Dropout Explained - Krish Naik](https://www.youtube.com/watch?v=LN3qH5OI_lM) - 12 min

______________________________________________________________________

## 10. Deep Learning Frameworks

### PyTorch

- Pythonic, flexible.
- Research preferred.
- Dynamic computation graph.

### TensorFlow/Keras

- Keras: High level API (easy).
- TensorFlow: robust production.
- Static computation graph (TF 1.x), eager execution (TF 2.x).

### JAX

- Modern alternative, functional approach.

📹 **Videos recommended:**

1. [PyTorch Tutorial - freeCodeCamp](https://www.youtube.com/watch?v=V_xro1bcAuA) - 10 horas (complete)
1. [TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk) - 7 horas
1. [PyTorch vs TensorFlow - Krish Naik](https://www.youtube.com/watch?v=sVm2FPi_WHQ) - 20 min

📚 **Resources written:**

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow / Keras Guides](https://www.tensorflow.org/guide)
- [Fast.ai Course](https://course.fast.ai/) - free practical course

______________________________________________________________________

## 11. Good Training Practices

- ✅ Normalize/standardize input data.
- ✅ Start with simple architecture (baseline).
- ✅ Monitor loss and accuracy curves (train vs validation).
- ✅ Use stopping early to prevent overfitting.
- ✅ Save Model checkpoints during Training.
- ✅ Try multiple learning rates (lr schedule, lr finder).
- ✅ Visualize Model Predictions in difficult cases.
- ✅ Use GPU to accelerate Training (free Google Colab).

______________________________________________________________________

## Final comprehension checklist

Before moving to the next Module, you should be able to:

- ✅ Explain architecture of a Neural network (Layers, Neurons, activations).
- ✅ Differentiate forward propagation vs backpropagation clearly.
- ✅ Choose appropriate Activation Function according to Layer and Problem.
- ✅ Select correct loss function (MSE, BCE, CCE).
- ✅ Implement and train a simple Neural network with PyTorch or TensorFlow.
- ✅ Interpret Training curves and detect overfitting/underfitting.
- ✅ Apply Regularization techniques (dropout, early stopping).
- ✅ Adjust Hyperparameters (learning rate, batch size, epochs).
- ✅ Use GPU to accelerate Training.

If you answered "yes" to all, you are ready for advanced architectures (CNNs, RNNs, Transformers).
