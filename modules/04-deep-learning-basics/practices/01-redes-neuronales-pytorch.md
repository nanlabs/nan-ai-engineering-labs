# Practice 01 — Construction and Training of neural networks

## 🎯 Objectives

- Build neural networks with PyTorch from scratch
- Implement forward pass and backpropagation
- Train Models with different optimizers
- Evaluate and validate performance

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: Simple Neural network with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_split import train_test_split

# Dataset binario
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert a tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Define architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Instanciar model
model = SimpleNN(input_size=20, hidden_size=64, output_size=1)

print("=== Arquitectura ===")
print(model)

# Contar parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
```

**✅ Solution - Training Loop:**

```python
# Loss y optimizador
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Forward pass
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

# Visualize losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', alpha=0.7)
plt.plot(test_losses, label='Test Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_losses.png', dpi=150)
plt.show()

# Accuracy
model.eval()
with torch.no_grad():
    predictions = (model(X_test) > 0.5).float()
    accuracy = (predictions == y_test).float().mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Optimizadores Comparison

**Statement:**
Compare 3 optimizadores: SGD, Adam, RMSprop

- Train same Model with each one
- Grafica learning curves
- Compare convergence speed

### Exercise 2.2: Learning Rate Scheduling

**Statement:**
Implement `torch.optim.lr_scheduler.ReduceLROnPlateau`:

- Reduce LR when loss se estanca
- Visualize how LR changes during training

### Exercise 2.3: Early Stopping

**Statement:**
Implement early stopping:

- Stop training if val loss no improvement for N epochs
- Guarda better Model
- Restores better checkpoint weights

### Exercise 2.4: Batch Normalization

**Statement:**
Add `nn.BatchNorm1d` after each Layer hidden.
Compare:

- Convergence with vs without BatchNorm
- Gradient stability

### Exercise 2.5: Dropout for Regularization

**Statement:**
Add `nn.Dropout(p=0.5)` after ReLU.
Compare overfitting with vs without Dropout.

______________________________________________________________________

## ✅ Checklist

- [ ] Build architectures with `nn.Module`
- [ ] Implement forward pass
- [ ] Configure functions and optimizers
- [ ] Implement training loop complete
- [ ] Monitorear train/val losses
- [ ] Apply early stopping
- [ ] Wear learning rate scheduling
- [ ] Implement Regularization (Dropout, L2)

______________________________________________________________________

## 📚 Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch Book](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
