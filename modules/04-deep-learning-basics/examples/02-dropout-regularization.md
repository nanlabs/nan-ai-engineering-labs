# Example 02 — Dropout and Regularization to Combat Overfitting

## Context

You detected overfitting in Example 01 (train acc 99.3%, test acc 97.9%). You will learn to use **Dropout**, **L2 regularization** and **Early Stopping** to improve generalization.

## Objective

Reduce the gap between train and test accuracy by implementing Regularization techniques.

______________________________________________________________________

## 🔄 Comparison: Model without vs with Regularization

We will train 3 versions:

1. **Baseline:** Red simple (Example 01)
1. **With Dropout:** Add dropout
1. **With Dropout + L2:** Dropout + weight decay

______________________________________________________________________

## 🚀 Paso 1: Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# Load MNIST (igual que Example 01)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

______________________________________________________________________

## 🏗️ Step 2: Define architectures

### Model 1: Baseline (sin Regularization)

```python
class BaselineNN(nn.Module):
    """Model sin regularization (Example 01)"""
    def __init__(self):
        super(BaselineNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Model 2: With Dropout

```python
class DropoutNN(nn.Module):
    """Model con Dropout"""
    def __init__(self, dropout_rate=0.5):
        super(DropoutNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # 👈 Dropout

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 👈 Apply dropout after de ReLU
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # 👈 Apply dropout
        x = self.fc3(x)
        return x
```

**What is Dropout?**

- During Training: Randomly disable the `p%` of Neurons (ex: p=0.5 = 50%)
- During Evaluation: Use all Neurons (but scale outputs by `1-p`)
- **Effect:** Prevents co-adaptation of Neurons → reduces overfitting

______________________________________________________________________

## 🧪 Step 3: Training Function with Early Stopping

```python
def train_with_validation(model, train_loader, test_loader, criterion, optimizer,
                          num_epochs=20, patience=5, model_name="Model"):
    """
    Train con early stopping

    patience: how many eras esperar sin improvement antes de parar
    """
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_test_loss = float('inf')
    best_model_weights = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # === ENTRENAMIENTO ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # === ASSESSMENT ===
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # === EARLY STOPPING ===
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Save mejores pesos
            epochs_without_improvement = 0
            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% ✅ (best)")
        else:
            epochs_without_improvement += 1
            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Parar si no improvement
        if epochs_without_improvement >= patience:
            print(f"⚠️ Early stopping en time {epoch+1} (sin improvement por {patience} eras)")
            break

    # Restaurar mejores pesos
    model.load_state_dict(best_model_weights)

    return train_losses, train_accs, test_losses, test_accs
```

______________________________________________________________________

## 🏋️ Step 4: Train the 3 Models

### Model 1: Baseline

```python
print("=== MODELO 1: BASELINE (Sin Regularization) ===\n")

model1 = BaselineNN().to(device)
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

history1 = train_with_validation(
    model1, train_loader, test_loader, criterion1, optimizer1,
    num_epochs=20, patience=5, model_name="Baseline"
)

train_losses1, train_accs1, test_losses1, test_accs1 = history1
```

**Output expected:**

```
=== MODELO 1: BASELINE (Sin Regularization) ===

[Baseline] Epoch 1/20 | Train Loss: 0.3156, Train Acc: 90.72% | Test Loss: 0.1542, Test Acc: 95.24% ✅ (best)
[Baseline] Epoch 2/20 | Train Loss: 0.1398, Train Acc: 95.84% | Test Loss: 0.1124, Test Acc: 96.62% ✅ (best)
...
[Baseline] Epoch 10/20 | Train Loss: 0.0234, Train Acc: 99.32% | Test Loss: 0.0856, Test Acc: 97.89% ✅ (best)
[Baseline] Epoch 11/20 | Train Loss: 0.0189, Train Acc: 99.54% | Test Loss: 0.0892, Test Acc: 97.81%
...
⚠️ Early stopping en time 15 (sin improvement por 5 eras)
```

### Model 2: With Dropout

```python
print("\n=== MODELO 2: CON DROPOUT ===\n")

model2 = DropoutNN(dropout_rate=0.5).to(device)
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

history2 = train_with_validation(
    model2, train_loader, test_loader, criterion2, optimizer2,
    num_epochs=20, patience=5, model_name="Dropout"
)

train_losses2, train_accs2, test_losses2, test_accs2 = history2
```

**Output expected:**

```
=== MODELO 2: CON DROPOUT ===

[Dropout] Epoch 1/20 | Train Loss: 0.4823, Train Acc: 85.12% | Test Loss: 0.1892, Test Acc: 94.32% ✅ (best)
[Dropout] Epoch 2/20 | Train Loss: 0.2134, Train Acc: 93.67% | Test Loss: 0.1234, Test Acc: 96.21% ✅ (best)
...
[Dropout] Epoch 12/20 | Train Loss: 0.0678, Train Acc: 97.89% | Test Loss: 0.0723, Test Acc: 98.12% ✅ (best)
```

**Remark:** Train acc lower (expected with dropout), but test acc higher

### Model 3: Dropout + L2 Regularization (Weight Decay)

```python
print("\n=== MODELO 3: DROPOUT + L2 REGULARIZATION ===\n")

model3 = DropoutNN(dropout_rate=0.5).to(device)
criterion3 = nn.CrossEntropyLoss()
optimizer3 = optim.Adam(model3.parameters(), lr=0.001, weight_decay=1e-4)  # 👈 L2

history3 = train_with_validation(
    model3, train_loader, test_loader, criterion3, optimizer3,
    num_epochs=20, patience=5, model_name="Dropout+L2"
)

train_losses3, train_accs3, test_losses3, test_accs3 = history3
```

**What is Weight Decay?**

- Penalizes large weights by adding term `λ * ||w||²` to loss
- Equivalent to L2 regularization in PyTorch optimizers
- **Effect:** Smaller weights → Simpler model → less overfitting

______________________________________________________________________

## 📊 Paso 5: Compare Results

### 5.1 Final Metrics Table

```python
import pandas as pd

results = pd.DataFrame({
    'Model': ['Baseline', 'Dropout (p=0.5)', 'Dropout + L2'],
    'Train Acc (%)': [train_accs1[-1], train_accs2[-1], train_accs3[-1]],
    'Test Acc (%)': [test_accs1[-1], test_accs2[-1], test_accs3[-1]],
    'Train Loss': [train_losses1[-1], train_losses2[-1], train_losses3[-1]],
    'Test Loss': [test_losses1[-1], test_losses2[-1], test_losses3[-1]],
})

results['Gap (Train - Test)'] = results['Train Acc (%)'] - results['Test Acc (%)']
print(results.to_string(index=False))
```

**Output expected:**

```
            Model  Train Acc (%)  Test Acc (%)  Train Loss  Test Loss  Gap (Train - Test)
          Baseline          99.32         97.89       0.0234     0.0856                1.43
  Dropout (p=0.5)          97.89         98.12       0.0678     0.0723                -0.23  ⬅️ Generalization!
       Dropout + L2          97.45         98.24       0.0712     0.0689                -0.79  ⬅️ Mejor!
```

**📊 Interpretation:**

- **Baseline:** Positive gap (+1.43%) → overfitting
- **Dropout:** Negative gap (-0.23%) → good generalization
- **Dropout + L2:** More negative gap (-0.79%) → better generalization + higher acc test

### 5.2 Comparative graphs

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs1 = range(1, len(train_losses1) + 1)
epochs2 = range(1, len(train_losses2) + 1)
epochs3 = range(1, len(train_losses3) + 1)

# Subplot 1: Train Loss
axes[0, 0].plot(epochs1, train_losses1, label='Baseline', marker='o')
axes[0, 0].plot(epochs2, train_losses2, label='Dropout', marker='s')
axes[0, 0].plot(epochs3, train_losses3, label='Dropout + L2', marker='^')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Train Loss')
axes[0, 0].set_title('Train Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Subplot 2: Test Loss
axes[0, 1].plot(epochs1, test_losses1, label='Baseline', marker='o')
axes[0, 1].plot(epochs2, test_losses2, label='Dropout', marker='s')
axes[0, 1].plot(epochs3, test_losses3, label='Dropout + L2', marker='^')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Test Loss')
axes[0, 1].set_title('Test Loss Comparison')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Subplot 3: Train Accuracy
axes[1, 0].plot(epochs1, train_accs1, label='Baseline', marker='o')
axes[1, 0].plot(epochs2, train_accs2, label='Dropout', marker='s')
axes[1, 0].plot(epochs3, train_accs3, label='Dropout + L2', marker='^')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Train Accuracy (%)')
axes[1, 0].set_title('Train Accuracy Comparison')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Subplot 4: Test Accuracy
axes[1, 1].plot(epochs1, test_accs1, label='Baseline', marker='o')
axes[1, 1].plot(epochs2, test_accs2, label='Dropout', marker='s')
axes[1, 1].plot(epochs3, test_accs3, label='Dropout + L2', marker='^')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Test Accuracy (%)')
axes[1, 1].set_title('Test Accuracy Comparison')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Visual observations:**

- **Train Loss:** Regularized models have higher loss (expected)
- **Test Loss:** Regularized models have lower losses (better generalization)
- **Train Acc:** Baseline reaches 99% fast; regularized slower but stable
- **Acc Test:** Regularized exceed baseline (~98.2% vs 97.9%)

______________________________________________________________________

## 🔬 Step 6: Experiment with different dropout rates

```python
# Probar multiple values de dropout
dropout_rates = [0.2, 0.3, 0.5, 0.7]
results_dropout = []

for rate in dropout_rates:
    print(f"\n=== Dropout Rate: {rate} ===")
    model_temp = DropoutNN(dropout_rate=rate).to(device)
    criterion_temp = nn.CrossEntropyLoss()
    optimizer_temp = optim.Adam(model_temp.parameters(), lr=0.001, weight_decay=1e-4)

    history_temp = train_with_validation(
        model_temp, train_loader, test_loader, criterion_temp, optimizer_temp,
        num_epochs=15, patience=5, model_name=f"Dropout-{rate}"
    )

    train_losses_temp, train_accs_temp, test_losses_temp, test_accs_temp = history_temp

    results_dropout.append({
        'Dropout Rate': rate,
        'Train Acc': train_accs_temp[-1],
        'Test Acc': test_accs_temp[-1],
        'Gap': train_accs_temp[-1] - test_accs_temp[-1]
    })

# Visualize results
df_dropout = pd.DataFrame(results_dropout)
print("\n=== Comparison de Dropout Rates ===")
print(df_dropout.to_string(index=False))

# Graph
plt.figure(figsize=(10, 6))
plt.plot(df_dropout['Dropout Rate'], df_dropout['Train Acc'], marker='o', label='Train Acc')
plt.plot(df_dropout['Dropout Rate'], df_dropout['Test Acc'], marker='s', label='Test Acc')
plt.xlabel('Dropout Rate')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Dropout Rate')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Output expected:**

```
=== Comparison de Dropout Rates ===
 Dropout Rate  Train Acc  Test Acc    Gap
          0.2      98.89     98.01   0.88
          0.3      98.34     98.17   0.17
          0.5      97.45     98.24  -0.79  ⬅️ Sweet spot
          0.7      95.12     97.34  -2.22  ⬅️ Demasiado dropout
```

**Conclusion:** Dropout = 0.5 offers better balance

______________________________________________________________________

## 📝 Executive summary

### ✅ Final comparison

| Metric | Baseline | Dropout | Dropout + L2 | Improve |
| ------------------- | -------- | ------- | ------------ | ------ |
| **Acc Test** | 97.89% | 98.12% | **98.24%** | +0.35% |
| **overfitting gap** | +1.43% | -0.23% | **-0.79%** | ✅ |
| **Test Loss** | 0.0856 | 0.0723 | **0.0689** | -19.5% |

### 🎯 Techniques implemented

1. **Dropout (p=0.5):**

   - Disable 50% of Neurons randomly during Training
- Reduces co-adaptation of Neurons
   - **Result:** +0.23% in acc test

1. **L2 Regularization (weight_decay=1e-4):**

   - Large pesos penalized
   - Simplify Model
   - **Result:** +0.12% additional in acc test

1. **Early Stopping (patience=5):**

   - Stop Training when test loss stops improving
   - Prevents overtraining
- **Result:** Save ~5-7 epochs

______________________________________________________________________

## 🎓 Lessons learned

### ✅ Dropout

**Operation:**

```python
# Durante TRAIN
x = [1.0, 2.0, 3.0, 4.0]
Dropout(p=0.5) → [0.0, 4.0, 0.0, 8.0]  # 50% desactivado, restantes escalados por 1/(1-p)

# Durante EVAL
x = [1.0, 2.0, 3.0, 4.0]
Dropout(p=0.5) → [1.0, 2.0, 3.0, 4.0]  # Sin dropout (but implicitly escalado)
```

**Where to apply:**

- ✅ After activations (ReLU, Tanh)
- ✅ In Layers fully connected
- ❌ Do not apply in the last Layer (before output)

**Typical values:**

- p=0.2-0.3: Early layers (simpler features)
- p=0.5: Deep layers (complex features)
- p=0.7-0.8: Too aggressive (underfitting)

### ✅ Weight Decay / L2 Regularization

**Mathematics:**

```
Loss_regularized = Loss_original + λ * Σ(w²)
```

**In PyTorch:**

```python
# weight_decay = λ
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

**Typical values:**

- 1e-4 to 1e-3: Moderate Regularization
- 1e-5: Light regularization
- 1e-2: Aggressive regularization (can cause underfitting)

### ✅ Early Stopping

**Criteria:**

- **Patience:** Number of epochs without improvement before stopping
- **Metric:** Validation loss (no accuracy)
- **Checkpoint:** Save best weights, restore at the end

**Implementation:**

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        break  # Parar training
```

### 🚫 Errors common

- ❌ **Forget `model.eval()`:** Dropout active during Evaluation → Incorrect results
- ❌ **Dropout in last Layer:** Adds unnecessary noise to Predictions
- ❌ **Weight decay too high:** underfitting (Model too simple)
- ❌ **Early stopping with train loss:** Always use validation loss

### 💡 Additional improvements

1. **Batch Normalization:** Normalize activations between Layers
1. **Data Augmentation:** Rotations, translations, noise
1. **Ensemble:** Combine multiple Models
1. **Learning Rate Scheduling:** Reduce lr during Training

______________________________________________________________________

## 🔧 Complete code for production

```python
# Configuration optimal basada en experiments
best_model = DropoutNN(dropout_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=0.001, weight_decay=1e-4)

# Train con early stopping
history = train_with_validation(
    best_model, train_loader, test_loader, criterion, optimizer,
    num_epochs=20, patience=5, model_name="Production"
)

# Save model
torch.save(best_model.state_dict(), 'mnist_regularized.pth')

# Evaluate
best_model.eval()
with torch.no_grad():
    # ... inferencia
```

### 📌 Regularization Checklist

- ✅ Use dropout (p=0.5) in intermediate layers
- ✅ Add weight decay (1e-4) to the optimizer
- ✅ Implement early stopping (patience=5)
- ✅ Validate with `model.eval()` / `model.train()`
- ✅ Monitor train/test gap (must be < 1%)
- ✅ Experiment with different dropout rates
- ✅ Save better pesos during early stopping
