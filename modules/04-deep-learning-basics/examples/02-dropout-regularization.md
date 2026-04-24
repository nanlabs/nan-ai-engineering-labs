# Example 02 — Dropout y Regularization para Combatir overfitting

## Contexto

Detectaste overfitting en el Example 01 (train acc 99.3%, test acc 97.9%). Aprenderás a usar **Dropout**, **L2 regularization** y **Early Stopping** para mejorar generalización.

## Objective

Reducir el gap entre train y test accuracy implementando técnicas de Regularization.

______________________________________________________________________

## 🔄 Comparación: Model sin vs con Regularization

Entrenaremos 3 versiones:

1. **Baseline:** Red simple (Example 01)
1. **Con Dropout:** Agregar dropout
1. **Con Dropout + L2:** Dropout + weight decay

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

# Cargar MNIST (igual que Ejemplo 01)
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

## 🏗️ Paso 2: Definir arquitecturas

### Model 1: Baseline (sin Regularization)

```python
class BaselineNN(nn.Module):
    """Modelo sin regularización (Ejemplo 01)"""
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

### Model 2: Con Dropout

```python
class DropoutNN(nn.Module):
    """Modelo con Dropout"""
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
        x = self.dropout(x)  # 👈 Aplicar dropout después de ReLU
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # 👈 Aplicar dropout
        x = self.fc3(x)
        return x
```

**¿Qué es Dropout?**

- Durante Training: Desactiva aleatoriamente el `p%` de Neurons (ej: p=0.5 = 50%)
- Durante Evaluation: Usa todas las Neurons (pero escala outputs por `1-p`)
- **Efecto:** Previene co-adaptación de Neurons → reduce overfitting

______________________________________________________________________

## 🧪 Paso 3: Function de Training con Early Stopping

```python
def train_with_validation(model, train_loader, test_loader, criterion, optimizer,
                          num_epochs=20, patience=5, model_name="Model"):
    """
    Entrenar con early stopping

    patience: cuántas épocas esperar sin mejora antes de parar
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

            # Métricas
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # === EVALUACIÓN ===
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
            best_model_weights = copy.deepcopy(model.state_dict())  # Guardar mejores pesos
            epochs_without_improvement = 0
            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% ✅ (best)")
        else:
            epochs_without_improvement += 1
            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Parar si no mejora
        if epochs_without_improvement >= patience:
            print(f"⚠️ Early stopping en época {epoch+1} (sin mejora por {patience} épocas)")
            break

    # Restaurar mejores pesos
    model.load_state_dict(best_model_weights)

    return train_losses, train_accs, test_losses, test_accs
```

______________________________________________________________________

## 🏋️ Paso 4: Entrenar los 3 Models

### Model 1: Baseline

```python
print("=== MODELO 1: BASELINE (Sin Regularización) ===\n")

model1 = BaselineNN().to(device)
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

history1 = train_with_validation(
    model1, train_loader, test_loader, criterion1, optimizer1,
    num_epochs=20, patience=5, model_name="Baseline"
)

train_losses1, train_accs1, test_losses1, test_accs1 = history1
```

**Salida esperada:**

```
=== MODELO 1: BASELINE (Sin Regularización) ===

[Baseline] Epoch 1/20 | Train Loss: 0.3156, Train Acc: 90.72% | Test Loss: 0.1542, Test Acc: 95.24% ✅ (best)
[Baseline] Epoch 2/20 | Train Loss: 0.1398, Train Acc: 95.84% | Test Loss: 0.1124, Test Acc: 96.62% ✅ (best)
...
[Baseline] Epoch 10/20 | Train Loss: 0.0234, Train Acc: 99.32% | Test Loss: 0.0856, Test Acc: 97.89% ✅ (best)
[Baseline] Epoch 11/20 | Train Loss: 0.0189, Train Acc: 99.54% | Test Loss: 0.0892, Test Acc: 97.81%
...
⚠️ Early stopping en época 15 (sin mejora por 5 épocas)
```

### Model 2: Con Dropout

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

**Salida esperada:**

```
=== MODELO 2: CON DROPOUT ===

[Dropout] Epoch 1/20 | Train Loss: 0.4823, Train Acc: 85.12% | Test Loss: 0.1892, Test Acc: 94.32% ✅ (best)
[Dropout] Epoch 2/20 | Train Loss: 0.2134, Train Acc: 93.67% | Test Loss: 0.1234, Test Acc: 96.21% ✅ (best)
...
[Dropout] Epoch 12/20 | Train Loss: 0.0678, Train Acc: 97.89% | Test Loss: 0.0723, Test Acc: 98.12% ✅ (best)
```

**Observación:** Train acc más baja (esperado con dropout), pero test acc más alta

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

**¿Qué es Weight Decay?**

- Penaliza pesos grandes agregando término `λ * ||w||²` a loss
- Equivalente a L2 regularization en optimizadores de PyTorch
- **Efecto:** Pesos más pequeños → Model más simple → menos overfitting

______________________________________________________________________

## 📊 Paso 5: Comparar Results

### 5.1 Tabla de Metrics finales

```python
import pandas as pd

results = pd.DataFrame({
    'Modelo': ['Baseline', 'Dropout (p=0.5)', 'Dropout + L2'],
    'Train Acc (%)': [train_accs1[-1], train_accs2[-1], train_accs3[-1]],
    'Test Acc (%)': [test_accs1[-1], test_accs2[-1], test_accs3[-1]],
    'Train Loss': [train_losses1[-1], train_losses2[-1], train_losses3[-1]],
    'Test Loss': [test_losses1[-1], test_losses2[-1], test_losses3[-1]],
})

results['Gap (Train - Test)'] = results['Train Acc (%)'] - results['Test Acc (%)']
print(results.to_string(index=False))
```

**Salida esperada:**

```
            Modelo  Train Acc (%)  Test Acc (%)  Train Loss  Test Loss  Gap (Train - Test)
          Baseline          99.32         97.89       0.0234     0.0856                1.43
  Dropout (p=0.5)          97.89         98.12       0.0678     0.0723                -0.23  ⬅️ Generalization!
       Dropout + L2          97.45         98.24       0.0712     0.0689                -0.79  ⬅️ Mejor!
```

**📊 Interpretación:**

- **Baseline:** Gap positivo (+1.43%) → overfitting
- **Dropout:** Gap negativo (-0.23%) → buena generalización
- **Dropout + L2:** Gap más negativo (-0.79%) → mejor generalización + test acc más alto

### 5.2 Gráficas comparativas

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

**Observaciones visuales:**

- **Train Loss:** Models regularizados tienen loss más alto (esperado)
- **Test Loss:** Models regularizados tienen loss más bajo (mejor generalización)
- **Train Acc:** Baseline llega a 99% rápido; regularizados más lentos pero estables
- **Test Acc:** Regularizados superan baseline (~98.2% vs 97.9%)

______________________________________________________________________

## 🔬 Paso 6: Experimento con diferentes dropout rates

```python
# Probar múltiples valores de dropout
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

# Visualizar resultados
df_dropout = pd.DataFrame(results_dropout)
print("\n=== Comparación de Dropout Rates ===")
print(df_dropout.to_string(index=False))

# Gráfica
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

**Salida esperada:**

```
=== Comparación de Dropout Rates ===
 Dropout Rate  Train Acc  Test Acc    Gap
          0.2      98.89     98.01   0.88
          0.3      98.34     98.17   0.17
          0.5      97.45     98.24  -0.79  ⬅️ Sweet spot
          0.7      95.12     97.34  -2.22  ⬅️ Demasiado dropout
```

**Conclusion:** Dropout = 0.5 ofrece mejor balance

______________________________________________________________________

## 📝 Resumen ejecutivo

### ✅ Comparación final

| Metric              | Baseline | Dropout | Dropout + L2 | Mejora |
| ------------------- | -------- | ------- | ------------ | ------ |
| **Test Acc**        | 97.89%   | 98.12%  | **98.24%**   | +0.35% |
| **overfitting Gap** | +1.43%   | -0.23%  | **-0.79%**   | ✅     |
| **Test Loss**       | 0.0856   | 0.0723  | **0.0689**   | -19.5% |

### 🎯 Técnicas implementadas

1. **Dropout (p=0.5):**

   - Desactiva 50% de Neurons aleatoriamente durante Training
   - Reduce co-adaptación de Neurons
   - **Result:** +0.23% en test acc

1. **L2 Regularization (weight_decay=1e-4):**

   - Penaliza pesos grandes
   - Simplifica Model
   - **Result:** +0.12% adicional en test acc

1. **Early Stopping (patience=5):**

   - Detiene Training cuando test loss deja de mejorar
   - Previene overtraining
   - **Result:** Ahorra ~5-7 épocas

______________________________________________________________________

## 🎓 Lessons aprendidas

### ✅ Dropout

**Funcionamiento:**

```python
# Durante TRAIN
x = [1.0, 2.0, 3.0, 4.0]
Dropout(p=0.5) → [0.0, 4.0, 0.0, 8.0]  # 50% desactivado, restantes escalados por 1/(1-p)

# Durante EVAL
x = [1.0, 2.0, 3.0, 4.0]
Dropout(p=0.5) → [1.0, 2.0, 3.0, 4.0]  # Sin dropout (pero implícitamente escalado)
```

**Dónde aplicar:**

- ✅ Después de activaciones (ReLU, Tanh)
- ✅ En Layers fully connected
- ❌ No aplicar en última Layer (antes de output)

**Valores típicos:**

- p=0.2-0.3: Layers tempranas (features más simples)
- p=0.5: Layers profundas (features complejas)
- p=0.7-0.8: Demasiado agresivo (underfitting)

### ✅ Weight Decay / L2 Regularization

**Matemática:**

```
Loss_regularized = Loss_original + λ * Σ(w²)
```

**En PyTorch:**

```python
# weight_decay = λ
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

**Valores típicos:**

- 1e-4 a 1e-3: Regularization moderada
- 1e-5: Regularization ligera
- 1e-2: Regularization agresiva (puede causar underfitting)

### ✅ Early Stopping

**Criterios:**

- **Patience:** Número de épocas sin mejora antes de parar
- **Metric:** Validation loss (no accuracy)
- **Checkpoint:** Guardar mejores pesos, restaurar al final

**Implementación:**

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        break  # Parar entrenamiento
```

### 🚫 Errors comunes

- ❌ **Olvidar `model.eval()`:** Dropout activo durante Evaluation → Results incorrectos
- ❌ **Dropout en última Layer:** Añade ruido innecesario a Predictions
- ❌ **Weight decay muy alto:** underfitting (Model demasiado simple)
- ❌ **Early stopping con train loss:** Siempre usar validation loss

### 💡 Mejoras adicionales

1. **Batch Normalization:** Normalizar activaciones entre Layers
1. **Data Augmentation:** Rotaciones, traslaciones, ruido
1. **Ensemble:** Combinar múltiples Models
1. **Learning Rate Scheduling:** Reducir lr durante Training

______________________________________________________________________

## 🔧 Código completo para producción

```python
# Configuración óptima basada en experimentos
best_model = DropoutNN(dropout_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=0.001, weight_decay=1e-4)

# Entrenar con early stopping
history = train_with_validation(
    best_model, train_loader, test_loader, criterion, optimizer,
    num_epochs=20, patience=5, model_name="Production"
)

# Guardar modelo
torch.save(best_model.state_dict(), 'mnist_regularized.pth')

# Evaluar
best_model.eval()
with torch.no_grad():
    # ... inferencia
```

### 📌 Checklist de Regularization

- ✅ Usar dropout (p=0.5) en Layers intermedias
- ✅ Agregar weight decay (1e-4) al optimizador
- ✅ Implementar early stopping (patience=5)
- ✅ Validar con `model.eval()` / `model.train()`
- ✅ Monitorear train/test gap (debe ser < 1%)
- ✅ Experimentar con diferentes dropout rates
- ✅ Guardar mejores pesos durante early stopping
