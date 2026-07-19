"""Эталонное решение КИМ 2.1 — Backprop и обучение сети (NumPy + PyTorch)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, sol, solution_header

nb = Notebook("КИМ 2.1 — эталон (NumPy + PyTorch)")
nb.add(solution_header("КИМ 2.1. Backprop и обучение сети", "kim-02-backprop-training.ipynb"))

nb.add(md("""В этом эталоне:
- **Часть А** — backprop **вручную на чистом NumPy** (педагогический стержень):
  увидеть, как градиенты реально текут через вычислительный граф.
- **Часть Б, В** — та же сеть и регуляризация на **PyTorch** (автодифф)."""))

# === Часть А. Backprop на NumPy ===
nb.add(md("---\n## Часть А. Backprop на чистом NumPy (обязательно)"))
nb.add(md("""Архитектура: $z_1 = W_1 x + b_1,\\ a_1 = \\mathrm{ReLU}(z_1),\\ z_2 = W_2 a_1 + b_2,\\ \\hat{y} = \\mathrm{softmax}(z_2)$.
Loss — кросс-энтропия: $L = -\\sum_k y_k \\log \\hat{y}_k$."""))

nb.add(md("### 0. Импорт и подмножество Fashion-MNIST"))
nb.add(sol("""import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
np.random.seed(42)

# Загрузка через torchvision (без transforms — берём сырые numpy)
train_ds = datasets.FashionMNIST(root='./data', train=True, download=True)
x_full = train_ds.data.numpy().astype('float32') / 255.0   # (60000, 28, 28)
y_full = np.array(train_ds.targets)

N = 10000  # подмножество для скорости ручного backprop
idx = np.random.choice(60000, N, replace=False)
X = x_full[idx].reshape(N, 784)
y = np.eye(10)[y_full[idx]]   # one-hot

split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(X_train.shape, y_train.shape)"""))

nb.add(md("### 1. Инициализация параметров (He для ReLU)"))
nb.add(sol("""def init_params():
    W1 = np.random.randn(784, 64) * np.sqrt(2.0 / 784)
    b1 = np.zeros(64)
    W2 = np.random.randn(64, 10) * np.sqrt(2.0 / 64)
    b2 = np.zeros(10)
    return W1, b1, W2, b2

W1, b1, W2, b2 = init_params()"""))

nb.add(md("### 2. Прямой проход"))
nb.add(sol("""def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)   # стабилизация численного порядка
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1        # (B, 64)
    a1 = relu(z1)           # (B, 64)
    z2 = a1 @ W2 + b2       # (B, 10)
    y_hat = softmax(z2)     # (B, 10)
    return z1, a1, z2, y_hat"""))

nb.add(md("### 3. Функция потерь — кросс-энтропия"))
nb.add(sol("""def loss(y_hat, y):
    eps = 1e-12
    B = y.shape[0]
    return -np.sum(y * np.log(y_hat + eps)) / B"""))

nb.add(md("""### 4. Обратный проход (backprop)

Для softmax + cross-entropy совместно: $\\partial L / \\partial z_2 = \\hat{y} - y$.
Дальше — правило цепи по графу."""))
nb.add(sol("""def backward(X, y, z1, a1, y_hat, W2):
    B = X.shape[0]
    dz2 = (y_hat - y) / B                   # (B, 10)
    dW2 = a1.T @ dz2                        # (64, 10)
    db2 = np.sum(dz2, axis=0)               # (10,)
    da1 = dz2 @ W2.T                        # (B, 64)
    dz1 = da1 * (z1 > 0)                    # производная ReLU
    dW1 = X.T @ dz1                         # (784, 64)
    db1 = np.sum(dz1, axis=0)               # (64,)
    return dW1, db1, dW2, db2"""))

nb.add(md("### 5. Цикл обучения с мини-батчами"))
nb.add(sol("""def iterate_minibatches(X, y, batch_size, shuffle=True):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), batch_size):
        yield X[idx[i:i+batch_size]], y[idx[i:i+batch_size]]

lr = 0.1
epochs = 30
batch_size = 64

train_losses, val_losses, val_accs = [], [], []
for epoch in range(epochs):
    for xb, yb in iterate_minibatches(X_train, y_train, batch_size):
        z1, a1, z2, y_hat = forward(xb, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(xb, yb, z1, a1, y_hat, W2)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    _, _, _, yh_tr = forward(X_train, W1, b1, W2, b2)
    _, _, _, yh_va = forward(X_val, W1, b1, W2, b2)
    train_losses.append(loss(yh_tr, y_train))
    val_losses.append(loss(yh_va, y_val))
    val_accs.append(np.mean(np.argmax(yh_va, axis=1) == np.argmax(y_val, axis=1)))

    if (epoch + 1) % 5 == 0:
        print(f'Эпоха {epoch+1:2d}: train_loss={train_losses[-1]:.4f}  '
              f'val_loss={val_losses[-1]:.4f}  val_acc={val_accs[-1]:.4f}')"""))

nb.add(md("### 6. Графики loss и accuracy"))
nb.add(sol("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(train_losses, label='train'); ax[0].plot(val_losses, label='val')
ax[0].set_title('Loss'); ax[0].set_xlabel('эпоха'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(val_accs, label='val')
ax[1].set_title('Val accuracy'); ax[1].set_xlabel('эпоха'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()"""))

nb.add(md("""**Ответ (правило цепи):** backprop основан на правиле цепи
$\\frac{\\partial L}{\\partial W_1} = \\frac{\\partial L}{\\partial \\hat{y}}
\\cdot \\frac{\\partial \\hat{y}}{\\partial z_2} \\cdot \\frac{\\partial z_2}{\\partial a_1}
\\cdot \\frac{\\partial a_1}{\\partial z_1} \\cdot \\frac{\\partial z_1}{\\partial W_1}$.
Для softmax+cross-entropy первые два сомножителя сворачиваются в $\\hat{y} - y$
(производная по $z_2$). Дальше: $\\partial z_2 / \\partial a_1 = W_2^T$,
$\\partial a_1 / \\partial z_1 = \\mathbb{1}[z_1 > 0]$ (ReLU),
$\\partial z_1 / \\partial W_1 = X^T$. Итого: $\\partial L / \\partial W_1 =
X^T \\cdot ((\\hat{y} - y) W_2^T) \\odot \\mathbb{1}[z_1 > 0]$ — что и реализовано."""))

# === Часть Б. PyTorch ===
nb.add(md("""---
## Часть Б. Цикл обучения на PyTorch

Та же двухслойная сеть на PyTorch — backprop выполняется автоматически через
`loss.backward()` (автодифференцирование вычислительного графа)."""))
nb.add(md("### 7. Импорт PyTorch и данные"))
nb.add(sol("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print('Устройство:', device)
torch.manual_seed(42)

# PyTorch принимает индексы классов (не one-hot) для CrossEntropyLoss
X_tr_t = torch.tensor(X_train, dtype=torch.float32)
y_tr_t = torch.tensor(y_train.argmax(axis=1), dtype=torch.long)
X_va_t = torch.tensor(X_val, dtype=torch.float32)
y_va_t = torch.tensor(y_val.argmax(axis=1), dtype=torch.long)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)"""))

nb.add(md("### 8. Модель, loss, оптимизатор"))
nb.add(sol("""model = nn.Sequential(
    nn.Linear(784, 64), nn.ReLU(),
    nn.Linear(64, 10),
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)"""))

nb.add(md("### 9. Цикл обучения PyTorch"))
nb.add(sol("""def train_pytorch(model, loader, criterion, optimizer, device, epochs=20):
    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()       # (1) обнулить градиенты
            logits = model(x)           # (2) прямой проход
            loss = criterion(logits, y) # (3) вычислить loss
            loss.backward()             # (4) backprop (автоматический)
            optimizer.step()            # (5) обновить веса
            total += loss.item() * x.size(0)
        losses.append(total / len(loader.dataset))
    return losses

losses = train_pytorch(model, train_loader, criterion, optimizer, device, epochs=20)
plt.plot(losses); plt.xlabel('эпоха'); plt.ylabel('loss'); plt.title('PyTorch SGD'); plt.grid(True); plt.show()"""))

nb.add(md("### 10. Сравнение GD и SGD (разные batch_size)"))
nb.add(sol("""batch_sizes = [10, 50, 200, 500]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for bs in batch_sizes:
    torch.manual_seed(42)
    m = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    losses = train_pytorch(m, loader, criterion, opt, device, epochs=15)
    ax[0].plot(losses, label=f'bs={bs}')
    m.eval()
    with torch.no_grad():
        preds = m(X_va_t.to(device)).argmax(1).cpu()
        acc = (preds == y_va_t).float().mean().item()
    print(f'batch_size={bs:4d}  val_acc={acc:.4f}')
ax[0].set_title('Train loss'); ax[0].set_xlabel('эпоха'); ax[0].legend(); ax[0].grid(True)
plt.tight_layout(); plt.show()
# Вывод: малый batch_size (10, 50) даёт более быструю сходимость по эпохе,
# но с большим шумом; большой batch (500) — медленнее, но более гладко."""))

# === Часть В. Переобучение и регуляризация ===
nb.add(md("---\n## Часть В. Переобучение и регуляризация\n### 11. Обнаружение переобучения"))
nb.add(sol("""torch.manual_seed(42)
m_overfit = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
).to(device)
opt = optim.Adam(m_overfit.parameters(), lr=1e-3)

tr_losses, va_losses = [], []
for epoch in range(50):
    m_overfit.train()
    total_tr = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(m_overfit(x), y)
        loss.backward(); opt.step()
        total_tr += loss.item() * x.size(0)
    tr_losses.append(total_tr / len(train_ds))

    m_overfit.eval()
    with torch.no_grad():
        Xv, yv = X_va_t.to(device), y_va_t.to(device)
        va_loss = criterion(m_overfit(Xv), yv).item()
    va_losses.append(va_loss)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(tr_losses, label='train'); ax[0].plot(va_losses, label='val')
ax[0].set_title('Loss: переобучение (val растёт)'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(tr_losses, label='train'); ax[1].plot(va_losses, label='val')
ax[1].set_yscale('log'); ax[1].set_title('Loss (log) — разрыв train/val')
ax[1].legend(); ax[1].grid(True); plt.tight_layout(); plt.show()"""))

nb.add(md("### 12. Регуляризация: Dropout + WeightDecay + ранняя остановка"))
nb.add(sol("""import copy

torch.manual_seed(42)
m_reg = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 10),
).to(device)
opt = optim.Adam(m_reg.parameters(), lr=1e-3, weight_decay=1e-4)  # L2-регуляризация

best_val = float('inf')
best_state = None
patience, bad_epochs = 8, 0
tr_losses_r, va_losses_r = [], []

for epoch in range(50):
    m_reg.train()
    total_tr = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(m_reg(x), y)
        loss.backward(); opt.step()
        total_tr += loss.item() * x.size(0)
    tr_losses_r.append(total_tr / len(train_ds))

    m_reg.eval()
    with torch.no_grad():
        va_loss = criterion(m_reg(X_va_t.to(device)), y_va_t.to(device)).item()
    va_losses_r.append(va_loss)

    # Early stopping
    if va_loss < best_val:
        best_val = va_loss
        best_state = copy.deepcopy(m_reg.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f'Early stopping на эпохе {epoch+1}, лучшая val_loss = {best_val:.4f}')
            m_reg.load_state_dict(best_state)
            break

plt.figure(figsize=(10, 5))
plt.plot(tr_losses, label='train (без регул.)', color='C0', alpha=0.5)
plt.plot(va_losses, label='val (без регул.)', color='C1', alpha=0.5)
plt.plot(tr_losses_r[:len(tr_losses_r)], label='train (Dropout+L2+ES)', color='C0')
plt.plot(va_losses_r[:len(va_losses_r)], label='val (Dropout+L2+ES)', color='C1', linestyle='--')
plt.xlabel('эпоха'); plt.ylabel('loss'); plt.title('Регуляризация подавляет переобучение')
plt.legend(); plt.grid(True); plt.show()"""))

nb.add(md("""**Вывод:** `Dropout(0.5)` + `weight_decay` (L2) + ранняя остановка
эффективно подавляют переобучение: val_loss перестаёт расти, расхождение train/val
уменьшается, а ранняя остановка экономит эпохи обучения, восстанавливая лучшие веса."""))

path = "M2-training/attachments/kim-02-backprop-training-solution.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
