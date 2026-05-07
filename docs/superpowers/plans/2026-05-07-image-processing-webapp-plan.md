# 图像处理 Web 应用实现计划

> **给执行代理的说明：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐个任务实现。步骤使用 checkbox (`- [ ]`) 语法跟踪。

**目标：** 在 image7/ 中构建一个 Streamlit 图像处理应用，包含 AE vs VAE 对比、DCGAN 生成、文本到图像三个功能 Tab。

**架构：** Streamlit 单页应用，三个 st.tabs 分别承载三个功能模块。通过 session_manager 按 Tab 切换管理显存，每个 Tab 独立训练/加载模型。所有模型权重缓存到 checkpoints/ 目录。

**技术栈：** Python 3.13, PyTorch 2.11, Streamlit 1.56, diffusers, plotly, matplotlib

---

## 文件结构

```
image7/
├── app.py                    # 入口：st.tabs 路由到三个 Tab 函数
├── models/
│   ├── __init__.py
│   ├── autoencoder.py        # Autoencoder 类 + train_autoencoder()
│   ├── vae.py                # VAE 类 + train_vae()
│   ├── dcgan.py              # Generator, Discriminator + train_dcgan()
│   └── diffusion_loader.py   # load_sd_pipeline(), generate_image()
├── ui/
│   ├── __init__.py
│   ├── tab_ae_vae.py         # render_ae_vae_tab() — Tab 1 全部 UI
│   ├── tab_dcgan.py          # render_dcgan_tab() — Tab 2 全部 UI
│   └── tab_diffusion.py      # render_diffusion_tab() — Tab 3 全部 UI
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # get_mnist_data(), get_fashion_mnist_data()
│   ├── visualization.py      # plot_loss_curves(), plot_latent_scatter(), make_heatmap()
│   └── session_manager.py    # clear_models_for_tab(), get_device()
├── checkpoints/              # 训练权重缓存目录 (autoencoder.pt, vae.pt, dcgan_g.pt, dcgan_d.pt)
├── assets/                   # 预生成静态图目录
├── docs/
│   ├── specs/
│   └── superpowers/plans/
└── requirements.txt
```

---

### Task 1: 项目初始化与依赖

**文件：**
- 创建: `image7/requirements.txt`

- [ ] **Step 1: 创建 requirements.txt**

```
streamlit>=1.56.0
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.30.0
transformers>=4.40.0
accelerate>=0.30.0
plotly>=5.20.0
matplotlib>=3.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

- [ ] **Step 2: 安装依赖**

```powershell
cd d:\project\AIvibe\image7
pip install -r requirements.txt
```

- [ ] **Step 3: 创建目录结构**

```powershell
cd d:\project\AIvibe\image7
New-Item -ItemType Directory -Force -Path models, ui, utils, checkpoints, assets
New-Item -ItemType File -Force -Path models/__init__.py, ui/__init__.py, utils/__init__.py
```

- [ ] **Step 4: 验证环境**

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import streamlit; print('Streamlit:', streamlit.__version__); import diffusers; print('Diffusers OK')"
```

预期输出:
```
CUDA: True
Streamlit: 1.56.0
Diffusers OK
```

- [ ] **Step 5: 提交**

```bash
cd d:/project/AIvibe
git add image7/requirements.txt image7/models/ image7/ui/ image7/utils/ image7/checkpoints/ image7/assets/
git commit -m "feat: 初始化 image7 项目结构与依赖"
```

---

### Task 2: 数据加载工具

**文件：**
- 创建: `image7/utils/data_loader.py`

- [ ] **Step 1: 编写 data_loader.py**

```python
"""MNIST / Fashion-MNIST 数据加载工具。"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(dataset_name: str, batch_size: int = 128, train: bool = True) -> DataLoader:
    """获取指定数据集和批大小的 DataLoader。

    Args:
        dataset_name: 'mnist' 或 'fashion_mnist'
        batch_size: 批大小
        train: True 返回训练集，False 返回测试集

    Returns:
        DataLoader 实例
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == 'mnist':
        ds_cls = datasets.MNIST
    elif dataset_name == 'fashion_mnist':
        ds_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    dataset = ds_cls(
        root='./data',
        train=train,
        download=True,
        transform=transform,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)
    return loader


def get_test_images(dataset_name: str, n: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    """获取 n 张测试图像及其标签。

    Returns:
        (images, labels): images shape (n, 1, 28, 28), labels shape (n,)
    """
    loader = get_dataloader(dataset_name, batch_size=n, train=False)
    images, labels = next(iter(loader))
    return images[:n], labels[:n]


def get_full_test_dataset(dataset_name: str) -> torch.Tensor:
    """获取完整测试集，用于潜空间散点图。

    Returns:
        images, labels: (N, 1, 28, 28), (N,)
    """
    loader = get_dataloader(dataset_name, batch_size=10000, train=False)
    images, labels = next(iter(loader))
    return images, labels
```

- [ ] **Step 2: 验证数据加载**

```powershell
cd d:\project\AIvibe\image7
python -c "from utils.data_loader import get_dataloader, get_test_images; loader = get_dataloader('mnist', batch_size=128); x, y = next(iter(loader)); print(f'Batch shape: {x.shape}, labels: {y.shape}'); imgs, lbls = get_test_images('mnist', 25); print(f'Test images: {imgs.shape}')"
```

预期输出:
```
Batch shape: torch.Size([128, 1, 28, 28]), labels: torch.Size([128])
Test images: torch.Size([25, 1, 28, 28])
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/utils/data_loader.py
git commit -m "feat: 添加 MNIST/Fashion-MNIST 数据加载工具"
```

---

### Task 3: 显存管理工具

**文件：**
- 创建: `image7/utils/session_manager.py`

- [ ] **Step 1: 编写 session_manager.py**

```python
"""显存管理与设备工具。"""
import torch
import gc


def get_device() -> torch.device:
    """获取可用设备（CUDA > MPS > CPU）。"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def clear_cuda_memory() -> None:
    """清理 CUDA 显存缓存并触发垃圾回收。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_vram_usage() -> dict:
    """获取当前显存使用情况（仅 CUDA）。"""
    if not torch.cuda.is_available():
        return {'allocated_mb': 0, 'reserved_mb': 0, 'total_mb': 0}
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
    }
```

- [ ] **Step 2: 验证设备检测**

```powershell
cd d:\project\AIvibe\image7
python -c "from utils.session_manager import get_device, get_vram_usage; print('Device:', get_device()); print('VRAM:', get_vram_usage())"
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/utils/session_manager.py
git commit -m "feat: 添加显存管理与设备检测工具"
```

---

### Task 4: 可视化工具

**文件：**
- 创建: `image7/utils/visualization.py`

- [ ] **Step 1: 编写 visualization.py**

```python
"""可视化工具：loss 曲线、潜空间散点图、误差热力图。"""
import io
import numpy as np
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image


def plot_loss_curves(ae_train_losses: list[float], ae_val_losses: list[float],
                     vae_total_losses: list[float], vae_recon_losses: list[float],
                     vae_kl_losses: list[float]) -> go.Figure:
    """绘制 AE 与 VAE 的 loss 曲线对比图。

    Returns:
        Plotly Figure 对象
    """
    epochs = list(range(1, len(ae_train_losses) + 1))
    fig = go.Figure()

    # AE losses
    if ae_train_losses:
        fig.add_trace(go.Scatter(x=epochs, y=ae_train_losses, mode='lines',
                                 name='AE 训练 Loss', line=dict(color='#1f77b4')))
    if ae_val_losses:
        fig.add_trace(go.Scatter(x=epochs, y=ae_val_losses, mode='lines',
                                 name='AE 验证 Loss', line=dict(color='#1f77b4', dash='dash')))

    # VAE losses
    if vae_total_losses:
        fig.add_trace(go.Scatter(x=epochs, y=vae_total_losses, mode='lines',
                                 name='VAE 总 Loss', line=dict(color='#ff7f0e')))
    if vae_recon_losses:
        fig.add_trace(go.Scatter(x=epochs, y=vae_recon_losses, mode='lines',
                                 name='VAE 重构 Loss', line=dict(color='#2ca02c')))
    if vae_kl_losses:
        fig.add_trace(go.Scatter(x=epochs, y=vae_kl_losses, mode='lines',
                                 name='VAE KL Loss', line=dict(color='#d62728')))

    fig.update_layout(
        title='训练 Loss 曲线',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_latent_scatter(latent_vecs: np.ndarray, labels: np.ndarray,
                        class_names: list[str]) -> go.Figure:
    """绘制 2D 潜空间散点图，按类别着色。

    Args:
        latent_vecs: (N, 2) 编码向量
        labels: (N,) 类别标签
        class_names: 类别名称列表

    Returns:
        Plotly Figure 对象（带 click 事件）
    """
    fig = go.Figure()
    for cls_id in sorted(set(labels)):
        mask = labels == cls_id
        fig.add_trace(go.Scatter(
            x=latent_vecs[mask, 0],
            y=latent_vecs[mask, 1],
            mode='markers',
            name=class_names[cls_id],
            marker=dict(size=3, opacity=0.6),
            customdata=np.column_stack([np.arange(len(labels))[mask], labels[mask]]),
            hovertemplate='Index: %{customdata[0]}<br>Class: %{customdata[1]}<extra></extra>',
        ))

    fig.update_layout(
        title='潜空间分布 (2D)',
        xaxis_title='z₁',
        yaxis_title='z₂',
        height=500,
        legend=dict(itemsizing='constant'),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def make_error_heatmaps(originals: torch.Tensor, reconstructions: torch.Tensor,
                        n_cols: int = 5) -> Image.Image:
    """为一批图像生成重构误差热力图。

    Args:
        originals: (N, 1, 28, 28) 原始图像
        reconstructions: (N, 1, 28, 28) 重构图像
        n_cols: 每行的图像数量

    Returns:
        PIL Image，热力图网格
    """
    n = originals.shape[0]
    n_rows = (n + n_cols - 1) // n_cols

    errors = torch.abs(originals - reconstructions).cpu().numpy()
    errors = errors.squeeze(1)  # (N, 28, 28)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        if i < n:
            im = ax.imshow(errors[i], cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'err={errors[i].mean():.3f}', fontsize=7)
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def make_image_grid(images: torch.Tensor, n_cols: int = 5, vmin: float = 0.0,
                    vmax: float = 1.0) -> Image.Image:
    """将一批图像渲染为网格 PIL Image。

    Args:
        images: (N, 1, 28, 28) 或 (N, 3, H, W) 形状的张量
        n_cols: 每行列数
    """
    from torchvision.utils import make_grid as tv_make_grid
    grid = tv_make_grid(images, nrow=n_cols, normalize=False, pad_value=1.0)
    grid_np = grid.cpu().detach()
    if grid_np.shape[0] == 1:
        grid_np = grid_np.repeat(3, 1, 1)  # 灰度转 RGB
    grid_np = (grid_np.clamp(vmin, vmax) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(grid_np)
```

- [ ] **Step 2: 验证可视化函数**

```powershell
cd d:\project\AIvibe\image7
python -c "
from utils.visualization import plot_loss_curves, make_error_heatmaps
import torch
# 测试 loss 曲线
fig = plot_loss_curves([0.1, 0.05, 0.03], [0.12, 0.06, 0.04], [0.15, 0.08, 0.05], [0.1, 0.05, 0.03], [0.05, 0.03, 0.02])
print('Loss figure type:', type(fig).__name__)
# 测试热力图
x = torch.rand(10, 1, 28, 28)
r = torch.rand(10, 1, 28, 28)
img = make_error_heatmaps(x, r)
print('Heatmap size:', img.size)
print('OK')
"
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/utils/visualization.py
git commit -m "feat: 添加可视化工具（loss 曲线、热力图、图像网格）"
```

---

### Task 5: Autoencoder 模型与训练

**文件：**
- 创建: `image7/models/autoencoder.py`

- [ ] **Step 1: 编写 autoencoder.py**

```python
"""Autoencoder 模型定义与训练。"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.session_manager import get_device


class Autoencoder(nn.Module):
    """简单卷积自编码器。"""

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14×14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7×7
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 4×4
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 7×7
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 14×14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    # 28×28
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Returns:
            (reconstructed, latent_code)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    latent_dim: int = 2,
    epochs: int = 20,
    lr: float = 1e-3,
    progress_callback=None,
) -> tuple[Autoencoder, list[float], list[float]]:
    """训练 Autoencoder。

    Returns:
        (model, train_losses, val_losses)
    """
    device = get_device()
    model = Autoencoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                val_loss += criterion(x_hat, x).item() * x.size(0)
        val_losses.append(val_loss / len(val_loader.dataset))

        if progress_callback:
            progress_callback(epoch + 1, epochs, train_losses[-1], val_losses[-1])

    return model, train_losses, val_losses


def save_ae_checkpoint(model: Autoencoder, path: str) -> None:
    """保存 AE 权重。"""
    torch.save({'model_state_dict': model.state_dict(), 'latent_dim': model.latent_dim}, path)


def load_ae_checkpoint(path: str) -> Autoencoder:
    """加载 AE 权重。"""
    data = torch.load(path, map_location=get_device(), weights_only=True)
    model = Autoencoder(latent_dim=data['latent_dim'])
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model
```

- [ ] **Step 2: 快速训练测试**

```powershell
cd d:\project\AIvibe\image7
python -c "
from utils.data_loader import get_dataloader
from models.autoencoder import train_autoencoder
train_loader = get_dataloader('mnist', batch_size=256, train=True)
val_loader = get_dataloader('mnist', batch_size=256, train=False)
model, tl, vl = train_autoencoder(train_loader, val_loader, latent_dim=2, epochs=2, lr=1e-3)
print(f'Train losses: {tl}')
print(f'Val losses: {vl}')
print('OK - AE training works')
"
```

预期输出:
```
Train losses: [0.0xx, 0.0xx]
Val losses: [0.0xx, 0.0xx]
OK - AE training works
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/models/autoencoder.py
git commit -m "feat: 添加 Autoencoder 模型定义与训练函数"
```

---

### Task 6: VAE 模型与训练

**文件：**
- 创建: `image7/models/vae.py`

- [ ] **Step 1: 编写 vae.py**

```python
"""VAE 模型定义与训练。"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.session_manager import get_device


class VAE(nn.Module):
    """简化卷积 VAE。"""

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder — 输出 mu 和 log_var
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Returns:
            (reconstructed, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """从潜空间向量直接解码（推理用）。"""
        return self.decode(z)


def vae_loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 VAE loss。

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def train_vae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    latent_dim: int = 2,
    epochs: int = 20,
    lr: float = 1e-3,
    beta: float = 0.1,
    progress_callback=None,
) -> tuple[VAE, list[float], list[float], list[float], list[float]]:
    """训练 VAE。

    Returns:
        (model, total_losses, recon_losses, kl_losses, val_total_losses)
    """
    device = get_device()
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_losses, recon_losses, kl_losses, val_total_losses = [], [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        n_samples = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            total, recon, kl = vae_loss(x_hat, x, mu, logvar, beta)
            total.backward()
            optimizer.step()
            epoch_total += total.item() * x.size(0)
            epoch_recon += recon.item() * x.size(0)
            epoch_kl += kl.item() * x.size(0)
            n_samples += x.size(0)
        total_losses.append(epoch_total / n_samples)
        recon_losses.append(epoch_recon / n_samples)
        kl_losses.append(epoch_kl / n_samples)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                total, _, _ = vae_loss(x_hat, x, mu, logvar, beta)
                val_loss += total.item() * x.size(0)
        val_total_losses.append(val_loss / len(val_loader.dataset))

        if progress_callback:
            progress_callback(epoch + 1, epochs, total_losses[-1], val_total_losses[-1])

    return model, total_losses, recon_losses, kl_losses, val_total_losses


def save_vae_checkpoint(model: VAE, path: str) -> None:
    """保存 VAE 权重。"""
    torch.save({'model_state_dict': model.state_dict(), 'latent_dim': model.latent_dim}, path)


def load_vae_checkpoint(path: str) -> VAE:
    """加载 VAE 权重。"""
    data = torch.load(path, map_location=get_device(), weights_only=True)
    model = VAE(latent_dim=data['latent_dim'])
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model
```

- [ ] **Step 2: 快速训练测试**

```powershell
cd d:\project\AIvibe\image7
python -c "
from utils.data_loader import get_dataloader
from models.vae import train_vae
train_loader = get_dataloader('mnist', batch_size=256, train=True)
val_loader = get_dataloader('mnist', batch_size=256, train=False)
model, tl, rl, kl, vl = train_vae(train_loader, val_loader, latent_dim=2, epochs=2, lr=1e-3, beta=0.1)
print(f'Total losses: {tl}')
print(f'VAE training OK')
"
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/models/vae.py
git commit -m "feat: 添加 VAE 模型定义与训练函数"
```

---

### Task 7: Tab 1 — AE vs VAE 对比 UI

**文件：**
- 创建: `image7/ui/tab_ae_vae.py`

- [ ] **Step 1: 编写 tab_ae_vae.py**

```python
"""Tab 1: AE vs VAE 重构对比。"""
import os
import io
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from utils.data_loader import get_dataloader, get_test_images, get_full_test_dataset
from utils.visualization import (
    plot_loss_curves, plot_latent_scatter, make_error_heatmaps, make_image_grid
)
from utils.session_manager import get_device, clear_cuda_memory
from models.autoencoder import Autoencoder, train_autoencoder, save_ae_checkpoint, load_ae_checkpoint
from models.vae import VAE, train_vae, save_vae_checkpoint, load_vae_checkpoint

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
FASHION_CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def render_ae_vae_tab() -> None:
    """渲染 'AE vs VAE' Tab 的全部 UI。"""
    st.header("自编码器 vs VAE 重构对比")

    # ── 控制栏 ──
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        dataset_name = st.selectbox("数据集", ["mnist", "fashion_mnist"],
                                    key="ae_vae_dataset")
    with col_ctrl2:
        latent_dim = st.slider("潜空间维度", 2, 20, 2, key="ae_vae_latent_dim")
    with col_ctrl3:
        st.write("")  # spacing
        st.write("")
        train_btn = st.button("训练模型", key="ae_vae_train_btn", type="primary")

    class_names = CLASS_NAMES if dataset_name == 'mnist' else FASHION_CLASS_NAMES

    # ── 初始化 session state ──
    for key, default in [
        ('ae_model', None), ('vae_model', None),
        ('ae_train_losses', []), ('ae_val_losses', []),
        ('vae_total_losses', []), ('vae_recon_losses', []),
        ('vae_kl_losses', []), ('vae_val_total_losses', []),
        ('ae_vae_trained', False), ('test_images', None),
        ('test_labels', None), ('latent_scatter_data', None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    ae_ckpt = os.path.join(CHECKPOINT_DIR, f'ae_{dataset_name}_ld{latent_dim}.pt')
    vae_ckpt = os.path.join(CHECKPOINT_DIR, f'vae_{dataset_name}_ld{latent_dim}.pt')

    # ── 训练或加载 ──
    if train_btn:
        with st.status("训练中...", expanded=True) as status:
            train_loader = get_dataloader(dataset_name, batch_size=128, train=True)
            val_loader = get_dataloader(dataset_name, batch_size=256, train=False)

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def ae_callback(epoch, total, train_loss, val_loss):
                progress_bar.progress(epoch / (total * 2), text=f"AE Epoch {epoch}/{total}")
                status_text.text(f"AE — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            st.write("**训练 Autoencoder...**")
            ae_model, ae_train, ae_val = train_autoencoder(
                train_loader, val_loader, latent_dim=latent_dim, epochs=20, progress_callback=ae_callback
            )
            save_ae_checkpoint(ae_model, ae_ckpt)

            st.write("**训练 VAE...**")
            def vae_callback(epoch, total, train_loss, val_loss):
                progress_bar.progress((20 + epoch) / (total * 2), text=f"VAE Epoch {epoch}/{total}")
                status_text.text(f"VAE — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            vae_model, vae_total, vae_recon, vae_kl, vae_val = train_vae(
                train_loader, val_loader, latent_dim=latent_dim, epochs=20, progress_callback=vae_callback
            )
            save_vae_checkpoint(vae_model, vae_ckpt)

            st.session_state.ae_model = ae_model
            st.session_state.vae_model = vae_model
            st.session_state.ae_train_losses = ae_train
            st.session_state.ae_val_losses = ae_val
            st.session_state.vae_total_losses = vae_total
            st.session_state.vae_recon_losses = vae_recon
            st.session_state.vae_kl_losses = vae_kl
            st.session_state.vae_val_total_losses = vae_val
            st.session_state.ae_vae_trained = True

            progress_bar.empty()
            status.update(label="训练完成!", state="complete")

    # ── 尝试加载已有权重 ──
    elif not st.session_state.ae_vae_trained:
        if os.path.exists(ae_ckpt) and os.path.exists(vae_ckpt):
            st.info("发现已有模型权重，正在加载...")
            st.session_state.ae_model = load_ae_checkpoint(ae_ckpt)
            st.session_state.vae_model = load_vae_checkpoint(vae_ckpt)
            st.session_state.ae_vae_trained = True
            st.success("模型权重已加载！")

    # ── 展示结果 ──
    if st.session_state.ae_vae_trained:
        ae_model = st.session_state.ae_model
        vae_model = st.session_state.vae_model
        device = get_device()

        # 加载测试图像（仅一次）
        if st.session_state.test_images is None:
            imgs, lbls = get_test_images(dataset_name, n=25)
            st.session_state.test_images = imgs
            st.session_state.test_labels = lbls

        test_imgs = st.session_state.test_images.to(device)
        test_lbls = st.session_state.test_labels

        # ── 重构对比 ──
        st.subheader("重构对比")
        ae_model.eval()
        vae_model.eval()
        with torch.no_grad():
            ae_recon, _ = ae_model(test_imgs.to(device))
            vae_recon, _, _ = vae_model(test_imgs.to(device))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**原始输入**")
            st.image(make_image_grid(test_imgs.cpu(), n_cols=5), use_container_width=True)
        with col2:
            st.write("**AE 重构**")
            st.image(make_image_grid(ae_recon.cpu(), n_cols=5), use_container_width=True)
        with col3:
            st.write("**VAE 重构**")
            st.image(make_image_grid(vae_recon.cpu(), n_cols=5), use_container_width=True)

        # ── 误差热力图 ──
        st.subheader("重构误差热力图")
        col_heat1, col_heat2 = st.columns(2)
        with col_heat1:
            st.write("**AE 重构误差**")
            st.image(make_error_heatmaps(test_imgs.cpu(), ae_recon.cpu(), n_cols=5),
                     use_container_width=True)
        with col_heat2:
            st.write("**VAE 重构误差**")
            st.image(make_error_heatmaps(test_imgs.cpu(), vae_recon.cpu(), n_cols=5),
                     use_container_width=True)

        # ── Loss 曲线 ──
        st.subheader("训练 Loss 曲线")
        loss_fig = plot_loss_curves(
            st.session_state.ae_train_losses,
            st.session_state.ae_val_losses,
            st.session_state.vae_total_losses,
            st.session_state.vae_recon_losses,
            st.session_state.vae_kl_losses,
        )
        st.plotly_chart(loss_fig, use_container_width=True)

        # ── 潜空间可视化（仅 latent_dim=2 时可用）──
        if latent_dim == 2:
            st.subheader("潜空间可视化")

            # 生成全量测试集的潜空间编码
            if st.session_state.latent_scatter_data is None:
                full_imgs, full_lbls = get_full_test_dataset(dataset_name)
                vae_model.eval()
                with torch.no_grad():
                    mu, _ = vae_model.encode(full_imgs.to(device))
                st.session_state.latent_scatter_data = {
                    'z': mu.cpu().numpy(),
                    'labels': full_lbls.cpu().numpy(),
                    'images': full_imgs,
                }

            scatter_data = st.session_state.latent_scatter_data

            col_scatter, col_interp = st.columns([3, 2])

            with col_scatter:
                scatter_fig = plot_latent_scatter(scatter_data['z'], scatter_data['labels'], class_names)
                clicked = st.plotly_chart(scatter_fig, use_container_width=True,
                                          key="latent_scatter", on_select="rerun")

                # 点击解码
                if clicked and clicked.selection and clicked.selection.points:
                    idx = clicked.selection.points[0]['point_index']
                    z_point = torch.tensor(scatter_data['z'][idx], dtype=torch.float32).unsqueeze(0).to(device)
                    vae_model.eval()
                    with torch.no_grad():
                        decoded = vae_model.decode_from_latent(z_point)
                    st.image(make_image_grid(decoded.cpu(), n_cols=1), width=150,
                             caption=f"索引 {idx}, 类别 {class_names[scatter_data['labels'][idx]]}")

            with col_interp:
                st.write("**潜空间插值**")
                idx_a = st.number_input("样本 A 索引", 0, len(scatter_data['labels']) - 1, 0,
                                        key="interp_a")
                idx_b = st.number_input("样本 B 索引", 0, len(scatter_data['labels']) - 1, 100,
                                        key="interp_b")
                steps = st.slider("插值步数", 2, 16, 8, key="interp_steps")

                if st.button("生成插值序列", key="interp_btn"):
                    z_a = torch.tensor(scatter_data['z'][idx_a], dtype=torch.float32)
                    z_b = torch.tensor(scatter_data['z'][idx_b], dtype=torch.float32)
                    alphas = torch.linspace(0, 1, steps)
                    z_interp = torch.stack([(1 - a) * z_a + a * z_b for a in alphas]).to(device)
                    vae_model.eval()
                    with torch.no_grad():
                        decoded_seq = vae_model.decode_from_latent(z_interp)
                    st.image(make_image_grid(decoded_seq.cpu(), n_cols=steps), use_container_width=True,
                             caption=f"{class_names[scatter_data['labels'][idx_a]]} → {class_names[scatter_data['labels'][idx_b]]}")
        else:
            st.info("潜空间散点图仅在 latent_dim=2 时可用。请将潜空间维度设为 2 后重新训练。")
    else:
        if not train_btn:
            st.info("点击「训练模型」按钮开始训练，或将权重文件放入 checkpoints/ 目录。")
```

- [ ] **Step 2: 验证模块导入**

```powershell
cd d:\project\AIvibe\image7
python -c "from ui.tab_ae_vae import render_ae_vae_tab; print('Tab AE VAE import OK')"
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/ui/tab_ae_vae.py
git commit -m "feat: 添加 Tab 1 — AE vs VAE 重构对比 UI"
```

---

### Task 8: DCGAN 模型与训练

**文件：**
- 创建: `image7/models/dcgan.py`

- [ ] **Step 1: 编写 dcgan.py**

```python
"""DCGAN 模型定义与训练。"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.session_manager import get_device


class Generator(nn.Module):
    """DCGAN 生成器：噪声 → 28×28×1 图像。"""

    def __init__(self, noise_dim: int = 100):
        super().__init__()
        self.noise_dim = noise_dim
        self.model = nn.Sequential(
            # 1×1 → 4×4
            nn.ConvTranspose2d(noise_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 4×4 → 7×7
            nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 7×7 → 14×14
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 14×14 → 28×28
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z.view(z.size(0), self.noise_dim, 1, 1))


class Discriminator(nn.Module):
    """DCGAN 判别器：28×28×1 → 真/假概率。"""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 28×28 → 14×14
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 14×14 → 7×7
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 7×7 → 4×4
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → 1
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1, 1)


def train_dcgan(
    train_loader: DataLoader,
    noise_dim: int = 100,
    epochs: int = 50,
    lr: float = 2e-4,
    betas: tuple = (0.5, 0.999),
    progress_callback=None,
    checkpoint_callback=None,
) -> tuple[Generator, Discriminator]:
    """训练 DCGAN。

    Returns:
        (generator, discriminator)
    """
    device = get_device()
    netG = Generator(noise_dim=noise_dim).to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCELoss()
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=betas)
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=betas)

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # 训练判别器
            netD.zero_grad()
            label = torch.full((batch_size, 1), real_label, device=device)
            output_real = netD(real_imgs)
            lossD_real = criterion(output_real, label)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = netG(noise)
            label.fill_(fake_label)
            output_fake = netD(fake_imgs.detach())
            lossD_fake = criterion(output_fake, label)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # 训练生成器
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_imgs)
            lossG = criterion(output, label)
            lossG.backward()
            optG.step()

        if progress_callback:
            progress_callback(epoch + 1, epochs, lossG.item(), lossD.item())

        if checkpoint_callback:
            checkpoint_callback(epoch + 1, netG, netD)

    return netG, netD


def save_dcgan_checkpoint(netG: Generator, netD: Discriminator,
                          optimizer_state: dict | None, path_g: str, path_d: str,
                          epoch: int | None = None) -> None:
    """保存 DCGAN 权重。"""
    g_data = {'model_state_dict': netG.state_dict(), 'noise_dim': netG.noise_dim}
    d_data = {'model_state_dict': netD.state_dict()}
    if epoch is not None:
        g_data['epoch'] = d_data['epoch'] = epoch
    torch.save(g_data, path_g)
    torch.save(d_data, path_d)


def load_dcgan_checkpoint(path_g: str, path_d: str) -> tuple[Generator, Discriminator]:
    """加载 DCGAN 权重。"""
    device = get_device()
    g_data = torch.load(path_g, map_location=device, weights_only=True)
    d_data = torch.load(path_d, map_location=device, weights_only=True)
    netG = Generator(noise_dim=g_data['noise_dim']).to(device)
    netG.load_state_dict(g_data['model_state_dict'])
    netG.eval()
    netD = Discriminator().to(device)
    netD.load_state_dict(d_data['model_state_dict'])
    netD.eval()
    return netG, netD
```

- [ ] **Step 2: 快速训练测试（2 epochs）**

```powershell
cd d:\project\AIvibe\image7
python -c "
from utils.data_loader import get_dataloader
from models.dcgan import train_dcgan
loader = get_dataloader('mnist', batch_size=128, train=True)
g, d = train_dcgan(loader, epochs=2, progress_callback=lambda e, t, lg, ld: print(f'Epoch {e}: G loss={lg:.4f}, D loss={ld:.4f}'))
print('DCGAN training OK')
"
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/models/dcgan.py
git commit -m "feat: 添加 DCGAN 生成器/判别器与训练函数"
```

---

### Task 9: Tab 2 — DCGAN 生成 UI

**文件：**
- 创建: `image7/ui/tab_dcgan.py`

- [ ] **Step 1: 编写 tab_dcgan.py**

```python
"""Tab 2: DCGAN 生成。"""
import os
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

from utils.data_loader import get_dataloader
from utils.visualization import make_image_grid
from utils.session_manager import get_device
from models.dcgan import (Generator, Discriminator, train_dcgan,
                          save_dcgan_checkpoint, load_dcgan_checkpoint)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')


def render_dcgan_tab() -> None:
    """渲染 'DCGAN 生成' Tab 的全部 UI。"""
    st.header("DCGAN 图像生成")

    # ── 控制栏 ──
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_name = st.selectbox("数据集", ["mnist", "fashion_mnist"], key="dcgan_dataset")
    with col2:
        noise_dim = st.slider("噪声维度", 20, 200, 100, 10, key="dcgan_noise_dim")
    with col3:
        grid_size = st.slider("每行样本数", 4, 12, 8, key="dcgan_grid_size")

    # ── 初始化 session state ──
    for key, default in [
        ('dcgan_g', None), ('dcgan_d', None), ('dcgan_trained', False),
        ('dcgan_noise', None), ('dcgan_fake_imgs', None), ('dcgan_scores', None),
        ('dcgan_ref_grid', None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    g_ckpt = os.path.join(CHECKPOINT_DIR, f'dcgan_g_{dataset_name}.pt')
    d_ckpt = os.path.join(CHECKPOINT_DIR, f'dcgan_d_{dataset_name}.pt')

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        train_btn = st.button("训练 DCGAN", key="dcgan_train_btn", type="primary")
    with col_btn2:
        gen_btn = st.button("生成新样本", key="dcgan_gen_btn")

    # ── 训练 ──
    if train_btn:
        with st.status("训练 DCGAN 中...", expanded=True) as status:
            loader = get_dataloader(dataset_name, batch_size=128, train=True)
            progress_bar = st.progress(0)

            def callback(epoch, total, g_loss, d_loss):
                progress_bar.progress(epoch / total, text=f"Epoch {epoch}/{total} — G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")

            g, d = train_dcgan(loader, noise_dim=noise_dim, epochs=50, progress_callback=callback)
            save_dcgan_checkpoint(g, d, None, g_ckpt, d_ckpt)
            st.session_state.dcgan_g = g
            st.session_state.dcgan_d = d
            st.session_state.dcgan_trained = True
            progress_bar.empty()
            status.update(label="训练完成!", state="complete")

    # ── 加载已有权重 ──
    elif not st.session_state.dcgan_trained:
        if os.path.exists(g_ckpt) and os.path.exists(d_ckpt):
            st.info("发现已有模型权重，正在加载...")
            g, d = load_dcgan_checkpoint(g_ckpt, d_ckpt)
            st.session_state.dcgan_g = g
            st.session_state.dcgan_d = d
            st.session_state.dcgan_trained = True
            st.success("模型权重已加载！")

    # ── 生成新样本 ──
    if gen_btn and st.session_state.dcgan_trained:
        device = get_device()
        g = st.session_state.dcgan_g
        d = st.session_state.dcgan_d
        g.eval()
        d.eval()
        n = grid_size * grid_size
        noise = torch.randn(n, noise_dim, device=device)
        with torch.no_grad():
            fake_imgs = g(noise)
            scores = d(fake_imgs)
        st.session_state.dcgan_noise = noise.cpu()
        st.session_state.dcgan_fake_imgs = fake_imgs.cpu()
        st.session_state.dcgan_scores = scores.cpu().numpy().flatten()

    # ── 展示生成结果 ──
    if st.session_state.dcgan_fake_imgs is not None:
        st.subheader("生成样本网格")

        col_grid, col_score = st.columns([3, 1])
        with col_grid:
            st.image(make_image_grid(st.session_state.dcgan_fake_imgs, n_cols=grid_size),
                     use_container_width=True)
        with col_score:
            st.write("**判别器分数 D(G(z))**")
            scores = st.session_state.dcgan_scores
            sorted_idx = np.argsort(scores)[::-1]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=np.arange(len(scores)),
                y=scores[sorted_idx],
                marker_color=np.where(scores[sorted_idx] > 0.5, '#2ca02c', '#d62728'),
            ))
            fig.update_layout(
                title='降序排列',
                xaxis_title='样本序号',
                yaxis_title='D(G(z))',
                height=400,
                yaxis_range=[0, 1],
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 噪声热力图 ──
        st.subheader("噪声向量可视化")
        selected_idx = st.slider("选择样本查看噪声", 0, len(scores) - 1, 0, key="noise_inspect")
        noise_vec = st.session_state.dcgan_noise[selected_idx].numpy()
        noise_2d = noise_vec.reshape(10, -1)
        fig2, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(noise_2d, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'噪声向量 (样本 {selected_idx})')
        buf = io.BytesIO()
        fig2.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig2)
        buf.seek(0)
        st.image(buf, use_container_width=True)

        # ── 保存参考 ──
        if st.button("保存当前网格作为参考", key="dcgan_ref_btn"):
            st.session_state.dcgan_ref_grid = st.session_state.dcgan_fake_imgs.clone()
            st.success("参考网格已保存！")

        # ── 对比展示 ──
        if st.session_state.dcgan_ref_grid is not None:
            st.subheader("当前 vs 参考对比")
            col_cur, col_ref = st.columns(2)
            with col_cur:
                st.write("**当前生成**")
                st.image(make_image_grid(st.session_state.dcgan_fake_imgs, n_cols=grid_size),
                         use_container_width=True)
            with col_ref:
                st.write("**参考网格**")
                st.image(make_image_grid(st.session_state.dcgan_ref_grid, n_cols=grid_size),
                         use_container_width=True)
    else:
        if st.session_state.dcgan_trained:
            pass  # 已训练但未生成
        else:
            st.info("点击「训练 DCGAN」开始训练，或将权重文件放入 checkpoints/ 目录。")
```

- [ ] **Step 2: 验证导入**

```powershell
cd d:\project\AIvibe\image7
python -c "from ui.tab_dcgan import render_dcgan_tab; print('Tab DCGAN import OK')"
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/ui/tab_dcgan.py
git commit -m "feat: 添加 Tab 2 — DCGAN 生成 UI"
```

---

### Task 10: Diffusion 加载器

**文件：**
- 创建: `image7/models/diffusion_loader.py`

- [ ] **Step 1: 编写 diffusion_loader.py**

```python
"""Stable Diffusion 1.5 加载与推理。"""
import torch
from utils.session_manager import get_device, clear_cuda_memory


def load_sd_pipeline():
    """加载 SD 1.5 pipeline（FP16, attention slicing）。"""
    from diffusers import StableDiffusionPipeline

    device = get_device()
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None,
    )
    if device.type == 'cuda':
        pipe.enable_attention_slicing()
    pipe = pipe.to(device)

    return pipe


def generate_image(pipe, prompt: str, negative_prompt: str = "",
                   num_inference_steps: int = 20, guidance_scale: float = 7.5,
                   seed: int | None = None, width: int = 512, height: int = 512):
    """使用 SD pipeline 生成图像。"""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=width,
        height=height,
    )
    return result.images[0]
```

- [ ] **Step 2: 提交**

```bash
cd d:/project/AIvibe
git add image7/models/diffusion_loader.py
git commit -m "feat: 添加 SD 1.5 加载与推理工具"
```

---

### Task 11: Tab 3 — 文本到图像 UI

**文件：**
- 创建: `image7/ui/tab_diffusion.py`

- [ ] **Step 1: 编写 tab_diffusion.py**

```python
"""Tab 3: 文本到图像（Diffusers）。"""
import time
import streamlit as st
from PIL import Image

from utils.session_manager import get_device, clear_cuda_memory
from models.diffusion_loader import load_sd_pipeline, generate_image


def render_diffusion_tab() -> None:
    """渲染 '文本到图像' Tab 的全部 UI。"""
    st.header("文本到图像生成 (Stable Diffusion 1.5)")

    # ── 初始化 session state ──
    for key, default in [
        ('diffusion_pipe', None), ('diffusion_loaded', False),
        ('diffusion_runs', []),  # list of dict: {params, image}
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── 侧边栏参数 ──
    with st.sidebar:
        st.subheader("生成参数")

        prompt = st.text_area("正向提示词", "A cute cat sitting on a cloud, digital art, high quality",
                              key="diff_prompt")
        neg_prompt = st.text_area("负向提示词", "blurry, low quality, distorted, ugly",
                                  key="diff_neg_prompt")

        col1, col2 = st.columns(2)
        with col1:
            steps = st.slider("采样步数", 5, 50, 20, 5, key="diff_steps")
        with col2:
            cfg = st.slider("CFG 引导强度", 1.0, 20.0, 7.5, 0.5, key="diff_cfg")

        col3, col4 = st.columns(2)
        with col3:
            use_seed = st.checkbox("固定种子", True, key="diff_use_seed")
            seed = st.number_input("随机种子", 0, 2**31 - 1, 42, key="diff_seed")
        with col4:
            resolution = st.selectbox("分辨率", [256, 384, 512], index=2, key="diff_resolution")

        st.divider()

        load_btn = st.button("加载 SD 模型", key="diff_load_btn")
        gen_btn = st.button("生成图像", key="diff_gen_btn", type="primary", use_container_width=True)

        st.divider()
        st.write(f"已保存结果: {len(st.session_state.diffusion_runs)}/4")
        if st.button("清空全部结果", key="diff_clear_btn"):
            st.session_state.diffusion_runs = []
            st.rerun()

    # ── 加载模型 ──
    if load_btn:
        with st.spinner("正在加载 Stable Diffusion 1.5... (首次需下载 ~5GB)"):
            st.session_state.diffusion_pipe = load_sd_pipeline()
            st.session_state.diffusion_loaded = True
        st.success("SD 1.5 模型已加载！")

    if not st.session_state.diffusion_loaded:
        # 尝试自动加载
        auto_load = st.checkbox("自动加载模型（需 ~4GB 显存）", key="diff_auto_load")
        if auto_load:
            with st.spinner("正在加载 Stable Diffusion 1.5..."):
                st.session_state.diffusion_pipe = load_sd_pipeline()
                st.session_state.diffusion_loaded = True
            st.success("模型已加载！")
        else:
            st.info("请先点击侧边栏「加载 SD 模型」，首次加载需下载约 5GB 模型文件。")
            return

    # ── 生成图像 ──
    if gen_btn:
        if len(st.session_state.diffusion_runs) >= 4:
            st.warning("最多保存 4 组结果。请先清空后再生成。")
        else:
            with st.spinner(f"生成中... (分辨率 {resolution}, 步数 {steps})"):
                t0 = time.time()
                img = generate_image(
                    st.session_state.diffusion_pipe,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    seed=seed if use_seed else None,
                    width=resolution,
                    height=resolution,
                )
                elapsed = time.time() - t0

            run = {
                'params': {
                    'prompt': prompt,
                    'negative_prompt': neg_prompt,
                    'steps': steps,
                    'cfg': cfg,
                    'seed': seed if use_seed else 'random',
                    'resolution': resolution,
                },
                'image': img,
                'time': elapsed,
            }
            st.session_state.diffusion_runs.append(run)
            st.success(f"生成完成！耗时 {elapsed:.1f}s")

    # ── 展示结果网格 ──
    runs = st.session_state.diffusion_runs
    if runs:
        st.subheader("生成结果对比")

        # 对比模式
        view_mode = st.radio("对比模式", ["2×2 网格", "并排对比 (A/B)", "滑块分割"],
                             horizontal=True, key="diff_view_mode")

        if view_mode == "2×2 网格":
            cols = st.columns(2)
            for i, run in enumerate(runs):
                p = run['params']
                with cols[i % 2]:
                    st.image(run['image'], use_container_width=True)
                    st.caption(
                        f"**#{i+1}** | Steps: {p['steps']} | CFG: {p['cfg']:.1f} | "
                        f"Seed: {p['seed']} | Res: {p['resolution']} | {run['time']:.1f}s\n"
                        f"Prompt: {p['prompt'][:80]}..."
                    )

        elif view_mode == "并排对比 (A/B)":
            if len(runs) >= 2:
                col_a, col_b = st.columns(2)
                a_idx = st.selectbox("选择 A", range(len(runs)), key="diff_ab_a",
                                     format_func=lambda i: f"结果 {i+1}")
                b_idx = st.selectbox("选择 B", range(len(runs)), key="diff_ab_b",
                                     format_func=lambda i: f"结果 {i+1}")
                with col_a:
                    st.image(runs[a_idx]['image'], use_container_width=True)
                    st.caption(f"**A** — Steps: {runs[a_idx]['params']['steps']}, "
                               f"CFG: {runs[a_idx]['params']['cfg']:.1f}")
                with col_b:
                    st.image(runs[b_idx]['image'], use_container_width=True)
                    st.caption(f"**B** — Steps: {runs[b_idx]['params']['steps']}, "
                               f"CFG: {runs[b_idx]['params']['cfg']:.1f}")
            else:
                st.info("需要至少 2 个结果才能进行 A/B 对比。")

        elif view_mode == "滑块分割":
            if len(runs) >= 2:
                a_idx = st.selectbox("选择左图", range(len(runs)), key="diff_split_a",
                                     format_func=lambda i: f"结果 {i+1}")
                b_idx = st.selectbox("选择右图", range(len(runs)), key="diff_split_b",
                                     format_func=lambda i: f"结果 {i+1}")

                # 使用 st.columns + 滑块模拟分割效果
                split_pos = st.slider("分割位置", 0, 100, 50, key="diff_split_pos")
                img_a = runs[a_idx]['image'].resize((512, 512))
                img_b = runs[b_idx]['image'].resize((512, 512))

                # 裁剪拼接
                left_part = img_a.crop((0, 0, int(512 * split_pos / 100), 512))
                right_part = img_b.crop((int(512 * split_pos / 100), 0, 512, 512))
                combined = Image.new('RGB', (512, 512))
                combined.paste(left_part, (0, 0))
                combined.paste(right_part, (int(512 * split_pos / 100), 0))

                st.image(combined, use_container_width=True,
                         caption=f"A (左): Steps={runs[a_idx]['params']['steps']} | "
                                 f"B (右): Steps={runs[b_idx]['params']['steps']}")
            else:
                st.info("需要至少 2 个结果才能进行滑块分割对比。")
```

- [ ] **Step 2: 验证导入**

```powershell
cd d:\project\AIvibe\image7
python -c "from ui.tab_diffusion import render_diffusion_tab; print('Tab Diffusion import OK')"
```

- [ ] **Step 3: 提交**

```bash
cd d:/project/AIvibe
git add image7/ui/tab_diffusion.py
git commit -m "feat: 添加 Tab 3 — 文本到图像 UI"
```

---

### Task 12: 主入口 app.py

**文件：**
- 创建: `image7/app.py`

- [ ] **Step 1: 编写 app.py**

```python
"""图像处理 Web 应用 — 主入口。

运行: streamlit run image7/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="图像处理实验室",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="auto",
)

from ui.tab_ae_vae import render_ae_vae_tab
from ui.tab_dcgan import render_dcgan_tab
from ui.tab_diffusion import render_diffusion_tab
from utils.session_manager import clear_cuda_memory


def main():
    st.title("🎨 图像处理实验室")

    st.markdown(
        "自编码器 vs VAE 对比 · DCGAN 生成 · 文本到图像 — 实时调参交互 · 前后效果对比"
    )

    # Tab 路由
    tab_labels = ["AE vs VAE 重构对比", "DCGAN 生成", "文本到图像 (SD 1.5)"]
    tab_funcs = [render_ae_vae_tab, render_dcgan_tab, render_diffusion_tab]

    tabs = st.tabs(tab_labels)

    # 追踪当前 Tab，切换时清理显存
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0

    for i, (tab, func) in enumerate(zip(tabs, tab_funcs)):
        with tab:
            if st.session_state.current_tab != i:
                clear_cuda_memory()
                st.session_state.current_tab = i
            func()

    # 侧边栏：显存监控
    with st.sidebar:
        st.divider()
        st.subheader("💾 显存状态")
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                st.metric("已分配", f"{alloc:.0f} MB")
                st.metric("已保留", f"{reserved:.0f} MB")
                st.progress(alloc / total, text=f"占用率 {alloc/total*100:.1f}%")
        except Exception:
            st.caption("显存信息不可用")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 启动应用进行冒烟测试**

```powershell
cd d:\project\AIvibe\image7
Start-Process -NoNewWindow streamlit -ArgumentList "run", "app.py", "--server.port", "8501"
```

等待加载后检查：
- 页面标题 "图像处理实验室" 可见
- 三个 Tab 正常渲染
- 无导入错误

- [ ] **Step 3: 关闭应用后提交**

```bash
cd d:/project/AIvibe
git add image7/app.py
git commit -m "feat: 添加 Streamlit 主入口 app.py"
```

---

## 实现顺序

按 Task 1 → 12 顺序执行：先基础设施（依赖、工具），再模型定义，再 UI，最后集成入口。

## 验证清单

- [ ] `streamlit run image7/app.py` 成功启动
- [ ] Tab 1: AE/VAE 训练完成后展示重构对比、热力图、loss 曲线、潜空间散点
- [ ] Tab 1: 潜空间点击解码正常，插值序列正常
- [ ] Tab 2: DCGAN 训练后生成样本网格、判别器分数柱状图
- [ ] Tab 2: 噪声热力图、参考保存与对比
- [ ] Tab 3: SD 1.5 加载成功，参数调整后生成图像
- [ ] Tab 3: 多轮结果 2×2 网格对比、A/B 并排、滑块分割均正常
- [ ] Tab 切换时不发生 CUDA OOM
