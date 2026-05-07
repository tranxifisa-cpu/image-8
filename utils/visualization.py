"""可视化工具：loss 曲线、潜空间散点图、误差热力图、图像网格。"""
import io
import numpy as np
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid as tv_make_grid


def plot_loss_curves(ae_train_losses: list[float], ae_val_losses: list[float],
                     vae_total_losses: list[float], vae_recon_losses: list[float],
                     vae_kl_losses: list[float]) -> go.Figure:
    epochs = list(range(1, len(ae_train_losses) + 1)) if ae_train_losses else list(range(1, len(vae_total_losses) + 1))
    fig = go.Figure()
    if ae_train_losses:
        fig.add_trace(go.Scatter(x=epochs, y=ae_train_losses, mode='lines', name='AE 训练 Loss', line=dict(color='#1f77b4')))
    if ae_val_losses:
        fig.add_trace(go.Scatter(x=epochs, y=ae_val_losses, mode='lines', name='AE 验证 Loss', line=dict(color='#1f77b4', dash='dash')))
    if vae_total_losses:
        fig.add_trace(go.Scatter(x=epochs, y=vae_total_losses, mode='lines', name='VAE 总 Loss', line=dict(color='#ff7f0e')))
    if vae_recon_losses:
        fig.add_trace(go.Scatter(x=epochs, y=vae_recon_losses, mode='lines', name='VAE 重构 Loss', line=dict(color='#2ca02c')))
    if vae_kl_losses:
        fig.add_trace(go.Scatter(x=epochs, y=vae_kl_losses, mode='lines', name='VAE KL Loss', line=dict(color='#d62728')))
    fig.update_layout(title='训练 Loss 曲线', xaxis_title='Epoch', yaxis_title='Loss', legend=dict(orientation='h', yanchor='bottom', y=1.02), height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_latent_scatter(latent_vecs: np.ndarray, labels: np.ndarray, class_names: list[str]) -> go.Figure:
    if latent_vecs.shape[1] != 2:
        raise ValueError(f"latent_vecs must be (N, 2), got shape {latent_vecs.shape}")
    fig = go.Figure()
    for cls_id in sorted(set(labels.tolist() if hasattr(labels, 'tolist') else list(labels))):
        mask = labels == cls_id
        fig.add_trace(go.Scatter(
            x=latent_vecs[mask, 0], y=latent_vecs[mask, 1],
            mode='markers', name=class_names[cls_id],
            marker=dict(size=3, opacity=0.6),
            customdata=np.column_stack([np.arange(len(labels))[mask], labels[mask]]),
            hovertemplate='Index: %{customdata[0]}<br>Class: %{customdata[1]}<extra></extra>',
        ))
    fig.update_layout(title='潜空间分布 (2D)', xaxis_title='z₁', yaxis_title='z₂', height=500, legend=dict(itemsizing='constant'), margin=dict(l=20, r=20, t=40, b=20))
    return fig


def make_error_heatmaps(originals: torch.Tensor, reconstructions: torch.Tensor, n_cols: int = 5) -> Image.Image:
    if originals.shape[0] == 0:
        return Image.new('RGB', (100, 100), color='white')
    n = originals.shape[0]
    n_rows = (n + n_cols - 1) // n_cols
    errors = torch.abs(originals - reconstructions).cpu().numpy().squeeze(1)
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


def make_image_grid(images: torch.Tensor, n_cols: int = 5, vmin: float = 0.0, vmax: float = 1.0) -> Image.Image:
    grid = tv_make_grid(images, nrow=n_cols, normalize=False, pad_value=1.0)
    grid_np = grid.cpu().detach()
    if grid_np.shape[0] == 1:
        grid_np = grid_np.repeat(3, 1, 1)
    grid_np = (grid_np.clamp(vmin, vmax) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(grid_np)
