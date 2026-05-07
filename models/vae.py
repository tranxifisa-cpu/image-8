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

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),
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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)


def vae_loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    torch.save({'model_state_dict': model.state_dict(), 'latent_dim': model.latent_dim}, path)


def load_vae_checkpoint(path: str) -> VAE:
    data = torch.load(path, map_location=get_device(), weights_only=True)
    model = VAE(latent_dim=data['latent_dim'])
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model
