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

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    torch.save({'model_state_dict': model.state_dict(), 'latent_dim': model.latent_dim}, path)


def load_ae_checkpoint(path: str) -> Autoencoder:
    data = torch.load(path, map_location=get_device(), weights_only=True)
    model = Autoencoder(latent_dim=data['latent_dim'])
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model
