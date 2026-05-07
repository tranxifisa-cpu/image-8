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
            nn.ConvTranspose2d(noise_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
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
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
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
            label_real = torch.full((batch_size, 1), real_label, device=device)
            output_real = netD(real_imgs)
            lossD_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = netG(noise)
            label_fake = torch.full((batch_size, 1), fake_label, device=device)
            output_fake = netD(fake_imgs.detach())
            lossD_fake = criterion(output_fake, label_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # 训练生成器
            netG.zero_grad()
            label_real2 = torch.full((batch_size, 1), real_label, device=device)
            output = netD(fake_imgs)
            lossG = criterion(output, label_real2)
            lossG.backward()
            optG.step()

        if progress_callback:
            progress_callback(epoch + 1, epochs, lossG.item(), lossD.item())

        if checkpoint_callback:
            checkpoint_callback(epoch + 1, netG, netD)

    return netG, netD


def save_dcgan_checkpoint(netG: Generator, netD: Discriminator,
                          path_g: str, path_d: str, epoch: int | None = None) -> None:
    g_data = {'model_state_dict': netG.state_dict(), 'noise_dim': netG.noise_dim}
    d_data = {'model_state_dict': netD.state_dict()}
    if epoch is not None:
        g_data['epoch'] = epoch
        d_data['epoch'] = epoch
    torch.save(g_data, path_g)
    torch.save(d_data, path_d)


def load_dcgan_checkpoint(path_g: str, path_d: str) -> tuple[Generator, Discriminator]:
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
