"""MNIST / Fashion-MNIST 数据加载工具。"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(dataset_name: str, batch_size: int = 128, train: bool = True) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'mnist':
        ds_cls = datasets.MNIST
    elif dataset_name == 'fashion_mnist':
        ds_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    dataset = ds_cls(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


def get_test_images(dataset_name: str, n: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    loader = get_dataloader(dataset_name, batch_size=n, train=False)
    images, labels = next(iter(loader))
    return images[:n], labels[:n]


def get_full_test_dataset(dataset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    loader = get_dataloader(dataset_name, batch_size=10000, train=False)
    images, labels = next(iter(loader))
    return images, labels
