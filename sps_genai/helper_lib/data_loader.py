# helper_lib/data_loader.py
import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    """Create and return a PyTorch DataLoader for MNIST or CIFAR datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )
    return loader
