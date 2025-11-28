# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.creator import generate_samples

# Prepare data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get model
model = get_model("diffusion")  # lowercase is fine

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train model
trained_model = train_diffusion(model, train_loader, criterion, optimizer, device, epochs=3)

# Generate samples
generate_samples(trained_model, device, num_samples=10, diffusion_steps=60)
