# helper_lib/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Diffusion Model Definition ---
class SimpleDiffusionUNet(nn.Module):
    def __init__(self, img_channels=1, hidden_dim=64):
        super(SimpleDiffusionUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, img_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, t=None):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# --- Model Selector ---
def get_model(model_name):
    """Return the appropriate model"""
    model_name = model_name.lower()  # normalize case
    if model_name == "diffusion":
        return SimpleDiffusionUNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")
