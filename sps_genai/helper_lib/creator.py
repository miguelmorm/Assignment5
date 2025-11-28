# helper_lib/creator.py
import torch
import matplotlib.pyplot as plt

def generate_samples(model, device, num_samples=8, diffusion_steps=50):
    model.eval()
    model.to(device)

    # start from pure noise
    x = torch.randn(num_samples, 1, 28, 28).to(device)

    with torch.no_grad():
        for t in range(diffusion_steps):
            x = model(x)  # progressively denoise

    # visualize
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 2))
    for i in range(num_samples):
        axes[i].imshow(x[i][0].cpu().numpy(), cmap="gray")
        axes[i].axis("off")
    plt.show()
