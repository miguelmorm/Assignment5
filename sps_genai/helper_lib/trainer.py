# helper_lib/trainer.py
import torch

def train_diffusion(model, data_loader, criterion, optimizer, device="cpu", epochs=5):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            noise = torch.randn_like(imgs)
            noisy_imgs = imgs + 0.1 * noise  # add light noise
            optimizer.zero_grad()
            output = model(noisy_imgs)
            loss = criterion(output, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(data_loader):.4f}")

    return model
