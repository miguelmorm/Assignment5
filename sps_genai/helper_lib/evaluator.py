# helper_lib/evaluator.py
import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):
    """Evaluate model on test set and return average loss and accuracy."""
    model.to(device)
    model.eval()
    total, correct, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
