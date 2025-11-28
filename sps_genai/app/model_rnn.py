import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


def generate_text(model, start_word, word_to_idx, idx_to_word, length=20, device="cpu"):
    model.eval()
    input_word = torch.tensor([[word_to_idx[start_word]]], device=device)
    hidden = None
    generated = [start_word]

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_word, hidden)
            probs = F.softmax(output[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            next_word = idx_to_word[next_id]
            generated.append(next_word)
            input_word = torch.tensor([[next_id]], device=device)

    return " ".join(generated)
