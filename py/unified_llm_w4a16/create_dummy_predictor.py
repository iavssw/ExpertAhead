
import torch
import torch.nn as nn
import os

class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_experts):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        # x: [batch, 32]
        emb = self.embedding(x) # [batch, 32, embed_dim]
        emb_flat = emb.view(x.size(0), -1) # [batch, 32*embed_dim]
        logits = self.mlp(emb_flat)
        return logits

def create_model(output_path):
    vocab_size = 32000
    embed_dim = 16
    hidden_dim = 64
    num_experts = 8

    model = SimpleMLP(vocab_size, embed_dim, hidden_dim, num_experts)
    model.eval()

    # Trace the model
    example_input = torch.randint(0, vocab_size, (1, 32))
    traced_model = torch.jit.trace(model, example_input)

    traced_model.save(output_path)
    print(f"Saved dummy predictor to {output_path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    create_model("models/dummy_predictor.pt")
