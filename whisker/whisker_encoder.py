import torch
import torch.nn as nn
    
class WhiskerEncoder(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128):
        super(WhiskerEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        # x shape: [B, 60, 15, 4]
        B, N, T, D = x.shape  # batch, whiskers, time steps, features

        x = x.view(B * N * T, D)        # Flatten for MLP
        x = self.encoder(x)
        x = x.view(B, N, T, -1)         # Reshape back

        x = x.mean(dim=2)              # ⏱️ Average over time
        x = x.mean(dim=1)              # 🪶 Average over whiskers

        return x                       # [B, embed_dim]

class TemporalWhiskerEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, embed_dim=128, time_steps=15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps

        # GRU: processes one whisker’s time sequence
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Projection layer (e.g., to match contrastive dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: [B, W, T, 4]
        B, W, T, D = x.shape
        x = x.view(B * W, T, D)  # [B*W, T, 4]

        # GRU: returns [B*W, T, hidden] and (final_hidden_state)
        _, h_n = self.gru(x)     # h_n: [1, B*W, hidden]
        h_n = h_n.squeeze(0)     # [B*W, hidden]

        # Project and reshape
        out = self.output_layer(h_n)        # [B*W, embed]
        out = out.view(B, W, -1)            # [B, W, embed]

        # Aggregate over whiskers (mean or max)
        out = out.mean(dim=1)               # [B, embed]
        return out


