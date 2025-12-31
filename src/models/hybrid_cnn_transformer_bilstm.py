from __future__ import annotations

import math
import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding for short fixed-length sequences (T=10).
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        idx = torch.arange(T, device=x.device)
        pe = self.pos(idx)[None, :, :]  # (1, T, D)
        return x + pe


class SnapshotCNNEncoder(nn.Module):
    """
    Encodes each timestep feature vector (F,) into an embedding (d_model,)
    using 1D conv along the feature axis.

    Input:  x (B, T, F)
    Output: z (B, T, d_model)
    """
    def __init__(self, d_model: int = 128, dropout: float = 0.15, k: int = 7):
        super().__init__()

        # We treat each timestep independently:
        # reshape (B*T, 1, F) and convolve over F.
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=k, padding=k // 2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=k, padding=k // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # compress variable F to a fixed representation
            nn.AdaptiveAvgPool1d(1),  # (B*T, 64, 1)
        )

        self.proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        xt = x.reshape(B * T, 1, F)           # (B*T, 1, F)
        h = self.net(xt).squeeze(-1)          # (B*T, 64)
        z = self.proj(h).reshape(B, T, -1)    # (B, T, d_model)
        return z


class HybridCNNTransformerBiLSTM(nn.Module):
    """
    CNN -> Transformer Encoder -> BiLSTM -> Classifier

    Input:  x (B, T=10, F)
    Output: logits (B, 2)
    """
    def __init__(
        self,
        window_size: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        ffn_dim: int = 256,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.cnn = SnapshotCNNEncoder(d_model=d_model, dropout=dropout, k=7)
        self.pos = LearnedPositionalEncoding(max_len=window_size, d_model=d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, T, D)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=0.0 if lstm_layers == 1 else dropout,
            bidirectional=True,
            batch_first=True,
        )

        # BiLSTM output dim = 2*lstm_hidden
        head_in = 2 * lstm_hidden
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        z = self.cnn(x)           # (B, T, d_model)
        z = self.pos(z)           # positional info
        z = self.transformer(z)   # (B, T, d_model)

        out, _ = self.bilstm(z)   # (B, T, 2*lstm_hidden)

        # Label corresponds to last timestep -> use last output
        last = out[:, -1, :]      # (B, 2*lstm_hidden)
        logits = self.head(last)  # (B, 2)
        return logits
