"""Transaction classification head, not a link-prediction decoder."""

import torch
from torch import Tensor, nn


class EdgeClassifier(nn.Module):
    """Map source/destination embeddings and current edge features to logits [B]."""

    def __init__(self, embedding_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src: Tensor, dst: Tensor, features: Tensor) -> Tensor:
        """Inputs [B,D], [B,D], [B,F]; output logits [B]."""
        return self.net(torch.cat([src, dst, features], dim=-1)).squeeze(-1)
