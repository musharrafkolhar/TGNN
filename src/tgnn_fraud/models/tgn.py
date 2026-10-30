"""PyG TGN with causal timestamp-group scoring and transaction classification."""

import torch
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    TGNMemory,
)

from tgnn_fraud.models.edge_classifier import EdgeClassifier


class TemporalFraudModel(nn.Module):
    """Memory-based temporal GNN. State addresses all known IDs, never future attributes."""

    def __init__(self, num_nodes: int, feature_dim: int, config: dict):
        super().__init__()
        memory_dim, time_dim = config["memory_dim"], config["time_dim"]
        self.memory = TGNMemory(
            num_nodes,
            feature_dim,
            memory_dim,
            time_dim,
            IdentityMessage(feature_dim, memory_dim, time_dim),
            LastAggregator(),
        )
        self.conv = TransformerConv(
            memory_dim,
            config["embedding_dim"],
            heads=config["heads"],
            concat=False,
            edge_dim=feature_dim + time_dim,
            dropout=0.0,
        )
        self.head = EdgeClassifier(config["embedding_dim"], feature_dim, config["hidden_dim"])
        self.num_nodes = num_nodes
        self.neighbor_size = config["neighbor_size"]
        self.loader = LastNeighborLoader(num_nodes, self.neighbor_size, device="cpu")
        self.register_buffer("assoc", torch.empty(num_nodes, dtype=torch.long), persistent=False)

    def reset_history(self) -> None:
        """Zero node memory and cached neighbours before an epoch/replay."""
        self.memory.reset_state()
        self.loader = LastNeighborLoader(
            self.num_nodes, self.neighbor_size, device=self.memory.device
        )

    def score(
        self, src: Tensor, dst: Tensor, current: Tensor, all_features: Tensor, all_times: Tensor
    ) -> tuple[Tensor, dict]:
        """Score [B] events before insertion. Attention describes graph embedding edges."""
        n_id, edge_index, historical_ids = self.loader(torch.cat([src, dst]).unique())
        self.assoc[n_id] = torch.arange(len(n_id), device=n_id.device)
        memory, last_update = self.memory(n_id)
        # PyG loader IDs follow insertion order, identical to canonical chronological edge IDs.
        delta = last_update[edge_index[0]] - all_times[historical_ids]
        encoded_time = self.memory.time_enc(delta.to(current.dtype))
        edge_attr = torch.cat([all_features[historical_ids], encoded_time], dim=-1)
        embeddings, (_, weights) = self.conv(
            memory, edge_index, edge_attr, return_attention_weights=True
        )
        logits = self.head(embeddings[self.assoc[src]], embeddings[self.assoc[dst]], current)
        attention = {
            "edge_ids": historical_ids.detach().cpu().tolist(),
            "targets": n_id[edge_index[1]].detach().cpu().tolist(),
            "weights": weights.mean(dim=-1).detach().cpu().tolist(),
        }
        return logits, attention

    def observe(self, src: Tensor, dst: Tensor, times: Tensor, features: Tensor) -> None:
        """Insert [B] events only after every event at that timestamp has been scored."""
        self.memory.update_state(src, dst, times, features)
        self.loader.insert(src, dst)
        self.memory.detach()
