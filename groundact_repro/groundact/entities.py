"""Prefill-time grounding (Section 3.2): entity queries and the per-state prediction heads."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EntityBank:
    """The K entity states M = {M_i} of one scene, addressable by index i (scene-local)."""
    states: torch.Tensor  # (K, D) final hidden states read at the entity-query positions
    valid: torch.Tensor  # (K,) bool, False for detector-null slots
    boxes: torch.Tensor | None = None  # (K, 6) normalized (cx, cy, l, w, sin, cos) if decoded
    classes: torch.Tensor | None = None  # (K,) argmax class, C = null

    @property
    def k(self) -> int:
        return self.states.shape[0]


class EntityQueries(nn.Module):
    """K learnable query embeddings appended to the multimodal prefill (q_1..q_K, K = 64)."""

    def __init__(self, hidden_size: int, num_entities: int = 64):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_entities, hidden_size) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.query.unsqueeze(0).expand(batch_size, -1, -1)


class EntityHeads(nn.Module):
    """Class (incl. null) and oriented BEV box heads decoding each M_i (Eq. 3 supervision)."""

    def __init__(self, hidden_size: int, num_classes: int, box_dim: int = 6):
        super().__init__()
        self.cls = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                 nn.Linear(hidden_size, num_classes + 1))
        self.box = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                 nn.Linear(hidden_size, box_dim))
        self.num_classes = num_classes

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls(states), self.box(states)

    @torch.no_grad()
    def read(self, states: torch.Tensor) -> EntityBank:
        logits, boxes = self(states)
        classes = logits.argmax(-1)
        valid = classes != self.num_classes  # null class excluded
        return EntityBank(states=states, valid=valid, boxes=boxes, classes=classes)


def read_entity_states(hidden_states: torch.Tensor, entity_positions: torch.Tensor) -> torch.Tensor:
    """M_i = H_theta(O, q_1..q_K)[q_i]: gather the final hidden states at the K query positions.

    hidden_states (B, L, D); entity_positions (B, K) -> (B, K, D)
    """
    idx = entity_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    return hidden_states.gather(1, idx)
