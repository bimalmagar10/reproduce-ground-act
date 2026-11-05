"""Interact(P^(k), S_i^(k)): proposal-relative interaction features (Section 3.4).

The differentiable clearance uses a 3-circle decomposition of each oriented box (a standard
planner approximation); the predicted-overlap signal that drives the risk latch uses the exact
separating-axis test of `groundact_repro.geometry`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from groundact_repro.geometry import obb_intersects
from groundact_repro.metrics import EgoDims

N_CIRCLES = 3


def box_circles(center: torch.Tensor, yaw: torch.Tensor, length: torch.Tensor, width: torch.Tensor):
    """Cover an oriented box with N_CIRCLES equal circles along its heading.

    center (..., 2), yaw (...), length/width broadcastable to (...). Returns (..., C, 2), (...)."""
    length = torch.as_tensor(length, dtype=center.dtype, device=center.device).expand(yaw.shape)
    width = torch.as_tensor(width, dtype=center.dtype, device=center.device).expand(yaw.shape)
    seg = length / N_CIRCLES
    offs = (torch.arange(N_CIRCLES, device=center.device, dtype=center.dtype) + 0.5) / N_CIRCLES - 0.5
    offs = offs * length[..., None]  # (..., C)
    d = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)  # (..., 2)
    centers = center[..., None, :] + offs[..., :, None] * d[..., None, :]
    radius = 0.5 * torch.sqrt(seg ** 2 + width ** 2)
    return centers, radius


def ego_box_center(xy: torch.Tensor, yaw: torch.Tensor, ego: EgoDims) -> torch.Tensor:
    off = ego.rear_axle_to_bbox_center
    return xy + off * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)


def circle_clearance(P_xy: torch.Tensor, P_yaw: torch.Tensor, ego: EgoDims, S_xy: torch.Tensor,
                     S_yaw: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """Clearance (m, negative when circles overlap) between ego proposals and entity forecasts.

    P_xy (B, T, 2), P_yaw (B, T); S_xy (N, T, 2), S_yaw (N, T), dims (N, 2) -> (B, N, T)."""
    ec, er = box_circles(ego_box_center(P_xy, P_yaw, ego), P_yaw, ego.length, ego.width)  # (B,T,C,2), (B,T)
    ac, ar = box_circles(S_xy, S_yaw, dims[:, None, 0], dims[:, None, 1])  # (N,T,C,2), (N,T)
    d = torch.linalg.norm(ec[:, None, :, :, None, :] - ac[None, :, :, None, :, :], dim=-1)  # (B,N,T,C,C)
    return d.amin(dim=(-1, -2)) - er[:, None] - ar[None]


def predicted_overlap(P_xy, P_yaw, ego: EgoDims, S_xy, S_yaw, dims, valid) -> torch.Tensor:
    """Exact footprint overlap over the horizon, (B, N) bool (numpy SAT under the hood)."""
    B, T = P_xy.shape[:2]
    N = S_xy.shape[0]
    if N == 0:
        return torch.zeros(B, 0, dtype=torch.bool, device=P_xy.device)
    p = P_xy.detach().float().cpu().numpy()
    py = P_yaw.detach().float().cpu().numpy()
    ec = p + ego.rear_axle_to_bbox_center * np.stack([np.cos(py), np.sin(py)], -1)
    hit = obb_intersects(ec[:, None], py[:, None], ego.length, ego.width,
                         S_xy.detach().float().cpu().numpy()[None], S_yaw.detach().float().cpu().numpy()[None],
                         dims[:, 0].cpu().numpy()[None, :, None], dims[:, 1].cpu().numpy()[None, :, None])
    hit &= valid.cpu().numpy()[None]
    return torch.as_tensor(hit.any(-1), device=P_xy.device)


@dataclass
class Interaction:
    clearance: torch.Tensor  # (B, N, T)
    rel_pos: torch.Tensor  # (B, N, T, 2) entity position in the ego-proposal frame at t
    rel_vel: torch.Tensor  # (B, N, T, 2)
    valid: torch.Tensor  # (N, T)
    overlap: torch.Tensor  # (B, N) exact predicted overlap anywhere on the horizon

    def min_clearance(self) -> torch.Tensor:  # (B, N)
        return self.clearance.masked_fill(~self.valid[None], float("inf")).amin(-1)


def interact(P_xy, P_yaw, ego: EgoDims, S_xy, S_yaw, dims, valid, dt: float = 0.1) -> Interaction:
    clearance = circle_clearance(P_xy, P_yaw, ego, S_xy, S_yaw, dims)
    c, s = torch.cos(P_yaw), torch.sin(P_yaw)  # (B, T)
    d = S_xy[None] - P_xy[:, None]  # (B, N, T, 2)
    rel_pos = torch.stack([d[..., 0] * c[:, None] + d[..., 1] * s[:, None],
                           -d[..., 0] * s[:, None] + d[..., 1] * c[:, None]], -1)
    rel_vel = torch.diff(rel_pos, dim=2, prepend=rel_pos[:, :, :1]) / dt
    overlap = predicted_overlap(P_xy, P_yaw, ego, S_xy, S_yaw, dims, valid)
    return Interaction(clearance, rel_pos, rel_vel, valid, overlap)
