"""Which entities enter the explicit correction pathway: R(r) from reasoning, or the Table 4(a)
heuristics used in the paper's intervention audit."""
from __future__ import annotations

import numpy as np
import torch

from groundact_repro.metrics import ActorSet


def _valid_at_t0(actors: ActorSet) -> np.ndarray:
    return actors.valid[:, 0]


def select_none(actors: ActorSet, **_) -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


def select_all(actors: ActorSet, **_) -> np.ndarray:
    return np.nonzero(_valid_at_t0(actors))[0]


def select_nearest(actors: ActorSet, k: int = 1, **_) -> np.ndarray:
    """The k actors closest to the ego origin at t0."""
    v = _valid_at_t0(actors)
    if not v.any():
        return np.zeros(0, dtype=np.int64)
    d = np.linalg.norm(actors.xy[:, 0], axis=-1)
    d[~v] = np.inf
    return np.argsort(d)[:k]


def select_highest_risk(actors: ActorSet, base_xy: np.ndarray, base_yaw: np.ndarray, ego_dims,
                        k: int = 1, horizon: int = 64, **_) -> np.ndarray:
    """The k actors with the smallest predicted clearance to the base ego plan over the horizon.

    base_xy (S, T, 2) / base_yaw (S, T): clearance is taken over all S candidates.
    """
    from groundact_repro.groundact.interaction import circle_clearance
    v = actors.valid[:, 1:horizon + 1]
    if not v.any():
        return np.zeros(0, dtype=np.int64)
    P = torch.as_tensor(base_xy[:, :horizon], dtype=torch.float32)
    Py = torch.as_tensor(base_yaw[:, :horizon], dtype=torch.float32)
    Sxy = torch.as_tensor(actors.xy[:, 1:horizon + 1], dtype=torch.float32)
    Sy = torch.as_tensor(actors.yaw[:, 1:horizon + 1], dtype=torch.float32)
    dims = torch.as_tensor(actors.dims, dtype=torch.float32)
    c = circle_clearance(P, Py, ego_dims, Sxy, Sy, dims)  # (S, N, T)
    c = c.masked_fill(~torch.as_tensor(v)[None], float("inf"))
    risk = -c.amin(dim=(0, 2)).numpy()  # highest risk = smallest clearance
    return np.argsort(-risk)[:k]


def select_matched_distractor(actors: ActorSet, target: int, base_xy: np.ndarray, base_yaw,
                              ego_dims, min_clearance: float = 1.0, **_) -> np.ndarray:
    """Another same-class entity at a similar distance with >= 1 m clearance from the base plan."""
    from groundact_repro.groundact.interaction import circle_clearance
    v = _valid_at_t0(actors)
    if actors.n == 0 or not v[target]:
        return np.zeros(0, dtype=np.int64)
    d = np.linalg.norm(actors.xy[:, 0], axis=-1)
    same = np.array([c == actors.classes[target] for c in actors.classes]) & v
    same[target] = False
    if not same.any():
        return np.zeros(0, dtype=np.int64)
    T = base_xy.shape[1]
    c = circle_clearance(torch.as_tensor(base_xy, dtype=torch.float32),
                         torch.as_tensor(base_yaw, dtype=torch.float32), ego_dims,
                         torch.as_tensor(actors.xy[:, 1:T + 1], dtype=torch.float32),
                         torch.as_tensor(actors.yaw[:, 1:T + 1], dtype=torch.float32),
                         torch.as_tensor(actors.dims, dtype=torch.float32))
    c = c.masked_fill(~torch.as_tensor(actors.valid[:, 1:T + 1])[None], float("inf")).amin(dim=(0, 2)).numpy()
    ok = same & (c >= min_clearance)
    if not ok.any():
        return np.zeros(0, dtype=np.int64)
    cand = np.nonzero(ok)[0]
    return cand[[np.argmin(np.abs(d[cand] - d[target]))]]


def select_from_references(actors: ActorSet, references: list[int], **_) -> np.ndarray:
    """R(r): indices referenced in the reasoning sequence (deduplicated, valid only)."""
    v = _valid_at_t0(actors)
    out = [i for i in dict.fromkeys(references) if 0 <= i < actors.n and v[i]]
    return np.array(out, dtype=np.int64)


SELECTORS = {
    "none": select_none,
    "all": select_all,
    "nearest": select_nearest,
    "highest_risk": select_highest_risk,
    "references": select_from_references,
}
