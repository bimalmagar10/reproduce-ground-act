"""Open-loop planning metrics of GroundAct, Appendix G / Table 10.

Every model produces S candidate trajectories (64 poses at 10 Hz) per sample. Each metric is
computed per candidate, averaged over the S candidates of the sample, and then over samples.

* ADE (m) at 2.5 / 5.0 / 6.4 s: mean ground-plane distance between predicted and GT ego positions.
* Corner distance (m) at 6.4 s: the same distance averaged over the four corners of the oriented
  ego box.
* Collision (%) at 2.5 / 5.0 s: fraction of candidates whose predicted ego footprint intersects any
  annotated actor footprint at any of the discrete poses within the horizon (no at-fault filter).
* Off-road: needs map data, which the public PAI-AV release does not contain -> not computed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from groundact_repro.geometry import box_corners, obb_intersects, yaw_from_rot

HORIZON_STEPS = {"2.5": 25, "5.0": 50, "6.4": 64}


@dataclass
class EgoDims:
    length: float
    width: float
    rear_axle_to_bbox_center: float  # ego frame origin (rig) -> box center offset along +x

    def box_center(self, xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        off = self.rear_axle_to_bbox_center
        return xy + off * np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)


@dataclass
class ActorSet:
    """Annotated actor tracks resampled on the 10 Hz prediction grid, in the ego frame at t0.

    Index 0 along the time axis is t0; indices 1..T are the T future steps.
    """
    track_ids: list
    classes: list
    dims: np.ndarray  # (N, 2) length, width
    xy: np.ndarray  # (N, T+1, 2)
    yaw: np.ndarray  # (N, T+1)
    valid: np.ndarray  # (N, T+1) bool
    hist_xy: np.ndarray  # (N, H, 2) observed frames t0-(H-1)*dt .. t0
    hist_yaw: np.ndarray  # (N, H)
    hist_valid: np.ndarray  # (N, H)

    @property
    def n(self) -> int:
        return self.xy.shape[0]

    @staticmethod
    def empty(n_future: int = 64, n_hist: int = 4) -> "ActorSet":
        return ActorSet([], [], np.zeros((0, 2)), np.zeros((0, n_future + 1, 2)),
                        np.zeros((0, n_future + 1)), np.zeros((0, n_future + 1), bool),
                        np.zeros((0, n_hist, 2)), np.zeros((0, n_hist)), np.zeros((0, n_hist), bool))


def ade(pred_xy: np.ndarray, gt_xy: np.ndarray, steps: int) -> np.ndarray:
    """pred_xy (S, T, 2), gt_xy (T, 2) -> (S,) mean displacement over the first `steps` poses."""
    d = np.linalg.norm(pred_xy[:, :steps] - gt_xy[None, :steps], axis=-1)
    return d.mean(-1)


def corner_distance(pred_xy, pred_yaw, gt_xy, gt_yaw, ego: EgoDims, steps: int = 64) -> np.ndarray:
    """Mean distance between the four corners of the predicted and GT oriented ego boxes, (S,)."""
    pc = box_corners(ego.box_center(pred_xy[:, :steps], pred_yaw[:, :steps]), pred_yaw[:, :steps],
                     ego.length, ego.width)  # (S, T, 4, 2)
    gc = box_corners(ego.box_center(gt_xy[:steps], gt_yaw[:steps]), gt_yaw[:steps],
                     ego.length, ego.width)  # (T, 4, 2)
    return np.linalg.norm(pc - gc[None], axis=-1).mean(axis=(1, 2))


def collision(pred_xy, pred_yaw, actors: ActorSet, ego: EgoDims, steps: int) -> np.ndarray:
    """(S,) bool: whether the predicted ego footprint overlaps any valid actor box within `steps`."""
    S = pred_xy.shape[0]
    if actors.n == 0:
        return np.zeros(S, dtype=bool)
    t = slice(1, steps + 1)  # future poses 1..steps align with actors.xy[:, 1:steps+1]
    ego_c = ego.box_center(pred_xy[:, :steps], pred_yaw[:, :steps])  # (S, T, 2)
    hit = obb_intersects(
        ego_c[:, None], pred_yaw[:, None, :steps], ego.length, ego.width,
        actors.xy[None, :, t], actors.yaw[None, :, t],
        actors.dims[None, :, None, 0], actors.dims[None, :, None, 1],
    )  # (S, N, T)
    hit &= actors.valid[None, :, t]
    return hit.any(axis=(1, 2))


def evaluate_candidates(pred_xyz, pred_rot, gt_xyz, gt_rot, actors: ActorSet | None,
                        ego: EgoDims) -> dict:
    """All per-sample metrics (averaged over the S candidates).

    pred_xyz (S, 64, 3), pred_rot (S, 64, 3, 3), gt_xyz (64, 3), gt_rot (64, 3, 3).
    """
    pred_xy = np.asarray(pred_xyz, dtype=np.float64)[..., :2]
    gt_xy = np.asarray(gt_xyz, dtype=np.float64)[..., :2]
    pred_yaw = yaw_from_rot(np.asarray(pred_rot, dtype=np.float64))
    gt_yaw = yaw_from_rot(np.asarray(gt_rot, dtype=np.float64))
    out = {}
    for name, steps in HORIZON_STEPS.items():
        a = ade(pred_xy, gt_xy, steps)
        out[f"ade_{name}"] = float(a.mean())
        out[f"min_ade_{name}"] = float(a.min())
    out["corner_6.4"] = float(corner_distance(pred_xy, pred_yaw, gt_xy, gt_yaw, ego).mean())
    out["has_actor_labels"] = actors is not None
    if actors is not None:
        for name in ("2.5", "5.0"):
            c = collision(pred_xy, pred_yaw, actors, ego, HORIZON_STEPS[name])
            out[f"collision_{name}"] = float(c.mean())
            out[f"collision_{name}_any"] = bool(c.any())
    return out


def summarize(rows: list[dict]) -> dict:
    """Average per-sample metrics over samples (collision over the annotation-available subset)."""
    keys = ["ade_2.5", "ade_5.0", "ade_6.4", "min_ade_6.4", "corner_6.4"]
    out = {"n_samples": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = float(np.mean(vals))
    coll_rows = [r for r in rows if r.get("has_actor_labels")]
    out["n_collision_samples"] = len(coll_rows)
    for k in ("collision_2.5", "collision_5.0"):
        vals = [r[k] for r in coll_rows if k in r]
        if vals:
            out[k + "_pct"] = 100.0 * float(np.mean(vals))
    return out
