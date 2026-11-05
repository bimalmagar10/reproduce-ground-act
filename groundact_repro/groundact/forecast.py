"""Entity forecasts S_i over the planning horizon (Section 3.4, forecast-based formulation)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from groundact_repro.metrics import ActorSet


@dataclass
class SceneEntities:
    """Torch view of the entities of one scene (ego frame at t0)."""
    hist_xy: torch.Tensor  # (N, H, 2) observed frames t0-(H-1)dt .. t0
    hist_yaw: torch.Tensor  # (N, H)
    hist_valid: torch.Tensor  # (N, H) bool
    dims: torch.Tensor  # (N, 2) length, width
    valid: torch.Tensor  # (N,) bool, valid at t0
    future_xy: torch.Tensor | None = None  # (N, T, 2) annotated future (oracle / supervision)
    future_yaw: torch.Tensor | None = None
    future_valid: torch.Tensor | None = None
    states: torch.Tensor | None = None  # (N, D) entity states M_i when available

    @property
    def n(self) -> int:
        return self.dims.shape[0]

    @staticmethod
    def from_actors(actors: ActorSet, device="cpu") -> "SceneEntities":
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        return SceneEntities(
            hist_xy=t(actors.hist_xy), hist_yaw=t(actors.hist_yaw), hist_valid=t(actors.hist_valid, torch.bool),
            dims=t(actors.dims), valid=t(actors.valid[:, 0], torch.bool),
            future_xy=t(actors.xy[:, 1:]), future_yaw=t(actors.yaw[:, 1:]),
            future_valid=t(actors.valid[:, 1:], torch.bool))


@dataclass
class Forecast:
    xy: torch.Tensor  # (N, T, 2)
    yaw: torch.Tensor  # (N, T)
    valid: torch.Tensor  # (N, T) bool


class ConstantVelocityForecaster:
    """Training-free stand-in: extrapolate the last two valid observed frames at constant velocity."""

    def __init__(self, horizon: int = 64, dt: float = 0.1):
        self.horizon, self.dt = horizon, dt

    def __call__(self, scene: SceneEntities, proposal=None) -> Forecast:
        N, H = scene.hist_xy.shape[:2]
        dev = scene.hist_xy.device
        last = scene.hist_xy[:, -1]
        v = torch.zeros_like(last)
        valid = scene.hist_valid[:, -1].clone()
        for h in range(H - 2, -1, -1):  # most recent earlier valid frame
            ok = scene.hist_valid[:, h] & valid & (v.abs().sum(-1) == 0)
            if ok.any():
                v[ok] = (last[ok] - scene.hist_xy[ok, h]) / ((H - 1 - h) * self.dt)
        steps = torch.arange(1, self.horizon + 1, device=dev, dtype=torch.float32) * self.dt
        xy = last[:, None] + v[:, None] * steps[None, :, None]
        yaw = scene.hist_yaw[:, -1:].expand(-1, self.horizon)
        return Forecast(xy, yaw, valid[:, None].expand(-1, self.horizon).clone())


class OracleForecaster:
    """Upper bound for analysis: the annotated future motion."""

    def __call__(self, scene: SceneEntities, proposal=None) -> Forecast:
        if scene.future_xy is None:
            raise ValueError("scene has no annotated future")
        return Forecast(scene.future_xy, scene.future_yaw, scene.future_valid)


class LearnedForecaster(nn.Module):
    """The paper's prediction module: forecasts from four observed frames, the entity state and
    (joint variant) the current ego proposal. Supervised with L_future (Eq. 12). Untrained here."""

    def __init__(self, horizon: int = 64, n_hist: int = 4, state_size: int | None = None,
                 hidden: int = 256, joint: bool = True):
        super().__init__()
        self.horizon, self.joint = horizon, joint
        in_dim = n_hist * 3 + 2 + (state_size or 0) + (horizon * 3 if joint else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, horizon * 3))

    def forward(self, scene: SceneEntities, proposal: tuple[torch.Tensor, torch.Tensor] | None = None) -> Forecast:
        N = scene.n
        feats = [scene.hist_xy.flatten(1), scene.hist_yaw, scene.dims]
        if scene.states is not None:
            feats.append(scene.states)
        if self.joint:
            if proposal is None:
                raise ValueError("joint forecaster needs the ego proposal (xy, yaw)")
            pxy, pyaw = proposal  # (T, 2), (T,) a single proposal row
            feats.append(torch.cat([pxy.flatten(), pyaw]).expand(N, -1))
        out = self.net(torch.cat(feats, -1)).view(N, self.horizon, 3)
        last = scene.hist_xy[:, -1:]
        xy = last + out[..., :2].cumsum(1)
        yaw = scene.hist_yaw[:, -1:] + out[..., 2].cumsum(1)
        return Forecast(xy, yaw, scene.valid[:, None].expand(-1, self.horizon).clone())
