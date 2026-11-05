"""Reference-driven planning (Section 3.4): selective residual correction of the flow field.

    v_GroundAct^(k) = v_base^(k) + gamma^(k) R_phi(x^(k), v_base^(k), z_R^(k))
    gamma^(k)       = b_risk * min(1, kappa s^2)        (risk latch x flow-time schedule, Eq. 14)
    z_R^(k)         = SetEnc({M_i, S_i^(k), C_i^(k)}_{i in R(r)})

At solver step k the base velocity is evaluated at the flow state x^(k); the endpoint estimate
x^(k) + (1 - s) v_base^(k) is decoded through the unicycle dynamics into the ego proposal P^(k);
the referenced entities' forecasts S_i are evaluated against it (clearance, relative motion,
validity, exact predicted overlap); the latch activates on predicted overlap and stays on for the
remaining steps; the corrected field is integrated. The pathway is bypassed when no valid entity
is referenced.

Two residuals are provided:
  * `ResidualAdapter` + `SetEncoder`: the learned R_phi of the paper (untrained unless a
    checkpoint is loaded).
  * `ClearanceGuidanceResidual`: a training-free stand-in that builds the alternative endpoint a+
    the way the paper's teacher does (Appendix C: clearance gain >= 0.5 m, progress ratio >= 0.95,
    no newly overlapping entity) and converts it to a field with u* = (a+ - x_s) / max(1 - s, delta).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import einops
import numpy as np
import torch
from torch import nn

from groundact_repro.groundact.forecast import ConstantVelocityForecaster, Forecast, SceneEntities
from groundact_repro.groundact.interaction import Interaction, circle_clearance, interact, predicted_overlap
from groundact_repro.metrics import EgoDims


def flow_time_schedule(s: torch.Tensor, kappa: float = 2.0) -> torch.Tensor:
    return torch.clamp(kappa * s ** 2, max=1.0)


def endpoint_estimate(x: torch.Tensor, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """x_1 estimate under the linear flow x_s = (1 - s) eps + s a with v = a - eps."""
    return x + (1.0 - s) * v


# ----------------------------------------------------------------------------- learned pathway

class SetEncoder(nn.Module):
    """Permutation-invariant encoder of {M_i, S_i, C_i} over the referenced set."""

    def __init__(self, horizon: int = 64, state_size: int | None = None, hidden: int = 256, out: int = 256):
        super().__init__()
        per_t = 1 + 2 + 2 + 1  # clearance, rel_pos, rel_vel, validity
        in_dim = horizon * (per_t + 3) + (state_size or 0)  # + forecast xy, yaw per t
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(2 * hidden, out), nn.GELU())
        self.out = out

    def forward(self, inter: Interaction, fc: Forecast, states: torch.Tensor | None,
                selected: torch.Tensor) -> torch.Tensor:
        """-> z_R (B, out); zero when the referenced set is empty."""
        B = inter.clearance.shape[0]
        if selected.numel() == 0:
            return inter.clearance.new_zeros(B, self.out)
        sel = selected
        feats = [inter.clearance[:, sel], inter.rel_pos[:, sel].flatten(2), inter.rel_vel[:, sel].flatten(2),
                 inter.valid[sel].float()[None].expand(B, -1, -1),
                 fc.xy[sel].flatten(1)[None].expand(B, -1, -1), fc.yaw[sel][None].expand(B, -1, -1)]
        if states is not None:
            feats.append(states[sel][None].expand(B, -1, -1))
        h = self.phi(torch.cat(feats, -1))  # (B, R, hidden)
        return self.rho(torch.cat([h.mean(1), h.amax(1)], -1))


class ResidualAdapter(nn.Module):
    """R_phi(x, v_base, z_R) -> correction of the base field, (B, T, 2)."""

    def __init__(self, horizon: int = 64, z_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(2 * horizon * 2 + z_dim + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # starts as the identity (no correction)

    def forward(self, x: torch.Tensor, v_base: torch.Tensor, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        s_b = torch.as_tensor(s, dtype=x.dtype, device=x.device).reshape(-1)
        s_b = s_b.expand(B) if s_b.numel() == 1 else s_b
        inp = torch.cat([x.flatten(1), v_base.flatten(1), z, s_b.view(B, 1)], -1)
        return self.net(inp).view(B, self.horizon, 2)


class LearnedResidual:
    """Wraps SetEncoder + ResidualAdapter behind the residual interface used by the sampler."""

    def __init__(self, set_encoder: SetEncoder, adapter: ResidualAdapter):
        self.enc, self.adapter = set_encoder, adapter

    def __call__(self, ctx: "StepContext") -> tuple[torch.Tensor, dict]:
        z = self.enc(ctx.interaction, ctx.forecast, ctx.scene.states, ctx.selected)
        return self.adapter(ctx.x, ctx.v_base, z, ctx.s), {}


# ----------------------------------------------------------------------------- training-free pathway

class ClearanceGuidanceResidual:
    """Teacher-style alternative endpoint from clearance guidance (training-free stand-in).

    a+ is found by a few RMS-normalized gradient steps on sum_t softplus(margin - clearance_it)
    over the referenced entities, through the unicycle decode. It is kept only when it passes the
    paper's screens; otherwise the residual is zero for that row.
    """

    def __init__(self, margin: float = 1.0, n_inner: int = 10, lr: float = 0.05, delta: float = 0.1,
                 min_gain: float = 0.5, min_progress_ratio: float = 0.95):
        self.margin, self.n_inner, self.lr, self.delta = margin, n_inner, lr, delta
        self.min_gain, self.min_progress = min_gain, min_progress_ratio

    @staticmethod
    def _progress(xy: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(torch.diff(xy, dim=1), dim=-1).sum(-1)

    def __call__(self, ctx: "StepContext") -> tuple[torch.Tensor, dict]:
        B = ctx.x.shape[0]
        zero = torch.zeros_like(ctx.v_base)
        if ctx.selected.numel() == 0:
            return zero, {"accepted": 0}
        sel = ctx.selected
        fc = ctx.forecast
        S_xy, S_yaw, dims, valid = fc.xy[sel], fc.yaw[sel], ctx.scene.dims[sel], fc.valid[sel]
        with torch.enable_grad():
            a = ctx.x_hat.detach().clone().requires_grad_(True)
            for _ in range(self.n_inner):
                P_xy, P_yaw = ctx.decode(a)
                c = circle_clearance(P_xy, P_yaw, ctx.ego, S_xy, S_yaw, dims)  # (B, R, T)
                J = (torch.nn.functional.softplus(self.margin - c) * valid[None]).sum()
                if J.item() <= 1e-6:
                    break
                (g,) = torch.autograd.grad(J, a)
                rms = g.flatten(1).pow(2).mean(-1).sqrt().clamp_min(1e-8)
                with torch.no_grad():
                    a -= self.lr * g / rms.view(B, 1, 1)
        a = a.detach()
        # --- teacher screens (Appendix C)
        base_xy, base_yaw = ctx.decode(ctx.x_hat)
        alt_xy, alt_yaw = ctx.decode(a)
        base_c = circle_clearance(base_xy, base_yaw, ctx.ego, S_xy, S_yaw, dims).masked_fill(~valid[None], float("inf")).amin(dim=(1, 2))
        alt_c = circle_clearance(alt_xy, alt_yaw, ctx.ego, S_xy, S_yaw, dims).masked_fill(~valid[None], float("inf")).amin(dim=(1, 2))
        gain_ok = (alt_c - base_c >= self.min_gain) | ((base_c <= 0) & (alt_c > 0))
        prog_ok = self._progress(alt_xy) >= self.min_progress * self._progress(base_xy)
        all_valid = ctx.scene.valid
        base_ov = predicted_overlap(base_xy, base_yaw, ctx.ego, fc.xy, fc.yaw, ctx.scene.dims, fc.valid & all_valid[:, None])
        alt_ov = predicted_overlap(alt_xy, alt_yaw, ctx.ego, fc.xy, fc.yaw, ctx.scene.dims, fc.valid & all_valid[:, None])
        no_new = ~(alt_ov & ~base_ov).any(-1)
        accept = gain_ok & prog_ok & no_new & torch.isfinite(a).flatten(1).all(-1)
        u_star = (a - ctx.x) / max(1.0 - float(ctx.s), self.delta)
        r = torch.where(accept.view(B, 1, 1), u_star - ctx.v_base, zero)
        return r, {"accepted": int(accept.sum()), "base_min_clearance": base_c.tolist(),
                   "alt_min_clearance": alt_c.tolist()}


# ----------------------------------------------------------------------------- sampler

@dataclass
class StepContext:
    x: torch.Tensor  # (B, T, 2) flow state
    v_base: torch.Tensor  # (B, T, 2)
    s: torch.Tensor  # scalar flow time
    x_hat: torch.Tensor  # endpoint estimate
    decode: callable  # action (B, T, 2) -> (xy (B, T, 2), yaw (B, T))
    scene: SceneEntities
    forecast: Forecast
    interaction: Interaction
    selected: torch.Tensor  # (R,) long
    ego: EgoDims


@dataclass
class CorrectionInfo:
    selected: list = field(default_factory=list)
    latch_step: list = field(default_factory=list)  # per row, -1 if never active
    accepted_steps: int = 0
    residual_norm: list = field(default_factory=list)
    bypassed: bool = False


class GroundActCorrector:
    """Drop-in replacement for `FlowMatching.sample` that integrates the corrected field.

    Call `set_scene` before each sample, then use it through `corrected_sampling`.
    """

    def __init__(self, residual=None, forecaster=None, selector: str = "nearest", select_k: int = 1,
                 kappa: float = 2.0, two_pass: bool = True):
        self.residual = residual or ClearanceGuidanceResidual()
        self.forecaster = forecaster or ConstantVelocityForecaster()
        self.selector, self.select_k, self.kappa, self.two_pass = selector, select_k, kappa, two_pass
        self.scene: SceneEntities | None = None
        self.ego: EgoDims | None = None
        self.references: list[int] | None = None
        self.decode = None
        self.n_steps = 10
        self.base_xyz: np.ndarray | None = None
        self.info = CorrectionInfo()

    def set_scene(self, scene: SceneEntities | None, ego: EgoDims, references: list[int] | None = None):
        self.scene, self.ego, self.references = scene, ego, references

    # --- selection ---------------------------------------------------------------
    def _select(self, base_xy, base_yaw) -> torch.Tensor:
        from groundact_repro.groundact import selection as sel_mod
        from groundact_repro.metrics import ActorSet
        sc = self.scene
        dev = sc.dims.device
        if sc is None or sc.n == 0:
            return torch.zeros(0, dtype=torch.long, device=dev)
        # Build a light ActorSet view for the heuristic selectors.
        T = sc.future_xy.shape[1] if sc.future_xy is not None else base_xy.shape[1]
        fc = self.forecaster(sc)
        xy = torch.cat([sc.hist_xy[:, -1:], fc.xy], 1).cpu().numpy()
        yaw = torch.cat([sc.hist_yaw[:, -1:], fc.yaw], 1).cpu().numpy()
        valid = torch.cat([sc.valid[:, None], fc.valid], 1).cpu().numpy()
        actors = ActorSet([str(i) for i in range(sc.n)], ["entity"] * sc.n, sc.dims.cpu().numpy(), xy, yaw, valid,
                          sc.hist_xy.cpu().numpy(), sc.hist_yaw.cpu().numpy(), sc.hist_valid.cpu().numpy())
        if self.selector == "references":
            idx = sel_mod.select_from_references(actors, self.references or [])
        elif self.selector == "highest_risk":
            idx = sel_mod.select_highest_risk(actors, base_xy.cpu().numpy(), base_yaw.cpu().numpy(),
                                              self.ego, k=self.select_k, horizon=T)
        else:
            idx = sel_mod.SELECTORS[self.selector](actors, k=self.select_k)
        return torch.as_tensor(np.asarray(idx, dtype=np.int64), device=dev)

    # --- integration -------------------------------------------------------------
    def _euler(self, x0, step_fn, n_steps, selected, correct: bool):
        x = x0.clone()
        B = x.shape[0]
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device)
        latch = torch.zeros(B, dtype=torch.bool, device=x.device)
        latch_step = torch.full((B,), -1, dtype=torch.long, device=x.device)
        for i in range(n_steps):
            s, dt = ts[i], ts[i + 1] - ts[i]
            s_b = s.view(1, 1, 1).expand(B, 1, 1)
            v_base = step_fn(x=x, t=s_b).float()
            v = v_base
            if correct and selected.numel() > 0:
                with torch.autocast("cuda", enabled=False):
                    x_hat = endpoint_estimate(x.float(), v_base, s)
                    P_xy, P_yaw = self.decode(x_hat)
                    fc = self.forecaster(self.scene, (P_xy[0], P_yaw[0]))
                    inter = interact(P_xy, P_yaw, self.ego, fc.xy, fc.yaw, self.scene.dims, fc.valid)
                    newly = inter.overlap[:, selected].any(-1) & ~latch
                    latch_step[newly] = i
                    latch |= newly
                    gamma = latch.float() * flow_time_schedule(s, self.kappa)
                    if latch.any():
                        ctx = StepContext(x.float(), v_base, s, x_hat, self.decode, self.scene, fc, inter,
                                          selected, self.ego)
                        r, rinfo = self.residual(ctx)
                        self.info.accepted_steps += int(rinfo.get("accepted", 0))
                        self.info.residual_norm.append(float((gamma.view(B, 1, 1) * r).norm()))
                        v = v_base + gamma.view(B, 1, 1) * r
            x = x + dt * v.to(x.dtype)
        return x, latch_step

    def sample(self, batch_size: int, step_fn, device=None, return_all_steps: bool = False,
               inference_step: int | None = None, temperature: float = 1.0, x_dims=None, **_):
        n_steps = inference_step or self.n_steps
        x0 = torch.randn(batch_size, *x_dims, device=device) * temperature
        self.info = CorrectionInfo()
        if self.scene is None or self.scene.n == 0 or self.decode is None:
            self.info.bypassed = True
            x, _ = self._euler(x0, step_fn, n_steps, torch.zeros(0, dtype=torch.long), correct=False)
            return x
        base, _ = self._euler(x0, step_fn, n_steps, torch.zeros(0, dtype=torch.long), correct=False)
        with torch.autocast("cuda", enabled=False):
            base_xy, base_yaw = self.decode(base.float())
        self.base_xyz = torch.cat([base_xy, torch.zeros_like(base_xy[..., :1])], -1).cpu().numpy()
        selected = self._select(base_xy, base_yaw)
        self.info.selected = selected.tolist()
        if selected.numel() == 0:
            self.info.bypassed = True
            return base
        x, latch_step = self._euler(x0, step_fn, n_steps, selected, correct=True)
        self.info.latch_step = latch_step.tolist()
        return x


class _Handle:
    def __init__(self, corrector):
        self.corrector = corrector

    @property
    def base_xyz(self):
        return self.corrector.base_xyz

    @property
    def info(self):
        c = self.corrector.info
        return {"selected": c.selected, "latch_step": c.latch_step, "accepted_steps": c.accepted_steps,
                "residual_norm": c.residual_norm, "bypassed": c.bypassed}


@contextlib.contextmanager
def corrected_sampling(model, corrector: GroundActCorrector, inputs: dict):
    """Temporarily route the model's flow sampler through the corrector (Alpamayo R1 / 1.5)."""
    diffusion = model.diffusion
    hist_xyz, hist_rot = inputs["ego_history_xyz"][:, -1], inputs["ego_history_rot"][:, -1]  # (1, 16, 3)
    x_dims = diffusion.x_dims
    action_space = model.action_space

    def decode(action: torch.Tensor):
        B = action.shape[0]
        hx = einops.repeat(hist_xyz, "b ... -> (b n) ...", n=B // hist_xyz.shape[0]).float()
        hr = einops.repeat(hist_rot, "b ... -> (b n) ...", n=B // hist_rot.shape[0]).float()
        xyz, rot = action_space.action_to_traj(action.float(), hx, hr)
        return xyz[..., :2], torch.atan2(rot[..., 1, 0], rot[..., 0, 0])

    corrector.decode = decode
    corrector.n_steps = getattr(diffusion, "num_inference_steps", 10)
    original = diffusion.sample

    def patched(batch_size, step_fn, device=None, return_all_steps=False, **kw):
        return corrector.sample(batch_size, step_fn, device=device, return_all_steps=return_all_steps,
                                x_dims=x_dims, **kw)

    diffusion.sample = patched
    try:
        yield _Handle(corrector)
    finally:
        diffusion.sample = original
        corrector.decode = None
