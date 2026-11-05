import numpy as np
import torch

from groundact_repro.groundact.correction import (ClearanceGuidanceResidual, GroundActCorrector, LearnedResidual,
                                                  ResidualAdapter, SetEncoder, StepContext, endpoint_estimate,
                                                  flow_time_schedule)
from groundact_repro.groundact.entities import EntityBank, EntityHeads, read_entity_states
from groundact_repro.groundact.forecast import ConstantVelocityForecaster, OracleForecaster, SceneEntities
from groundact_repro.groundact.interaction import circle_clearance, interact
from groundact_repro.groundact.pointer import GatedPointerHead, ReferenceConsumer, parse_references, resolve_reference
from groundact_repro.metrics import ActorSet, EgoDims

EGO = EgoDims(4.872, 2.121, 1.327)
T = 64


def test_pointer_distribution_sums_to_one_and_masks_null():
    torch.manual_seed(0)
    head = GatedPointerHead(hidden_size=32, entity_size=16, proj_size=8)
    h = torch.randn(3, 32); M = torch.randn(3, 5, 16); logits = torch.randn(3, 50)
    valid = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1]]).bool()
    d = head(h, logits, M, valid)
    total = torch.cat([d.log_p_vocab, d.log_p_ref], -1).exp().sum(-1)
    assert torch.allclose(total, torch.ones(3), atol=1e-5)
    assert torch.allclose(d.log_p_ref.exp().sum(-1), d.gate, atol=1e-5)
    assert d.log_p_ref[0, 2:].exp().max() < 1e-6  # null slots excluded when non-null exist
    assert d.log_p_ref[1].exp().sum() > 0  # all-null: no masking, pointer still defined
    is_ref, idx = d.sample()
    assert is_ref.shape == (3,) and idx.shape == (3,)
    assert head.gate_loss(h, torch.tensor([True, False, True])).item() > 0
    assert head.association_loss(h, M, torch.tensor([1, 0, 4]), torch.tensor([True, True, False])).item() > 0


def test_reference_resolution_and_consumption():
    torch.manual_seed(0)
    M = torch.randn(6, 16)
    bank = EntityBank(M, torch.ones(6, dtype=torch.bool))
    assert torch.equal(resolve_reference(bank, torch.tensor([4])), M[4:5])
    cons = ReferenceConsumer(entity_size=16, embed_size=32, num_entities=6, alpha=0.5)
    u = cons(bank, torch.tensor([4, 1]))
    assert u.shape == (2, 32)
    assert parse_references("vehicle <|entity_6|> and <|entity_41|><|entity_6|>") == [6, 41, 6]
    heads = EntityHeads(16, num_classes=3)
    rb = heads.read(M)
    assert rb.valid.shape == (6,) and rb.boxes.shape == (6, 6)
    hs = torch.randn(2, 10, 16)
    assert read_entity_states(hs, torch.tensor([[1, 3], [0, 9]])).shape == (2, 2, 16)


def _scene(car_x=20.0, n=1):
    xy = np.zeros((n, T + 1, 2)); xy[:, :, 0] = car_x
    hist = np.zeros((n, 4, 2)); hist[:, :, 0] = car_x
    actors = ActorSet([str(i) for i in range(n)], ["automobile"] * n, np.tile([[4.5, 1.9]], (n, 1)), xy,
                      np.zeros((n, T + 1)), np.ones((n, T + 1), bool), hist, np.zeros((n, 4)), np.ones((n, 4), bool))
    return SceneEntities.from_actors(actors)


def test_forecasters_and_interaction():
    sc = _scene()
    cv = ConstantVelocityForecaster()(sc)
    assert cv.xy.shape == (1, T, 2) and torch.allclose(cv.xy[0, -1], torch.tensor([20.0, 0.0]))
    oc = OracleForecaster()(sc)
    assert torch.allclose(oc.xy, sc.future_xy)
    t = torch.arange(1, T + 1) * 0.1
    P = torch.stack([10 * t, torch.zeros(T)], -1)[None]  # ego drives through the parked car
    inter = interact(P, torch.zeros(1, T), EGO, cv.xy, cv.yaw, sc.dims, cv.valid)
    assert inter.overlap.tolist() == [[True]]
    assert inter.min_clearance()[0, 0] < 0
    P2 = P.clone(); P2[..., 1] = 5.0
    assert circle_clearance(P2, torch.zeros(1, T), EGO, cv.xy, cv.yaw, sc.dims).min() > 1.0


def test_schedule_and_endpoint():
    s = torch.tensor([0.0, 0.5, 0.8, 1.0])
    assert torch.allclose(flow_time_schedule(s), torch.tensor([0.0, 0.5, 1.0, 1.0]))
    x, v = torch.ones(2, T, 2), torch.ones(2, T, 2)
    assert torch.allclose(endpoint_estimate(x, v, torch.tensor(0.25)), 1.75 * torch.ones(2, T, 2))


class ToyActionSpace:
    """x,y = cumulative sums of the action: a stand-in for the unicycle decode (differentiable)."""

    def __call__(self, a):
        xy = a.cumsum(1)
        yaw = torch.atan2(torch.diff(xy[..., 1], dim=1, prepend=xy[..., 1:2, 0] * 0),
                          torch.diff(xy[..., 0], dim=1, prepend=xy[..., 0:1, 0] * 0).clamp_min(1e-3))
        return xy, yaw


def test_clearance_guidance_residual_moves_plan_away():
    torch.manual_seed(0)
    sc = _scene(car_x=20.0)
    dec = ToyActionSpace()
    x = torch.zeros(1, T, 2); x[..., 0] = 1.0  # 1 m per step straight through the car
    v = torch.zeros(1, T, 2)
    xh = endpoint_estimate(x, v, torch.tensor(0.5))
    fc = ConstantVelocityForecaster()(sc)
    P_xy, P_yaw = dec(xh)
    inter = interact(P_xy, P_yaw, EGO, fc.xy, fc.yaw, sc.dims, fc.valid)
    ctx = StepContext(x, v, torch.tensor(0.5), xh, dec, sc, fc, inter, torch.tensor([0]), EGO)
    r, info = ClearanceGuidanceResidual(margin=1.0, n_inner=30, lr=0.05, min_progress_ratio=0.0)(ctx)
    assert r.shape == (1, T, 2)
    assert info["alt_min_clearance"][0] > info["base_min_clearance"][0]
    # with no referenced entity: zero residual
    ctx0 = StepContext(x, v, torch.tensor(0.5), xh, dec, sc, fc, inter, torch.zeros(0, dtype=torch.long), EGO)
    assert ClearanceGuidanceResidual()(ctx0)[0].abs().sum() == 0


def test_learned_residual_shapes():
    sc = _scene(n=3)
    dec = ToyActionSpace()
    x = torch.randn(2, T, 2); v = torch.randn(2, T, 2)
    fc = ConstantVelocityForecaster()(sc)
    P_xy, P_yaw = dec(x)
    inter = interact(P_xy, P_yaw, EGO, fc.xy, fc.yaw, sc.dims, fc.valid)
    res = LearnedResidual(SetEncoder(T), ResidualAdapter(T))
    ctx = StepContext(x, v, torch.tensor(0.3), x, dec, sc, fc, inter, torch.tensor([0, 2]), EGO)
    r, _ = res(ctx)
    assert r.shape == (2, T, 2) and r.abs().sum() == 0  # zero-init adapter = identity field


def test_corrector_two_pass_with_toy_step_fn():
    torch.manual_seed(0)
    sc = _scene(car_x=10.0)
    dec = ToyActionSpace()
    corr = GroundActCorrector(residual=ClearanceGuidanceResidual(n_inner=20, min_progress_ratio=0.0),
                              selector="nearest")
    corr.set_scene(sc, EGO)
    corr.decode = dec
    target = torch.zeros(1, T, 2); target[..., 0] = 0.5  # the "model" wants to drive straight ahead

    def step_fn(x, t):  # flow-matching field toward `target`: v = a - eps, eps = (x - t a)/(1 - t)
        tt = t.view(-1, 1, 1)
        eps = (x - tt * target) / (1 - tt).clamp_min(0.05)
        return target - eps

    x = corr.sample(2, step_fn, device="cpu", inference_step=10, x_dims=[T, 2])
    assert x.shape == (2, T, 2) and corr.base_xyz.shape == (2, T, 3)
    assert corr.info.selected == [0] and not corr.info.bypassed
    base_xy, _ = dec(torch.as_tensor(corr.base_xyz[..., :2]))
    corr_xy, _ = dec(x)
    assert (corr_xy - base_xy).abs().sum() > 0  # the latch fired and the plan changed
    # empty scene -> bypass, plain Euler
    corr.set_scene(None, EGO)
    x2 = corr.sample(1, step_fn, device="cpu", inference_step=10, x_dims=[T, 2])
    assert corr.info.bypassed and x2.shape == (1, T, 2)
