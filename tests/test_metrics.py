import numpy as np

from groundact_repro.geometry import box_corners, obb_intersects, yaw_from_rot
from groundact_repro.metrics import ActorSet, EgoDims, ade, collision, corner_distance, evaluate_candidates

EGO = EgoDims(4.872, 2.121, 1.327)


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.zeros(yaw.shape + (3, 3)); r[..., 0, 0] = c; r[..., 0, 1] = -s; r[..., 1, 0] = s; r[..., 1, 1] = c; r[..., 2, 2] = 1
    return r


def straight(v, T=64, dt=0.1, lateral=0.0):
    t = np.arange(1, T + 1) * dt
    xyz = np.stack([v * t, np.full(T, lateral), np.zeros(T)], -1)
    return xyz, rot_z(np.zeros(T))


def test_sat_basic():
    assert obb_intersects(np.array([0., 0.]), 0.0, 4, 2, np.array([3., 0.]), 0.0, 4, 2)
    assert not obb_intersects(np.array([0., 0.]), 0.0, 4, 2, np.array([5., 0.]), 0.0, 4, 2)
    # rotated box touching diagonally
    assert obb_intersects(np.array([0., 0.]), 0.0, 4, 2, np.array([2.5, 1.5]), np.pi / 4, 4, 2)
    assert not obb_intersects(np.array([0., 0.]), 0.0, 4, 2, np.array([0., 2.5]), 0.0, 4, 2)


def test_corners_shape():
    c = box_corners(np.zeros((5, 2)), np.zeros(5), 4.0, 2.0)
    assert c.shape == (5, 4, 2) and np.allclose(c[0, 0], [2, 1])


def test_ade_and_corner():
    gt, gtr = straight(10.0)
    pred = np.stack([gt, gt + np.array([0, 1, 0])])
    pr = np.stack([gtr, gtr])
    a = ade(pred[..., :2], gt[:, :2], 25)
    assert np.allclose(a, [0.0, 1.0])
    cd = corner_distance(pred[..., :2], yaw_from_rot(pr), gt[:, :2], yaw_from_rot(gtr), EGO)
    assert np.allclose(cd, [0.0, 1.0])


def test_collision_with_stopped_car_ahead():
    gt, gtr = straight(10.0)  # ego drives 64 m
    T = 64
    xy = np.zeros((1, T + 1, 2)); xy[0, :, 0] = 20.0  # car parked 20 m ahead, same lane
    actors = ActorSet(["a"], ["automobile"], np.array([[4.5, 1.9]]), xy, np.zeros((1, T + 1)),
                      np.ones((1, T + 1), bool), np.zeros((1, 4, 2)), np.zeros((1, 4)), np.ones((1, 4), bool))
    pred = np.stack([gt, gt])
    pred[1, :, 1] = 4.0  # second candidate shifted one lane left
    c25 = collision(pred[..., :2], np.zeros((2, T)), actors, EGO, 25)
    c50 = collision(pred[..., :2], np.zeros((2, T)), actors, EGO, 50)
    assert c25.tolist() == [True, False] and c50.tolist() == [True, False]
    m = evaluate_candidates(pred, np.stack([gtr, gtr]), gt, gtr, actors, EGO)
    assert m["collision_2.5"] == 0.5 and m["ade_6.4"] == 2.0


def test_invalid_actor_steps_ignored():
    gt, gtr = straight(10.0)
    T = 64
    xy = np.zeros((1, T + 1, 2)); xy[0, :, 0] = 20.0
    valid = np.zeros((1, T + 1), bool)
    actors = ActorSet(["a"], ["automobile"], np.array([[4.5, 1.9]]), xy, np.zeros((1, T + 1)), valid,
                      np.zeros((1, 4, 2)), np.zeros((1, 4)), np.ones((1, 4), bool))
    assert not collision(gt[None, :, :2], np.zeros((1, T)), actors, EGO, 64).any()
