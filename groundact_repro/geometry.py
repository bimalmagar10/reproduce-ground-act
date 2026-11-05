"""BEV box geometry shared by the metrics and the GroundAct interaction module (numpy)."""
from __future__ import annotations

import numpy as np

_LOCAL = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)  # (4, 2) corner signs


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    """Heading angle from (..., 3, 3) rotation matrices (yaw about +z)."""
    return np.arctan2(rot[..., 1, 0], rot[..., 0, 0])


def yaw_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Yaw from (..., 4) quaternions in (x, y, z, w) order."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def box_corners(center: np.ndarray, yaw: np.ndarray, length, width) -> np.ndarray:
    """Corners of oriented BEV boxes.

    center: (..., 2); yaw: (...); length, width: scalars or (...).
    Returns (..., 4, 2) in the order front-left, front-right, rear-right, rear-left.
    """
    center = np.asarray(center, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    half = np.stack([np.broadcast_to(length, yaw.shape) / 2.0,
                     np.broadcast_to(width, yaw.shape) / 2.0], axis=-1)  # (..., 2)
    local = _LOCAL * half[..., None, :]  # (..., 4, 2)
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    x = center[..., None, 0] + local[..., 0] * c - local[..., 1] * s
    y = center[..., None, 1] + local[..., 0] * s + local[..., 1] * c
    return np.stack([x, y], axis=-1)


def obb_intersects(ca, ya, la, wa, cb, yb, lb, wb) -> np.ndarray:
    """Separating-axis test between two sets of oriented boxes, broadcast over leading dims.

    Returns a boolean array of shape broadcast(ca[..., 0], cb[..., 0]).
    """
    ya = np.asarray(ya, dtype=np.float64)
    yb = np.asarray(yb, dtype=np.float64)
    A = box_corners(ca, ya, la, wa)  # (..., 4, 2)
    B = box_corners(cb, yb, lb, wb)
    A, B = np.broadcast_arrays(A, B)
    shape = A.shape[:-2]
    ya_b = np.broadcast_to(ya, shape)
    yb_b = np.broadcast_to(yb, shape)
    axes = np.stack(
        [
            np.stack([np.cos(ya_b), np.sin(ya_b)], -1),
            np.stack([-np.sin(ya_b), np.cos(ya_b)], -1),
            np.stack([np.cos(yb_b), np.sin(yb_b)], -1),
            np.stack([-np.sin(yb_b), np.cos(yb_b)], -1),
        ],
        axis=-2,
    )  # (..., 4 axes, 2)
    pa = np.einsum("...cd,...ad->...ac", A, axes)  # (..., 4 axes, 4 corners)
    pb = np.einsum("...cd,...ad->...ac", B, axes)
    separated = (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))  # (..., 4 axes)
    return ~separated.any(-1)
