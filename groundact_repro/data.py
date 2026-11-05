"""PAI-AV sample lists, sample loading and actor tracks for the GroundAct open-loop protocol.

Benchmarks (paper, Section 4.1 / Appendix D):
  normal       first 4,000 samples of the PAI-AV default split (we take the `test` partition of
               `splits/default.parquet` in file order, t0 = 5.1 s, the Alpamayo convention).
  ood          the 290-clip OOD split (`reasoning/ood_reasoning.parquet`, split == "val"); t0 is
               the first annotated event timestamp that leaves room for 1.6 s of history and 6.4 s
               of future, otherwise 5.1 s.
  challenging  NVIDIA's public `challenging_scenarios` open-loop set (1,434 (clip, t0) pairs). Not
               in the paper; offered as a public stand-in because PAI-AV* (512 hand-picked
               safety-critical samples) was not released.
  custom       a CSV/parquet with columns clip_id, t0_us.

Inputs follow the paper: four temporally ordered front-wide frames (t-0.3, t-0.2, t-0.1, t), the
16-step ego history, a 6.4 s / 64-pose future. Actor boxes come from `labels/obstacle.offline`
(auto-labels, rig frame at their own timestamp) and are moved into the ego frame at t0 with the
egomotion poses, then linearly resampled on the 10 Hz prediction grid.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
from huggingface_hub import hf_hub_download

from groundact_repro.metrics import ActorSet, EgoDims

DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DEFAULT_T0_US = 5_100_000
HISTORY_US = 1_600_000
FUTURE_US = 6_400_000
CLIP_LEN_US = 20_000_000
DT_US = 100_000


@dataclass(frozen=True)
class EvalSample:
    clip_id: str
    t0_us: int
    tag: str = ""

    @property
    def key(self) -> str:
        return f"{self.clip_id}@{self.t0_us}"


def dataset_file(rel_path: str, cache_dir: str | None = None) -> Path:
    return Path(hf_hub_download(DATASET_REPO, rel_path, repo_type="dataset", cache_dir=cache_dir))


def _t0_fits(t0_us: int) -> bool:
    return t0_us > HISTORY_US and t0_us + FUTURE_US <= CLIP_LEN_US


def build_sample_list(benchmark: str, limit: int | None = None, *, split_partition: str = "test",
                      ood_partition: str = "val", custom_path: str | None = None,
                      cache_dir: str | None = None) -> list[EvalSample]:
    if benchmark == "normal":
        df = pd.read_parquet(dataset_file("splits/default.parquet", cache_dir))
        clips = df.index[df["split"] == split_partition].tolist()
        samples = [EvalSample(c, DEFAULT_T0_US, "normal") for c in clips]
    elif benchmark == "ood":
        df = pd.read_parquet(dataset_file("reasoning/ood_reasoning.parquet", cache_dir))
        df = df[df["split"] == ood_partition]
        samples = []
        for clip_id, row in df.iterrows():
            t0 = DEFAULT_T0_US
            try:
                events = json.loads(row["events"]) if isinstance(row["events"], str) else []
            except json.JSONDecodeError:
                events = []
            for ev in events:
                cand = int(ev.get("event_start_timestamp", -1))
                if _t0_fits(cand):
                    t0 = cand
                    break
            samples.append(EvalSample(clip_id, t0, "ood"))
    elif benchmark == "challenging":
        df = pd.read_parquet(dataset_file("splits/challenging_scenarios.parquet", cache_dir))
        samples = [EvalSample(c, int(t0), "challenging")
                   for c, t0s in df["t0"].items() for t0 in t0s if _t0_fits(int(t0))]
    elif benchmark == "custom":
        if custom_path is None:
            raise ValueError("--custom-path is required for the custom benchmark")
        p = Path(custom_path)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        if "clip_id" not in df.columns:
            df = df.reset_index()
        samples = [EvalSample(str(r.clip_id), int(getattr(r, "t0_us", DEFAULT_T0_US)), "custom")
                   for r in df.itertuples()]
    else:
        raise ValueError(f"unknown benchmark {benchmark}")
    if limit:
        samples = samples[:limit]
    return samples


# ----------------------------------------------------------------------------- sample loading

def _loader():
    try:
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    except ImportError:
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    return load_physical_aiavdataset


def make_avdi(cache_dir: str | None = None, local_dir: str | None = None):
    import physical_ai_av
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if local_dir:
        kwargs["local_dir"] = local_dir
    return physical_ai_av.PhysicalAIAVDatasetInterface(**kwargs)


def camera_features(avdi, cameras: str) -> list | None:
    cam = avdi.features.CAMERA
    if cameras == "front_wide":  # the paper's protocol
        return [cam.CAMERA_FRONT_WIDE_120FOV]
    if cameras == "all":  # the Alpamayo-native 4-camera input
        return None
    raise ValueError(cameras)


def load_sample(avdi, sample: EvalSample, cameras: str = "front_wide") -> dict:
    """Model-ready sample dict (see the Alpamayo loader docstring for keys)."""
    return _loader()(sample.clip_id, t0_us=sample.t0_us, avdi=avdi,
                     camera_features=camera_features(avdi, cameras))


def load_ego_dims(avdi, clip_id: str) -> EgoDims:
    try:
        vd = avdi.get_clip_feature(clip_id, avdi.features.CALIBRATION.VEHICLE_DIMENSIONS,
                                   maybe_stream=True)
        return EgoDims(float(vd.length), float(vd.width), float(vd.rear_axle_to_bbox_center))
    except Exception:  # noqa: BLE001 - fall back to the fleet's dominant vehicle
        return EgoDims(4.872, 2.121, 1.327)


def _obstacle_df(avdi, clip_id: str) -> pd.DataFrame | None:
    feat = getattr(avdi.features.LABELS, "OBSTACLE_OFFLINE", None)
    if feat is None:
        return None
    try:
        obj = avdi.get_clip_feature(clip_id, feat, maybe_stream=True)
    except Exception:  # noqa: BLE001 - missing labels for ~3% of clips
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
            if isinstance(v, io.BytesIO):
                return pd.read_parquet(v)
        return None
    return obj if isinstance(obj, pd.DataFrame) else None


def _resample_track(obs_t: np.ndarray, vals: np.ndarray, query_t: np.ndarray,
                    max_gap_us: int = 600_000, edge_us: int = 200_000):
    """Linear resampling of a track; a query is valid if bracketed by observations closer than
    max_gap or within edge_us of the first/last observation."""
    order = np.argsort(obs_t)
    obs_t, vals = obs_t[order], vals[order]
    out = np.stack([np.interp(query_t, obs_t, vals[:, j]) for j in range(vals.shape[1])], -1)
    idx = np.searchsorted(obs_t, query_t)
    left = np.clip(idx - 1, 0, len(obs_t) - 1)
    right = np.clip(idx, 0, len(obs_t) - 1)
    bracketed = (idx > 0) & (idx < len(obs_t)) & ((obs_t[right] - obs_t[left]) <= max_gap_us)
    near_edge = (np.abs(query_t - obs_t[0]) <= edge_us) | (np.abs(query_t - obs_t[-1]) <= edge_us)
    return out, bracketed | near_edge


def load_actors(avdi, clip_id: str, t0_us: int, n_future: int = 64, n_hist: int = 4,
                egomotion=None, classes: set[str] | None = None) -> ActorSet | None:
    """Actor boxes in the ego frame at t0 on the 10 Hz grid; None when the clip has no labels."""
    df = _obstacle_df(avdi, clip_id)
    if df is None or len(df) == 0:
        return None
    if classes:
        df = df[df["label_class"].isin(classes)]
    lo = t0_us - (n_hist - 1) * DT_US - 1_000_000
    hi = t0_us + n_future * DT_US + 1_000_000
    df = df[(df["timestamp_us"] >= lo) & (df["timestamp_us"] <= hi)]
    if len(df) == 0:
        return ActorSet.empty(n_future, n_hist)
    if not (df["reference_frame"] == "rig").all():
        raise ValueError("unexpected obstacle reference frame")
    if egomotion is None:
        egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)

    ts = df["timestamp_us"].to_numpy().astype(np.int64)
    pose_obs = egomotion(ts).pose  # rig(t_obs) -> anchor, one transform per row
    pose_t0 = egomotion(np.array([t0_us], dtype=np.int64)).pose
    centers_rig = df[["center_x", "center_y", "center_z"]].to_numpy()
    centers_t0 = pose_t0.inv().apply(pose_obs.apply(centers_rig))
    rot_box = spt.Rotation.from_quat(
        df[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy())
    rot_t0 = pose_t0.rotation.inv() * pose_obs.rotation * rot_box
    yaw = rot_t0.as_euler("zyx")[:, 0]

    hist_t = t0_us + np.arange(-(n_hist - 1), 1) * DT_US
    fut_t = t0_us + np.arange(0, n_future + 1) * DT_US
    track_ids, cls, dims, xy, yaws, valid, hxy, hyaw, hvalid = [], [], [], [], [], [], [], [], []
    for tid, g_idx in df.groupby("track_id").indices.items():
        g = df.iloc[g_idx]
        obs_t = ts[g_idx]
        u_yaw = np.unwrap(yaw[g_idx][np.argsort(obs_t)])
        vals = np.column_stack([centers_t0[g_idx, :2][np.argsort(obs_t)], u_yaw])
        obs_sorted = np.sort(obs_t)
        f_vals, f_ok = _resample_track(obs_sorted, vals, fut_t)
        h_vals, h_ok = _resample_track(obs_sorted, vals, hist_t)
        if not (f_ok.any() or h_ok.any()):
            continue
        track_ids.append(str(tid))
        cls.append(str(g["label_class"].iloc[0]))
        dims.append([float(np.median(g["size_x"])), float(np.median(g["size_y"]))])
        xy.append(f_vals[:, :2]); yaws.append(f_vals[:, 2]); valid.append(f_ok)
        hxy.append(h_vals[:, :2]); hyaw.append(h_vals[:, 2]); hvalid.append(h_ok)
    if not track_ids:
        return ActorSet.empty(n_future, n_hist)
    return ActorSet(track_ids, cls, np.array(dims), np.stack(xy), np.stack(yaws), np.stack(valid),
                    np.stack(hxy), np.stack(hyaw), np.stack(hvalid))


def actors_from_local_zip(zip_path: str, clip_id: str) -> pd.DataFrame:
    """Debug helper: read one clip's obstacle table straight from a downloaded chunk zip."""
    with zipfile.ZipFile(zip_path) as z:
        return pd.read_parquet(io.BytesIO(z.read(f"{clip_id}.obstacle.offline.parquet")))
