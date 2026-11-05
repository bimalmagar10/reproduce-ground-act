import json

import numpy as np
import pandas as pd

from groundact_repro import data as D


def test_t0_fits():
    assert D._t0_fits(5_100_000) and not D._t0_fits(1_000_000) and not D._t0_fits(14_000_000)


def test_resample_track():
    obs_t = np.array([0, 100_000, 200_000, 2_000_000])
    vals = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [2.0, 0.0, 0.2], [20.0, 0.0, 2.0]])
    q = np.array([50_000, 150_000, 1_000_000, 2_100_000, 3_000_000])
    out, ok = D._resample_track(obs_t, vals, q)
    assert np.allclose(out[0], [0.5, 0, 0.05]) and np.allclose(out[1], [1.5, 0, 0.15])
    assert ok.tolist() == [True, True, False, True, False]  # 1.0 s falls in a 1.8 s gap


def test_custom_sample_list(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"clip_id": ["a", "b"], "t0_us": [5_100_000, 3_000_000]}).to_csv(p, index=False)
    s = D.build_sample_list("custom", custom_path=str(p))
    assert [x.key for x in s] == ["a@5100000", "b@3000000"]


def test_ood_t0_choice(monkeypatch, tmp_path):
    events = json.dumps([{"event_start_timestamp": 1_000_000, "coc": "x"},
                         {"event_start_timestamp": 7_000_000, "coc": "y"}])
    df = pd.DataFrame({"events": [events, "[]"], "split": ["val", "val"]}, index=["c1", "c2"])
    p = tmp_path / "ood.parquet"; df.to_parquet(p)
    monkeypatch.setattr(D, "dataset_file", lambda *a, **k: p)
    s = D.build_sample_list("ood")
    assert s[0].t0_us == 7_000_000 and s[1].t0_us == D.DEFAULT_T0_US
