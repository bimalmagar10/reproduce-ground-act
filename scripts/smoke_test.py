#!/usr/bin/env python3
"""One-clip end-to-end check: loads a model, runs both generation modes, prints metrics.

  python scripts/smoke_test.py --model /mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-1.5-10B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundact_repro import data as D  # noqa: E402
from groundact_repro.backends import AlpamayoBackend  # noqa: E402
from groundact_repro.metrics import evaluate_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clip", default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--cameras", default="front_wide")
    ap.add_argument("--num-candidates", type=int, default=2)
    ap.add_argument("--correction", action="store_true")
    args = ap.parse_args()

    backend = AlpamayoBackend(args.model, attn=args.attn)
    avdi = D.make_avdi()
    s = D.EvalSample(args.clip, args.t0_us)
    sample = D.load_sample(avdi, s, cameras=args.cameras)
    ego = D.load_ego_dims(avdi, s.clip_id)
    actors = D.load_actors(avdi, s.clip_id, s.t0_us)
    print("ego dims:", ego, "| actors:", None if actors is None else actors.n)
    inputs = backend.build_inputs(sample)
    gt_xyz, gt_rot = sample["ego_future_xyz"][0, 0].numpy(), sample["ego_future_rot"][0, 0].numpy()
    for mode in ("continuous", "discrete"):
        corrector = None
        if args.correction and mode == "continuous":
            from groundact_repro.groundact.correction import GroundActCorrector
            from groundact_repro.groundact.forecast import SceneEntities
            corrector = GroundActCorrector(selector="nearest")
            corrector.set_scene(SceneEntities.from_actors(actors, backend.device) if actors else None, ego)
        res = backend.generate(inputs, mode=mode, num_candidates=args.num_candidates, corrector=corrector)
        m = evaluate_candidates(res.pred_xyz, res.pred_rot, gt_xyz, gt_rot, actors, ego)
        print(f"== {mode}: CoC[0] = {res.cot[0]!r}")
        if res.n_traj_tokens:
            print("   valid traj tokens per candidate:", res.n_traj_tokens)
        if res.extra:
            print("   correction:", res.extra)
        print("   metrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == "__main__":
    main()
