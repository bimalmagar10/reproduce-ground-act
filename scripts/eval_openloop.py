#!/usr/bin/env python3
"""Open-loop evaluation of Alpamayo-R1 / Alpamayo 1.5 under the GroundAct protocol (Table 1).

Examples (on the H100 node, after `source env.sh` and activating the venv):

  # Alpamayo 1.5, continuous mode, PAI-AV Normal, first 50 samples
  python scripts/eval_openloop.py --model $ALPAMAYO_15 --benchmark normal --mode continuous --limit 50

  # Alpamayo-R1, discrete action tokens, OOD split
  python scripts/eval_openloop.py --model $ALPAMAYO_R1 --benchmark ood --mode discrete

  # Alpamayo 1.5 + GroundAct-style selective correction (training-free residual, nearest entity)
  python scripts/eval_openloop.py --model $ALPAMAYO_15 --benchmark challenging --mode continuous \
      --correction nearest

Outputs: <out>/samples.jsonl (one row per sample: CoC, predictions, metrics) and <out>/summary.json.
Re-running with the same --out resumes (finished samples are skipped).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groundact_repro import data as D  # noqa: E402
from groundact_repro.backends import AlpamayoBackend  # noqa: E402
from groundact_repro.metrics import evaluate_candidates, summarize  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local checkpoint dir (Alpamayo-R1-10B or Alpamayo-1.5-10B)")
    ap.add_argument("--family", choices=["r1", "1.5"], default=None, help="auto-detected from config.json")
    ap.add_argument("--benchmark", choices=["normal", "ood", "challenging", "custom"], default="normal")
    ap.add_argument("--custom-path", default=None)
    ap.add_argument("--mode", choices=["continuous", "discrete"], default="continuous")
    ap.add_argument("--cameras", choices=["front_wide", "all"], default="front_wide",
                    help="front_wide = the paper's protocol; all = Alpamayo-native 4 cameras")
    ap.add_argument("--num-candidates", type=int, default=6)
    ap.add_argument("--top-p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-cot-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--no-actors", action="store_true", help="skip actor labels / collision metric")
    ap.add_argument("--correction", default="off",
                    choices=["off", "nearest", "highest_risk", "all", "none"],
                    help="continuous mode only: GroundAct-style selective correction with this "
                         "entity selector ('none' runs the two-pass sampler with an empty set)")
    ap.add_argument("--forecaster", default="cv", choices=["cv", "oracle"])
    ap.add_argument("--select-k", type=int, default=1)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--dataset-cache", default=os.environ.get("HF_HOME"))
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("eval")
    out_dir = Path(args.out or f"results/{Path(args.model).name}_{args.benchmark}_{args.mode}_{args.cameras}"
                   + ("" if args.correction == "off" else f"_corr-{args.correction}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "samples.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])

    samples = D.build_sample_list(args.benchmark, limit=None, custom_path=args.custom_path)
    samples = samples[args.start:]
    if args.limit:
        samples = samples[: args.limit]
    log.info("benchmark=%s samples=%d (done=%d)", args.benchmark, len(samples), len(done))

    backend = AlpamayoBackend(args.model, family=args.family, attn=args.attn, cuda_graph=args.cuda_graph)
    log.info("loaded %s (family %s)", args.model, backend.family)
    avdi = D.make_avdi()

    corrector = None
    if args.correction != "off":
        if args.mode != "continuous":
            sys.exit("--correction requires --mode continuous")
        from groundact_repro.groundact.correction import ClearanceGuidanceResidual, GroundActCorrector
        from groundact_repro.groundact.forecast import ConstantVelocityForecaster, OracleForecaster
        corrector = GroundActCorrector(
            residual=ClearanceGuidanceResidual(margin=args.margin),
            forecaster=OracleForecaster() if args.forecaster == "oracle" else ConstantVelocityForecaster(),
            selector=args.correction, select_k=args.select_k)

    rows, failed, t_start = [], [], time.time()
    with jsonl.open("a") as fh:
        for i, s in enumerate(samples):
            if s.key in done:
                continue
            try:
                t0 = time.time()
                sample = D.load_sample(avdi, s, cameras=args.cameras)
                ego = D.load_ego_dims(avdi, s.clip_id)
                actors = None if args.no_actors else D.load_actors(avdi, s.clip_id, s.t0_us)
                inputs = backend.build_inputs(sample)
                if corrector is not None:
                    from groundact_repro.groundact.forecast import SceneEntities
                    scene = SceneEntities.from_actors(actors, device=backend.device) if actors is not None else None
                    corrector.set_scene(scene, ego)
                res = backend.generate(inputs, mode=args.mode, num_candidates=args.num_candidates,
                                       top_p=args.top_p, temperature=args.temperature,
                                       max_cot_tokens=args.max_cot_tokens, seed=args.seed, corrector=corrector)
                gt_xyz = sample["ego_future_xyz"][0, 0].numpy()
                gt_rot = sample["ego_future_rot"][0, 0].numpy()
                m = evaluate_candidates(res.pred_xyz, res.pred_rot, gt_xyz, gt_rot, actors, ego)
                row = {"key": s.key, "clip_id": s.clip_id, "t0_us": s.t0_us, "mode": res.mode,
                       "cot": res.cot, "metrics": m, "n_actors": 0 if actors is None else actors.n,
                       "pred_xy": np.round(res.pred_xyz[..., :2], 3).tolist(),
                       "gt_xy": np.round(gt_xyz[:, :2], 3).tolist(),
                       "seconds": round(time.time() - t0, 2)}
                if res.n_traj_tokens:
                    row["n_traj_tokens"] = res.n_traj_tokens
                if res.base_xyz is not None:
                    mb = evaluate_candidates(res.base_xyz, res.pred_rot, gt_xyz, gt_rot, actors, ego)
                    row["base_metrics"] = mb
                    row["correction"] = res.extra
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(m)
                log.info("[%d/%d] %s ade6.4=%.3f coll5.0=%s %.1fs", i + 1, len(samples), s.key[:12],
                         m["ade_6.4"], m.get("collision_5.0"), time.time() - t0)
            except Exception as e:  # noqa: BLE001
                failed.append({"key": s.key, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                log.exception("FAIL %s", s.key)
                torch.cuda.empty_cache()

    all_rows = [json.loads(l)["metrics"] for l in jsonl.read_text().splitlines() if l.strip()]
    summary = {"model": args.model, "family": backend.family, "benchmark": args.benchmark, "mode": args.mode,
               "cameras": args.cameras, "correction": args.correction, "num_candidates": args.num_candidates,
               "top_p": args.top_p, "temperature": args.temperature, "wall_seconds": time.time() - t_start,
               "failed": failed, **summarize(all_rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps({k: v for k, v in summary.items() if k != "failed"}, indent=1))


if __name__ == "__main__":
    main()
