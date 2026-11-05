# GroundAct reproduction (inference side) on Alpamayo-R1 / Alpamayo 1.5

Reproduction of the inference and evaluation logic of
*Grounding What Shapes the Plan: Rethinking Groundedness for Physical Intelligence in Autonomous
Driving* (GroundAct, arXiv:2610.07521) using NVIDIA's released Alpamayo checkpoints.

## What is (and is not) reproducible

| Piece of the paper | Status | Where |
|---|---|---|
| Open-loop protocol of §4.1 / App. D, G (4 front-wide frames, 6 candidates, 6.4 s, ADE 2.5/5.0/6.4, corner distance, collision rate) | reproduced | `groundact_repro/data.py`, `metrics.py`, `scripts/eval_openloop.py` |
| Alpamayo-R1 and Alpamayo 1.5 rows of Table 1, **continuous** mode (CoC → flow-matching expert) | reproduced from the released checkpoints | `backends.py` |
| Alpamayo-R1 and Alpamayo 1.5 rows of Table 1, **discrete** mode (CoC → 128 action tokens from the VLM head) | reproduced; the released inference code never exposes this path, so it is re-implemented as a constrained-decoding grammar + the checkpoint's own `DiscreteTrajectoryTokenizer` | `discrete.py` |
| PAI-AV Normal (first 4,000) and OOD (290) benchmarks | reproduced (see assumptions) | `data.py` |
| PAI-AV* safety-critical set (512 samples) | **not public**; NVIDIA's public `challenging_scenarios` set (1,434 samples) is offered as a stand-in | `--benchmark challenging` |
| Off-road rate | **not computable** (public PAI-AV has no map) | – |
| GroundAct model itself (Qwen3-VL-2B + entity queries + pointer + expert + residual) | **no checkpoint or code released**; the inference logic of §3.2–3.4 is implemented as reference modules, exact where it is training-free, untrained `nn.Module`s where it is learned | `groundact_repro/groundact/` |
| GroundAct-style selective correction *on top of* an Alpamayo expert | runnable end-to-end with a training-free residual that mirrors the paper's teacher construction (App. C) and GT actor tracks as the entity pool | `--correction nearest|highest_risk|all` |
| Closed-loop AlpaSim, transfer-gap audit, detection/referencing metrics | out of scope | – |

The learned GroundAct pieces need training data that is not released (entity-level 3D supervision
aligned with reasoning references, teacher records), so numbers for the "GroundAct" rows of Table 1
cannot be reproduced. The Alpamayo rows can.

## Layout

```
groundact_repro/
  data.py        sample lists (normal / ood / challenging / custom), sample loading (front-wide only
                 or all 4 cameras), actor tracks from labels/obstacle.offline in the t0 ego frame
  metrics.py     ADE, corner distance, collision (SAT on oriented footprints), summaries
  discrete.py    grammar-constrained decoding of the discrete action tokens
  backends.py    one wrapper for both checkpoints and both generation modes
  geometry.py    oriented-box helpers
  groundact/
    entities.py    §3.2 entity queries, M_i readout, class/box heads
    pointer.py     §3.3 Eq. 1 gated pointer, Eq. 2 reference resolution/consumption, losses Eq. 7-8
    selection.py   R(r) from reasoning text, and the Table 4(a) heuristics
    forecast.py    constant-velocity / oracle / learned (untrained) forecasters
    interaction.py Interact(P, S_i): clearance, relative motion, validity, exact predicted overlap
    correction.py  risk latch, flow-time schedule, SetEnc + R_phi, teacher-style training-free
                   residual, corrected Euler sampler, context manager that patches Alpamayo
scripts/
  setup_h100.sh    one venv with alpamayo1_5 + alpamayo_r1 + this package
  smoke_test.py    one clip, both modes
  eval_openloop.py full benchmark runs, resumable, JSONL + summary
  summarize.py     markdown table next to the paper's numbers
tests/             CPU unit tests (19)
```

## Running on the H100 node

```bash
# once
source env.sh                      # ALPAMAYO_R1, ALPAMAYO_15, GROUNDACT_WORK, HF_HOME
bash scripts/setup_h100.sh         # clones both Alpamayo repos, builds the venv, caches configs
hf auth login                      # if not already: dataset + both checkpoints are gated

# every session
source env.sh && source $GROUNDACT_WORK/venv/bin/activate

python scripts/smoke_test.py --model $ALPAMAYO_15                   # one clip, both modes
python scripts/smoke_test.py --model $ALPAMAYO_R1 --correction       # + selective correction

# Table 1 rows (add --limit N while iterating; runs resume)
for M in $ALPAMAYO_R1 $ALPAMAYO_15; do
  for MODE in continuous discrete; do
    python scripts/eval_openloop.py --model $M --benchmark normal --mode $MODE
    python scripts/eval_openloop.py --model $M --benchmark ood    --mode $MODE
  done
done
python scripts/summarize.py results
```

Use `--attn sdpa` if flash-attn did not build. `--cameras all` runs the Alpamayo-native
4-camera input instead of the paper's front-wide-only protocol. GPU memory: ~40 GB for six
candidates (the six candidates are six independent CoC rollouts, as in Alpamayo's evaluation code).

Per-sample output rows (`samples.jsonl`) keep the six CoC strings and the six predicted
trajectories, so reasoning–action analyses (Table 5 style) can be run offline later.

## Protocol assumptions (things the paper leaves implicit)

* **Normal**: "first 4,000 samples of the PAI-AV default split" is read as the first 4,000 clips of
  the `test` partition of `splits/default.parquet` in file order, at t0 = 5.1 s (the Alpamayo
  convention). Change with `--benchmark custom --custom-path`.
* **OOD**: the 290 clips of `reasoning/ood_reasoning.parquet` with `split == "val"`; t0 is the
  first annotated event start that leaves 1.6 s of history and 6.4 s of future, else 5.1 s.
* **Collision**: ego footprint = oriented box from `vehicle_dimensions` centred
  `rear_axle_to_bbox_center` ahead of the rig origin; actor footprints from `obstacle.offline`
  auto-labels (all classes), moved into the t0 ego frame through the egomotion poses and
  linearly resampled at 10 Hz; a candidate collides if any of its 64 poses overlaps a valid actor
  box within the horizon. Samples whose clip has no obstacle labels are excluded from the collision
  average (the paper also reports collision on "annotation-available subsets").
* **Discrete mode**: CoC is sampled with the same top-p / temperature as the continuous mode, then
  the grammar forces `<|traj_future_start|>` → 128 tokens in `<i0>..<i2999>` → `<|traj_future_end|>`.
  Rows that did not emit 128 in-range tokens are still decoded (zero-filled) and flagged in
  `n_traj_tokens`.
* **Selective correction stand-in**: entity pool = GT actor tracks (the paper's entity states come
  from the model's own detector queries); forecasts = constant velocity from the four observed
  frames (`--forecaster oracle` for the GT upper bound); selection = `nearest` / `highest_risk` /
  `all` (Table 4(a) heuristics; `references` needs a trained pointer); residual = teacher-style
  clearance guidance (margin 1 m, clearance gain ≥ 0.5 m, progress ratio ≥ 0.95, no newly
  overlapping entity, u* = (a+ − x_s)/max(1−s, δ), γ = latch · min(1, 2 s²)). The sampler runs
  two passes from the same noise (base, then corrected) so `base_metrics` in each row gives the
  matched "preserved / safe→unsafe" analysis of Table 4(b).

## Local tests

```bash
uv venv .venv-test --python 3.12 && source .venv-test/bin/activate
uv pip install torch "numpy<2" "pandas<2.3" "scipy<1.15" "pyarrow<19" transformers==4.57.1 einops huggingface-hub pytest
python -m pytest tests -q
```
