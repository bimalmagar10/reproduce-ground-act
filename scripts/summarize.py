#!/usr/bin/env python3
"""Aggregate results/*/summary.json into a Table-1-style markdown table (and the paper's numbers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PAPER = {  # Table 1: (ADE2.5, ADE5.0, ADE6.4, Coll2.5, Coll5.0, Corner6.4) per benchmark
    ("Alpamayo-R1", "D", "normal"): (0.551, 2.612, 4.417, 4.269, 13.443, 4.599),
    ("Alpamayo-R1", "C", "normal"): (0.535, 2.263, 3.802, 3.714, 12.472, 3.945),
    ("Alpamayo-1.5", "D", "normal"): (0.425, 1.881, 3.163, 3.311, 10.951, 3.279),
    ("Alpamayo-1.5", "C", "normal"): (0.575, 2.247, 3.647, 3.952, 12.806, 3.770),
    ("Alpamayo-R1", "D", "ood"): (0.596, 2.691, 4.402, 5.805, 24.844, 4.568),
    ("Alpamayo-R1", "C", "ood"): (0.541, 2.295, 3.751, 4.182, 21.036, 3.872),
    ("Alpamayo-1.5", "D", "ood"): (0.522, 2.148, 3.412, 3.995, 21.223, 3.513),
    ("Alpamayo-1.5", "C", "ood"): (0.585, 2.240, 3.528, 4.370, 21.848, 3.627),
}


def main(root="results"):
    rows = []
    for p in sorted(Path(root).glob("*/summary.json")):
        s = json.loads(p.read_text())
        name = "Alpamayo-R1" if s["family"] == "r1" else "Alpamayo-1.5"
        mode = "D" if s["mode"] == "discrete" else "C"
        rows.append((p.parent.name, name, mode, s))
    print("| run | model | mode | bench | n | ADE2.5 | ADE5.0 | ADE6.4 | Coll2.5% | Coll5.0% | Corner6.4 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda s, k: f"{s[k]:.3f}" if k in s else "-"  # noqa: E731
    for run, name, mode, s in rows:
        print(f"| {run} | {name} | {mode} | {s['benchmark']} | {s['n_samples']} | {f(s,'ade_2.5')} | "
              f"{f(s,'ade_5.0')} | {f(s,'ade_6.4')} | {f(s,'collision_2.5_pct')} | {f(s,'collision_5.0_pct')} | "
              f"{f(s,'corner_6.4')} |")
        paper = PAPER.get((name, mode, s["benchmark"]))
        if paper:
            print(f"| paper | {name} | {mode} | {s['benchmark']} | - | " + " | ".join(f"{v:.3f}" for v in paper) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
