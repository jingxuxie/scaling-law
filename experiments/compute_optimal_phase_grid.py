"""Run compute-optimal visible-spectrum phase-grid experiments.

This wrapper calls the deterministic compute-optimal proxy from
`compute_optimal_scaling.py` across parameter regimes that illustrate different
phases of the theory:

1. current/default: a=1.5, b=1.4, where the compute-risk exponent saturates once
   theta >= theta_c ~= 0.133;
2. hard_full_range: a=3.0, b=1.4, where the whole theta range remains
   time-limited and the compute-risk exponent improves monotonically with theta;
3. wd_cap: AdamW-style weight decay cap, showing saturation when the effective
   horizon exceeds delta^{-1}.

Run from the repository root:

    python experiments/compute_optimal_phase_grid.py

Outputs are written to experiments/results/compute_optimal_phase_grid/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results/compute_optimal_phase_grid"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-c", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=600)
    args = parser.parse_args()

    base = [
        sys.executable,
        "experiments/compute_optimal_scaling.py",
        "--theta-values",
        "0,0.25,0.5,0.75,1",
        "--c-min",
        "1e4",
        "--c-max",
        "1e10",
        "--num-c",
        str(args.num_c),
        "--grid-size",
        str(args.grid_size),
    ]

    runs = [
        ("current_saturating", ["--a", "1.5", "--b", "1.4", "--weight-decay", "0"]),
        ("hard_full_range", ["--a", "3.0", "--b", "1.4", "--weight-decay", "0"]),
        ("wd_cap_delta_1e_minus_3", ["--a", "3.0", "--b", "1.4", "--weight-decay", "1e-3"]),
        ("wd_cap_delta_1e_minus_4", ["--a", "3.0", "--b", "1.4", "--weight-decay", "1e-4"]),
    ]

    for name, extra in runs:
        out = args.outdir / name
        cmd = base + extra + ["--outdir", str(out)]
        run(cmd, args.dry_run)

    print(f"Completed phase-grid runs under {args.outdir}")


if __name__ == "__main__":
    main()
