"""Learning-rate calibrated stochastic alignment sweep.

The previous stochastic alignment experiment showed the predicted q_eff diagnostics,
but risk slopes can be confounded by optimizer-specific learning-rate constants.
This runner repeats ``stochastic_alignment_scaling.py`` over learning-rate grids and
selects the best learning rate for each (case, optimizer) pair.

Quick smoke test
----------------
python experiments/lr_sweep_stochastic_alignment.py --quick

Main run
--------
python experiments/lr_sweep_stochastic_alignment.py \
  --cases aligned,band,flat,haar \
  --optimizers sgd,rmsprop,adam,adamw,coord_oracle,spectral_oracle \
  --lr-sgd-values 0.01,0.03,0.05,0.1 \
  --lr-adaptive-values 0.001,0.003,0.005,0.01,0.02 \
  --dimension 512 \
  --checkpoints 250,500,1000,2000,4000,8000 \
  --trials 5 \
  --batch-size 16 \
  --outdir experiments/results/stochastic_alignment_lr_sweep

Outputs
-------
- lr_sweep_all.csv: one row per (case, optimizer, lr)
- best_by_final_risk.csv: best learning rate per (case, optimizer)
- subruns/<case>__<optimizer>__lr_<lr>/summary.csv from the underlying script

Interpretation
--------------
Use this to distinguish spectral-exponent effects from scalar learning-rate
normalization.  The theory predicts q_eff≈1/2 for aligned/band adaptive methods
and q_eff≈0 for flat/Haar adaptive methods, regardless of the learning rate that
minimizes final risk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_list(text: str, typ=str):
    return [typ(x.strip()) for x in text.split(",") if x.strip()]


def sanitize_float(x: float) -> str:
    return (f"{x:.6g}").replace("-", "m").replace(".", "p")


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def final_summary_row(summary_rows: list[dict[str, str]]) -> dict[str, str]:
    if not summary_rows:
        raise ValueError("empty summary")
    return max(summary_rows, key=lambda r: int(float(r["step"])))


def build_command(
    args: argparse.Namespace,
    case: str,
    optimizer: str,
    lr: float,
    run_outdir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/stochastic_alignment_scaling.py",
        "--dimension",
        str(args.dimension),
        "--a",
        str(args.a),
        "--b",
        str(args.b),
        "--sigma",
        str(args.sigma),
        "--cases",
        case,
        "--optimizers",
        optimizer,
        "--checkpoints",
        args.checkpoints,
        "--trials",
        str(args.trials),
        "--batch-size",
        str(args.batch_size),
        "--rho",
        str(args.rho),
        "--epsilon",
        str(args.epsilon),
        "--beta1",
        str(args.beta1),
        "--beta2",
        str(args.beta2),
        "--weight-decay",
        str(args.weight_decay),
        "--outdir",
        str(run_outdir),
    ]
    if args.no_plots:
        cmd.append("--no-plots")
    if optimizer == "sgd":
        cmd += ["--lr-sgd", str(lr)]
    else:
        cmd += ["--lr-adaptive", str(lr)]
        cmd += ["--lr-sgd", str(args.lr_sgd_baseline)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results/stochastic_alignment_lr_sweep"))
    parser.add_argument("--cases", type=str, default="aligned,band,flat,haar")
    parser.add_argument("--optimizers", type=str, default="sgd,rmsprop,adam,adamw,coord_oracle,spectral_oracle")
    parser.add_argument("--lr-sgd-values", type=str, default="0.01,0.03,0.05,0.1")
    parser.add_argument("--lr-adaptive-values", type=str, default="0.001,0.003,0.005,0.01,0.02")
    parser.add_argument("--lr-sgd-baseline", type=float, default=0.05)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--checkpoints", type=str, default="250,500,1000,2000,4000,8000")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rho", type=float, default=1e-8)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Skip completed subruns with summary.csv.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.outdir = Path("experiments/results/stochastic_alignment_lr_quick")
        args.cases = "aligned,flat"
        args.optimizers = "sgd,rmsprop,adam"
        args.lr_sgd_values = "0.03,0.05"
        args.lr_adaptive_values = "0.003,0.005"
        args.dimension = 256
        args.checkpoints = "200,400,800"
        args.trials = 2
        args.batch_size = 8
        args.no_plots = True

    cases = parse_list(args.cases, str)
    optimizers = parse_list(args.optimizers, str)
    lr_sgd_values = parse_list(args.lr_sgd_values, float)
    lr_adaptive_values = parse_list(args.lr_adaptive_values, float)

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    all_rows: list[dict[str, Any]] = []
    subruns_dir = args.outdir / "subruns"

    for case in cases:
        for optimizer in optimizers:
            lr_values = lr_sgd_values if optimizer == "sgd" else lr_adaptive_values
            for lr in lr_values:
                run_outdir = subruns_dir / f"{case}__{optimizer}__lr_{sanitize_float(lr)}"
                summary_path = run_outdir / "summary.csv"
                cmd = build_command(args, case, optimizer, lr, run_outdir)
                print("RUN", " ".join(cmd))
                if not args.dry_run:
                    if args.resume and summary_path.exists():
                        print(f"  using existing {summary_path}")
                    else:
                        subprocess.run(cmd, check=True)
                    summary_rows = read_summary(summary_path)
                    final = final_summary_row(summary_rows)
                    all_rows.append(
                        {
                            "case": case,
                            "optimizer": optimizer,
                            "lr": lr,
                            "final_step": int(float(final["step"])),
                            "final_risk_mean": float(final["risk_mean"]),
                            "final_risk_std": float(final["risk_std"]),
                            "risk_slope_loglog": float(final["risk_slope_loglog"]),
                            "final_q_eff_mean": float(final["q_eff_mean"]),
                            "run_outdir": str(run_outdir),
                        }
                    )

    if args.dry_run:
        print("dry-run complete")
        return

    write_csv(args.outdir / "lr_sweep_all.csv", all_rows)

    best_rows: list[dict[str, Any]] = []
    for case in cases:
        for optimizer in optimizers:
            rows = [r for r in all_rows if r["case"] == case and r["optimizer"] == optimizer]
            if not rows:
                continue
            best = min(rows, key=lambda r: float(r["final_risk_mean"]))
            best_rows.append(best)
    write_csv(args.outdir / "best_by_final_risk.csv", best_rows)

    print(f"Wrote {args.outdir / 'lr_sweep_all.csv'}")
    print(f"Wrote {args.outdir / 'best_by_final_risk.csv'}")


if __name__ == "__main__":
    main()
