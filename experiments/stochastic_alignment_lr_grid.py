"""Learning-rate grid for stochastic coordinate-alignment experiments.

The first stochastic-alignment run validates the q_eff diagnostic, but raw risk
slopes are sensitive to optimizer learning-rate choices.  This script repeats the
same experiment over a learning-rate grid and reports, for each case/optimizer,
the best final-risk setting and its fitted risk slope.

Recommended quick run:

    python experiments/stochastic_alignment_lr_grid.py --quick

Recommended main run:

    python experiments/stochastic_alignment_lr_grid.py \
      --cases aligned,band,flat,haar \
      --optimizers sgd,rmsprop,adam,adamw,coord_oracle,spectral_oracle \
      --lr-sgd-values 0.01,0.03,0.05,0.1 \
      --lr-adaptive-values 0.001,0.003,0.005,0.01 \
      --dimension 512 \
      --checkpoints 250,500,1000,2000,4000,8000 \
      --trials 5 \
      --outdir experiments/results/stochastic_alignment_lr_grid

Outputs:

    grid_curves.csv      all checkpoint risks for every learning rate
    grid_summary.csv     mean final risk and slope for every grid point
    best_by_final.csv    best learning rate for each case/optimizer
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# When this file is run as `python experiments/stochastic_alignment_lr_grid.py`,
# the experiments directory is on sys.path, so this import works.
import stochastic_alignment_scaling as base

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def parse_list(text: str, typ=str):
    return [typ(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_grid(curves: list[dict[str, object]], checkpoints: list[int]) -> list[dict[str, object]]:
    keys = sorted(
        set(
            (
                str(r["case"]),
                str(r["optimizer"]),
                float(r["lr"]),
            )
            for r in curves
        )
    )
    rows: list[dict[str, object]] = []
    final_step = max(checkpoints)
    for case, opt, lr in keys:
        subset = [
            r
            for r in curves
            if str(r["case"]) == case and str(r["optimizer"]) == opt and float(r["lr"]) == lr
        ]
        mean_risks = []
        q_means = []
        for step in checkpoints:
            vals = np.array([float(r["excess_risk"]) for r in subset if int(r["step"]) == step])
            qs = np.array([float(r["q_eff"]) for r in subset if int(r["step"]) == step])
            if len(vals) == 0:
                continue
            mean_risks.append((step, float(vals.mean())))
            q_means.append(float(np.nanmean(qs)))
        if len(mean_risks) < 3:
            slope = float("nan")
        else:
            steps = np.array([x[0] for x in mean_risks], dtype=float)
            risks = np.array([x[1] for x in mean_risks], dtype=float)
            slope = base.fit_slope(steps, risks)
        final_vals = np.array([float(r["excess_risk"]) for r in subset if int(r["step"]) == final_step])
        rows.append(
            {
                "case": case,
                "optimizer": opt,
                "lr": lr,
                "final_step": final_step,
                "final_risk_mean": float(final_vals.mean()) if len(final_vals) else float("nan"),
                "final_risk_std": float(final_vals.std(ddof=1)) if len(final_vals) > 1 else 0.0,
                "risk_slope_loglog": slope,
                "q_eff_mean": float(np.nanmean(q_means)) if q_means else float("nan"),
            }
        )
    return rows


def best_by_final(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted(set((str(r["case"]), str(r["optimizer"])) for r in summary))
    best: list[dict[str, object]] = []
    for case, opt in groups:
        candidates = [r for r in summary if str(r["case"]) == case and str(r["optimizer"]) == opt]
        candidates = [r for r in candidates if np.isfinite(float(r["final_risk_mean"]))]
        if not candidates:
            continue
        winner = min(candidates, key=lambda r: float(r["final_risk_mean"]))
        best.append(dict(winner))
    return best


def maybe_plot(outdir: Path, best: list[dict[str, object]], no_plots: bool) -> None:
    if no_plots or plt is None or not best:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(set(str(r["case"]) for r in best))
    optimizers = sorted(set(str(r["optimizer"]) for r in best))
    for metric in ["final_risk_mean", "risk_slope_loglog", "q_eff_mean"]:
        fig = plt.figure(figsize=(max(8, len(cases) * 1.5), 4))
        width = 0.8 / max(1, len(optimizers))
        x = np.arange(len(cases), dtype=float)
        for j, opt in enumerate(optimizers):
            vals = []
            for case in cases:
                rows = [r for r in best if str(r["case"]) == case and str(r["optimizer"]) == opt]
                vals.append(float(rows[0][metric]) if rows else np.nan)
            plt.bar(x + (j - len(optimizers) / 2) * width, vals, width=width, label=opt)
        plt.xticks(x, cases)
        plt.ylabel(metric)
        plt.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(plot_dir / f"best_{metric}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="experiments/results/stochastic_alignment_lr_grid")
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--band-kappa", type=float, default=2.0)
    parser.add_argument("--cases", type=str, default="aligned,band,flat,haar")
    parser.add_argument("--optimizers", type=str, default="sgd,rmsprop,adam,adamw,coord_oracle,spectral_oracle")
    parser.add_argument("--checkpoints", type=str, default="250,500,1000,2000,4000,8000")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-sgd-values", type=str, default="0.01,0.03,0.05,0.1")
    parser.add_argument("--lr-adaptive-values", type=str, default="0.001,0.003,0.005,0.01")
    parser.add_argument("--rho", type=float, default=1e-8)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.dimension = 256
        args.cases = "aligned,flat,haar"
        args.optimizers = "sgd,rmsprop,adam,coord_oracle,spectral_oracle"
        args.checkpoints = "200,400,800"
        args.trials = 2
        args.batch_size = 8
        args.lr_sgd_values = "0.03,0.05"
        args.lr_adaptive_values = "0.003,0.005"
        args.outdir = "experiments/results/stochastic_alignment_lr_grid_quick"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    cases = parse_list(args.cases, str)
    optimizers = parse_list(args.optimizers, str)
    checkpoints = sorted(set(parse_list(args.checkpoints, int)))
    lr_sgd_values = parse_list(args.lr_sgd_values, float)
    lr_adaptive_values = parse_list(args.lr_adaptive_values, float)

    curves: list[dict[str, object]] = []
    base_rng = np.random.default_rng(args.seed)
    for case in cases:
        for opt in optimizers:
            lr_values = lr_sgd_values if opt == "sgd" else lr_adaptive_values
            for lr in lr_values:
                for trial in range(args.trials):
                    trial_args = SimpleNamespace(**vars(args))
                    trial_args.lr_sgd = lr if opt == "sgd" else lr_sgd_values[0]
                    trial_args.lr_adaptive = lr if opt != "sgd" else lr_adaptive_values[0]
                    trial_seed = int(base_rng.integers(0, 2**31 - 1))
                    rows = base.train_one(case, opt, trial_args, trial_seed)
                    for r in rows:
                        r["trial"] = trial
                        r["lr"] = lr
                    curves.extend(rows)
                print(f"case={case:8s} opt={opt:15s} lr={lr:g} done")

    write_csv(outdir / "grid_curves.csv", curves)
    summary = summarize_grid(curves, checkpoints)
    write_csv(outdir / "grid_summary.csv", summary)
    best = best_by_final(summary)
    write_csv(outdir / "best_by_final.csv", best)
    maybe_plot(outdir, best, args.no_plots)

    print(f"Wrote grid results to {outdir}")
    print(f"  {outdir / 'grid_curves.csv'}")
    print(f"  {outdir / 'grid_summary.csv'}")
    print(f"  {outdir / 'best_by_final.csv'}")


if __name__ == "__main__":
    main()
