"""Compute-optimal visible-spectrum scaling with C-dependent AdamW weight decay.

This experiment tests the theorem in notes/20_weight_decay_compute_scaling.md.
For a visible-spectrum exponent theta, alpha=a*(1-theta/2).  If weight decay
scales as

    delta(C) = C^{-s},

then the predicted compute-risk exponent is

    beta_wd(s) = min((b-1)/(max(alpha,b)+1), s*(b-1)/alpha).

The fixed-weight-decay case corresponds to s=0 and has asymptotic exponent zero.

Example:
    python experiments/compute_optimal_weight_decay_schedule.py --quick

Main:
    python experiments/compute_optimal_weight_decay_schedule.py \
      --a 3.0 --b 1.4 --theta 1.0 \
      --s-values 0,0.05,0.1,0.2,0.4,0.8 \
      --c-min 1e4 --c-max 1e10 --num-c 40 \
      --outdir experiments/results/compute_optimal_wd_schedule
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float = 0.15, hi: float = 0.85) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return float("nan")
    a = int(lo * len(x))
    b = max(a + 3, int(hi * len(x)))
    b = min(b, len(x))
    X = np.vstack([np.ones(b - a), np.log(x[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(y[a:b]), rcond=None)
    return float(coef[1])


def risk_proxy(M: float, N: float, K: float, b: float) -> float:
    return M ** (1.0 - b) + K ** (1.0 - b) + K / N


def optimize_for_C(C: float, alpha: float, b: float, grid_size: int, delta: float) -> dict[str, float]:
    M_grid = np.geomspace(2.0, max(4.0, C / 2.0), grid_size)
    K_wd_cap = float("inf") if delta <= 0 else delta ** (-1.0 / alpha)
    best = None
    for M in M_grid:
        N = C / M
        if N < 2:
            continue
        K_time = N ** (1.0 / alpha)
        K_var = N ** (1.0 / b)
        K_cap = min(M, K_time, K_wd_cap)
        K = max(1.0, min(K_var, K_cap))
        R = risk_proxy(M, N, K, b)
        cand = {
            "C": C,
            "delta": delta,
            "M_opt": M,
            "N_opt": N,
            "K_opt": K,
            "risk_opt": R,
            "K_wd_cap": K_wd_cap,
            "K_time_cap": K_time,
        }
        if best is None or R < best["risk_opt"]:
            best = cand
    assert best is not None
    return best


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(outdir: Path, curve_rows: list[dict[str, object]], summary_rows: list[dict[str, object]], no_plots: bool) -> None:
    if no_plots or plt is None:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    for s in sorted(set(float(r["s"]) for r in curve_rows)):
        rows = [r for r in curve_rows if float(r["s"]) == s]
        C = np.array([float(r["C"]) for r in rows])
        R = np.array([float(r["risk_opt"]) for r in rows])
        plt.loglog(C, R, marker="o", label=f"s={s:g}")
    plt.xlabel("compute C")
    plt.ylabel("optimized risk proxy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "risk_vs_compute_by_s.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    s_vals = np.array([float(r["s"]) for r in summary_rows])
    obs = -np.array([float(r["obs_risk_exp"]) for r in summary_rows])
    pred = np.array([float(r["pred_positive_risk_exp"]) for r in summary_rows])
    plt.plot(s_vals, obs, marker="o", label="observed")
    plt.plot(s_vals, pred, marker="x", label="predicted")
    plt.xlabel("weight-decay schedule exponent s in delta(C)=C^{-s}")
    plt.ylabel("positive compute-risk exponent")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "risk_exponent_vs_s.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="experiments/results/compute_optimal_wd_schedule")
    parser.add_argument("--a", type=float, default=3.0)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--s-values", type=str, default="0,0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--c-min", type=float, default=1e4)
    parser.add_argument("--c-max", type=float, default=1e10)
    parser.add_argument("--num-c", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=600)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.c_min = 1e3
        args.c_max = 1e8
        args.num_c = 25
        args.grid_size = 300
        args.s_values = "0,0.05,0.1,0.2,0.4"
        args.outdir = "experiments/results/compute_optimal_wd_schedule_quick"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    alpha = args.a * (1.0 - args.theta / 2.0)
    m = max(alpha, args.b)
    beta_no_wd = (args.b - 1.0) / (m + 1.0)
    s_critical = alpha / (m + 1.0)
    C_values = np.geomspace(args.c_min, args.c_max, args.num_c)

    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for s in parse_float_list(args.s_values):
        rows_s = []
        for C in C_values:
            delta = C ** (-s)
            row = optimize_for_C(C, alpha, args.b, args.grid_size, delta)
            row.update({
                "a": args.a,
                "b": args.b,
                "theta": args.theta,
                "alpha": alpha,
                "s": s,
                "s_critical": s_critical,
                "beta_no_wd": beta_no_wd,
            })
            curve_rows.append(row)
            rows_s.append(row)
        C_arr = np.array([float(r["C"]) for r in rows_s])
        risk_exp = fit_slope(C_arr, np.array([float(r["risk_opt"]) for r in rows_s]))
        K_exp = fit_slope(C_arr, np.array([float(r["K_opt"]) for r in rows_s]))
        pred_beta = min(beta_no_wd, s * (args.b - 1.0) / alpha)
        summary_rows.append({
            "a": args.a,
            "b": args.b,
            "theta": args.theta,
            "alpha": alpha,
            "s": s,
            "s_critical": s_critical,
            "obs_risk_exp": risk_exp,
            "obs_positive_risk_exp": -risk_exp,
            "pred_positive_risk_exp": pred_beta,
            "risk_abs_error": abs((-risk_exp) - pred_beta),
            "obs_K_exp": K_exp,
            "pred_K_exp_cap_active": s / alpha if s < s_critical else 1.0 / (m + 1.0),
            "beta_no_wd": beta_no_wd,
        })
        print(f"s={s:g} risk_exp={risk_exp:.4f} pred={-pred_beta:.4f} K_exp={K_exp:.4f}")

    write_csv(outdir / "wd_schedule_curves.csv", curve_rows)
    write_csv(outdir / "wd_schedule_summary.csv", summary_rows)
    maybe_plot(outdir, curve_rows, summary_rows, args.no_plots)
    print(f"Wrote results to {outdir}")


if __name__ == "__main__":
    main()
