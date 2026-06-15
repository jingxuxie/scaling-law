"""Compute-optimal visible-spectrum scaling experiment.

This deterministic experiment tests the compute-allocation theorem for the
visible-spectrum risk law.  It optimizes the simplified risk proxy

    R(M,N,K) = M^{1-b} + K^{1-b} + K/N

under compute C = M N and the algorithmic constraint

    K <= N^{1/alpha},   alpha = a * (1 - theta/2).

The variable K is the number of learned modes.  In the hard-source phase
b <= alpha, the time constraint is active.  In the easy-source phase b > alpha,
variance forces K ~ N^{1/b}, equivalently a smaller effective learning-rate
horizon n = K^alpha < N.

The theorem predicts, with m = max(alpha, b),

    M_* ~ C^{1/(m+1)}
    N_* ~ C^{m/(m+1)}
    K_* ~ C^{1/(m+1)}
    R_* ~ C^{-(b-1)/(m+1)}.

Examples
--------
Quick:
    python experiments/compute_optimal_scaling.py --quick

Main:
    python experiments/compute_optimal_scaling.py \
      --a 1.5 --b 1.4 \
      --theta-values 0,0.25,0.5,0.75,1 \
      --c-min 1e4 --c-max 1e10 --num-c 40 \
      --outdir experiments/results/compute_optimal_visible
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


def optimize_for_C(C: float, alpha: float, b: float, grid_size: int, wd_delta: float) -> dict[str, float]:
    # Search over M.  N=C/M.  For each N, the best unconstrained K for
    # K^{1-b}+K/N is K=N^{1/b}; constraints then cap it.
    m_min = 2.0
    m_max = max(4.0, C / 2.0)
    M_grid = np.geomspace(m_min, m_max, grid_size)
    best: dict[str, float] | None = None
    wd_k_cap = float("inf") if wd_delta <= 0 else wd_delta ** (-1.0 / alpha)
    for M in M_grid:
        N = C / M
        if N < 2:
            continue
        K_time = N ** (1.0 / alpha)
        K_var = N ** (1.0 / b)
        K_cap = min(M, K_time, wd_k_cap)
        if K_cap < 1:
            K = max(1.0, K_cap)
        else:
            K = min(K_var, K_cap)
            K = max(1.0, K)
        n_eff = K ** alpha
        gamma_eff = min(1.0, n_eff / N)
        R = risk_proxy(M, N, K, b)
        cand = {
            "C": C,
            "M_opt": M,
            "N_opt": N,
            "K_opt": K,
            "n_eff_opt": n_eff,
            "gamma_eff_opt": gamma_eff,
            "risk_opt": R,
            "K_time_cap": K_time,
            "K_wd_cap": wd_k_cap,
        }
        if best is None or R < best["risk_opt"]:
            best = cand
    assert best is not None
    return best


def predicted_exponents(alpha: float, b: float) -> dict[str, float | str]:
    m = max(alpha, b)
    phase = "hard_source_time_limited" if b <= alpha else "easy_source_variance_limited"
    return {
        "phase": phase,
        "pred_M_exp": 1.0 / (m + 1.0),
        "pred_N_exp": m / (m + 1.0),
        "pred_K_exp": 1.0 / (m + 1.0),
        "pred_risk_exp": -(b - 1.0) / (m + 1.0),
        "pred_n_eff_exp": alpha / (m + 1.0),
        "pred_gamma_eff_exp": 0.0 if b <= alpha else (alpha - b) / (b + 1.0),
    }


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
    thetas = sorted(set(float(r["theta"]) for r in curve_rows))
    fig = plt.figure()
    for theta in thetas:
        rows = [r for r in curve_rows if float(r["theta"]) == theta]
        C = np.array([float(r["C"]) for r in rows])
        R = np.array([float(r["risk_opt"]) for r in rows])
        plt.loglog(C, R, marker="o", label=f"theta={theta:g}")
    plt.xlabel("compute C = M N")
    plt.ylabel("optimized risk proxy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "risk_vs_compute.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    theta_vals = np.array([float(r["theta"]) for r in summary_rows])
    obs = np.array([float(r["obs_risk_exp"]) for r in summary_rows])
    pred = np.array([float(r["pred_risk_exp"]) for r in summary_rows])
    plt.plot(theta_vals, -obs, marker="o", label="observed positive exponent")
    plt.plot(theta_vals, -pred, marker="x", label="predicted positive exponent")
    plt.xlabel("visible profile exponent theta")
    plt.ylabel("compute-risk exponent")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "compute_exponent_vs_theta.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="experiments/results/compute_optimal_visible")
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--theta-values", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--c-min", type=float, default=1e4)
    parser.add_argument("--c-max", type=float, default=1e10)
    parser.add_argument("--num-c", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=600)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optional delta cap; 0 means no cap.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.c_min = 1e3
        args.c_max = 1e7
        args.num_c = 20
        args.grid_size = 250
        args.theta_values = "0,0.5,1"
        args.outdir = "experiments/results/compute_optimal_visible_quick"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    C_values = np.geomspace(args.c_min, args.c_max, args.num_c)
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for theta in parse_float_list(args.theta_values):
        alpha = args.a * (1.0 - theta / 2.0)
        preds = predicted_exponents(alpha, args.b)
        rows_theta = []
        for C in C_values:
            row = optimize_for_C(C, alpha, args.b, args.grid_size, args.weight_decay)
            row.update({
                "a": args.a,
                "b": args.b,
                "theta": theta,
                "q_eff": theta / 2.0,
                "alpha": alpha,
                "weight_decay": args.weight_decay,
                "phase": preds["phase"],
            })
            curve_rows.append(row)
            rows_theta.append(row)
        C_arr = np.array([float(r["C"]) for r in rows_theta])
        summary = {
            "a": args.a,
            "b": args.b,
            "theta": theta,
            "q_eff": theta / 2.0,
            "alpha": alpha,
            "phase": preds["phase"],
            "weight_decay": args.weight_decay,
            "obs_M_exp": fit_slope(C_arr, np.array([float(r["M_opt"]) for r in rows_theta])),
            "obs_N_exp": fit_slope(C_arr, np.array([float(r["N_opt"]) for r in rows_theta])),
            "obs_K_exp": fit_slope(C_arr, np.array([float(r["K_opt"]) for r in rows_theta])),
            "obs_n_eff_exp": fit_slope(C_arr, np.array([float(r["n_eff_opt"]) for r in rows_theta])),
            "obs_gamma_eff_exp": fit_slope(C_arr, np.array([float(r["gamma_eff_opt"]) for r in rows_theta])),
            "obs_risk_exp": fit_slope(C_arr, np.array([float(r["risk_opt"]) for r in rows_theta])),
            **preds,
        }
        for key in ["M", "N", "K", "n_eff", "gamma_eff", "risk"]:
            obs_key = f"obs_{key}_exp"
            pred_key = f"pred_{key}_exp"
            if pred_key in summary:
                summary[f"{key}_abs_error"] = abs(float(summary[obs_key]) - float(summary[pred_key]))
        summary_rows.append(summary)
        print(
            f"theta={theta:g} alpha={alpha:.3f} phase={preds['phase']} "
            f"risk={summary['obs_risk_exp']:.4f}/{summary['pred_risk_exp']:.4f} "
            f"M={summary['obs_M_exp']:.4f}/{summary['pred_M_exp']:.4f} "
            f"N={summary['obs_N_exp']:.4f}/{summary['pred_N_exp']:.4f}"
        )

    write_csv(outdir / "compute_optimal_curves.csv", curve_rows)
    write_csv(outdir / "compute_optimal_summary.csv", summary_rows)
    maybe_plot(outdir, curve_rows, summary_rows, args.no_plots)
    print(f"Wrote results to {outdir}")


if __name__ == "__main__":
    main()
