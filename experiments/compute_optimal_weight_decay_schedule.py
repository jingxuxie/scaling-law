"""Compute-optimal scaling with compute-dependent AdamW weight decay.

This experiment tests the theorem in notes/20_weight_decay_compute_scaling.md
and notes/20_compute_optimal_weight_decay_scaling.md.  It optimizes the same
simplified risk proxy as compute_optimal_scaling.py,

    R(M,N,K) = M^(1-b) + K^(1-b) + K/N,

subject to C=MN and

    K <= N^(1/alpha),
    K <= delta(C)^(-1/alpha),    delta(C)=C^(-s).

For alpha=a*(1-theta/2), the predicted compute-risk exponent is

    beta_wd(theta,s) = min((b-1)/(max(alpha,b)+1), s*(b-1)/alpha).

Examples:
    python experiments/compute_optimal_weight_decay_schedule.py --quick

    python experiments/compute_optimal_weight_decay_schedule.py \
      --a 3.0 --b 1.4 --theta 1.0 \
      --s-values 0,0.05,0.1,0.2,0.4,0.8 \
      --c-min 1e4 --c-max 1e10 --num-c 40 \
      --outdir experiments/results/compute_optimal_wd_schedule

    python experiments/compute_optimal_weight_decay_schedule.py \
      --a 3.0 --b 1.4 --theta-values 0,0.5,1 \
      --s-values 0,0.05,0.1,0.2,0.4,0.8 \
      --c-min 1e4 --c-max 1e10 --num-c 40 \
      --outdir experiments/results/compute_optimal_weight_decay_schedule
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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


def optimize_for_C(C: float, alpha: float, b: float, grid_size: int, s: float) -> dict[str, float]:
    if s < 0:
        raise ValueError("s must be nonnegative")

    M_grid = np.geomspace(2.0, max(4.0, C / 2.0), grid_size)
    delta = C ** (-s)
    K_wd_cap = 1.0 if s == 0 else delta ** (-1.0 / alpha)

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


def predicted(alpha: float, b: float, s: float) -> dict[str, float | str]:
    m = max(alpha, b)
    beta_no_wd = (b - 1.0) / (m + 1.0)
    s_threshold = alpha / (m + 1.0)

    if s >= s_threshold:
        return {
            "phase": "cap_inactive_no_decay_law",
            "pred_risk_exp": -beta_no_wd,
            "pred_K_exp": 1.0 / (m + 1.0),
            "pred_M_exp": 1.0 / (m + 1.0),
            "pred_N_exp": m / (m + 1.0),
            "s_threshold": s_threshold,
            "beta_no_wd": beta_no_wd,
        }

    k_exp = s / alpha
    beta = (b - 1.0) * k_exp
    phase = "fixed_decay_floor" if s == 0 else "cap_active_decay_limited"
    m_exp = (1.0 - k_exp) / b
    return {
        "phase": phase,
        "pred_risk_exp": -beta,
        "pred_K_exp": k_exp,
        "pred_M_exp": m_exp,
        "pred_N_exp": 1.0 - m_exp,
        "s_threshold": s_threshold,
        "beta_no_wd": beta_no_wd,
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


def maybe_plot(outdir: Path, curves: list[dict[str, object]], summary: list[dict[str, object]], no_plots: bool) -> None:
    if no_plots or not summary:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    for theta in sorted(set(float(r["theta"]) for r in curves)):
        for s in sorted(set(float(r["s"]) for r in curves if float(r["theta"]) == theta)):
            rows = [r for r in curves if float(r["theta"]) == theta and float(r["s"]) == s]
            C = np.array([float(r["C"]) for r in rows])
            R = np.array([float(r["risk_opt"]) for r in rows])
            plt.loglog(C, R, marker="o", label=f"theta={theta:g}, s={s:g}")
    plt.xlabel("compute C")
    plt.ylabel("optimized risk proxy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "risk_vs_compute_by_s.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    for theta in sorted(set(float(r["theta"]) for r in summary)):
        rows = [r for r in summary if float(r["theta"]) == theta]
        s_vals = np.array([float(r["s"]) for r in rows])
        obs = -np.array([float(r["obs_risk_exp"]) for r in rows])
        pred = -np.array([float(r["pred_risk_exp"]) for r in rows])
        plt.plot(s_vals, obs, marker="o", label=f"theta={theta:g} observed")
        plt.plot(s_vals, pred, marker="x", linestyle="--", label=f"theta={theta:g} predicted")
    plt.xlabel("weight-decay schedule exponent s in delta(C)=C^{-s}")
    plt.ylabel("positive compute-risk exponent")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "risk_exponent_vs_decay_schedule.png", dpi=160)
    fig.savefig(plot_dir / "risk_exponent_vs_s.png", dpi=160)
    plt.close(fig)


def resolve_theta_values(theta: float, theta_values: str | None) -> list[float]:
    if theta_values is not None:
        values = parse_float_list(theta_values)
    else:
        values = [theta]
    if not values:
        raise ValueError("at least one theta value is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="experiments/results/compute_optimal_weight_decay_schedule")
    parser.add_argument("--a", type=float, default=3.0)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--theta-values", type=str, default=None)
    parser.add_argument("--s-values", type=str, default="0,0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--c-min", type=float, default=1e4)
    parser.add_argument("--c-max", type=float, default=1e10)
    parser.add_argument("--num-c", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=600)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.theta_values = "0,1"
        args.s_values = "0,0.1,0.4,0.8"
        args.c_min = 1e3
        args.c_max = 1e8
        args.num_c = 25
        args.grid_size = 300
        args.outdir = "experiments/results/compute_optimal_weight_decay_schedule_quick"

    theta_values = resolve_theta_values(args.theta, args.theta_values)
    s_values = parse_float_list(args.s_values)
    if not s_values:
        raise ValueError("at least one s value is required")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["resolved_theta_values"] = theta_values
    config["resolved_s_values"] = s_values
    with (outdir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    C_values = np.geomspace(args.c_min, args.c_max, args.num_c)
    curves: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for theta in theta_values:
        alpha = args.a * (1.0 - theta / 2.0)
        for s in s_values:
            rows = []
            pred = predicted(alpha, args.b, s)
            for C in C_values:
                row = optimize_for_C(C, alpha, args.b, args.grid_size, s)
                row.update({"a": args.a, "b": args.b, "theta": theta, "alpha": alpha, "s": s, **pred})
                curves.append(row)
                rows.append(row)

            C_arr = np.array([float(r["C"]) for r in rows])
            summ = {
                "a": args.a,
                "b": args.b,
                "theta": theta,
                "alpha": alpha,
                "s": s,
                **pred,
                "obs_risk_exp": fit_slope(C_arr, np.array([float(r["risk_opt"]) for r in rows])),
                "obs_K_exp": fit_slope(C_arr, np.array([float(r["K_opt"]) for r in rows])),
                "obs_M_exp": fit_slope(C_arr, np.array([float(r["M_opt"]) for r in rows])),
                "obs_N_exp": fit_slope(C_arr, np.array([float(r["N_opt"]) for r in rows])),
            }
            for key in ["risk", "K", "M", "N"]:
                summ[f"{key}_abs_error"] = abs(float(summ[f"obs_{key}_exp"]) - float(summ[f"pred_{key}_exp"]))
            summary.append(summ)
            print(
                f"theta={theta:g} s={s:g} phase={pred['phase']} "
                f"risk={summ['obs_risk_exp']:.4f}/{summ['pred_risk_exp']:.4f} "
                f"K={summ['obs_K_exp']:.4f}/{summ['pred_K_exp']:.4f}",
                flush=True,
            )

    write_csv(outdir / "weight_decay_schedule_curves.csv", curves)
    write_csv(outdir / "weight_decay_schedule_summary.csv", summary)
    write_csv(outdir / "wd_schedule_curves.csv", curves)
    write_csv(outdir / "wd_schedule_summary.csv", summary)
    maybe_plot(outdir, curves, summary, args.no_plots)
    print(f"Wrote results to {outdir}")


if __name__ == "__main__":
    main()
