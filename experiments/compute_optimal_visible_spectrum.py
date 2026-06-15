"""Compute-optimal visible-spectrum scaling experiment.

This deterministic experiment optimizes the proved spectral risk proxy over
model size M and sample/step count N under compute C ~= M*N.  It tests the
compute-optimal phase diagram for the visible-spectrum theorem.

For a visible exponent theta, the effective spectrum is

    mu_i = lambda_i (lambda_i**theta + rho)^(-1/2),
    lambda_i ~= i^(-a).

In the active, no-damping, no-weight-decay hard-source regime, the theory
predicts alpha = a * (1 - theta/2) and

    M_*(C) ~= C^{1/(alpha+1)}
    N_*(C) ~= C^{alpha/(alpha+1)}
    risk(C) ~= C^{-(b-1)/(alpha+1)}.
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


def prefix_sums(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(values)])


def range_sum(prefix: np.ndarray, lo: int, hi: int) -> float:
    lo = max(0, int(lo))
    hi = min(len(prefix) - 1, int(hi))
    if hi <= lo:
        return 0.0
    return float(prefix[hi] - prefix[lo])


def risk_proxy(M: int, N: int, source_prefix: np.ndarray, source: np.ndarray, mu: np.ndarray, dimension: int, delta: float) -> float:
    M = max(1, min(int(M), dimension))
    n_eff = float(N) if delta <= 0 else min(float(N), 1.0 / delta)
    approx = range_sum(source_prefix, M, dimension)
    idx = slice(0, M)
    bias = float(np.sum(source[idx] / (1.0 + n_eff * mu[idx])))
    variance = float(np.sum(np.minimum(1.0, (n_eff * mu[idx]) ** 2)) / max(N, 1))
    return approx + bias + variance


def optimize_for_compute(C: float, source_prefix: np.ndarray, source: np.ndarray, mu: np.ndarray, dimension: int, delta: float, grid_width: float, num_grid: int, alpha: float) -> dict[str, float]:
    M0 = C ** (1.0 / (alpha + 1.0))
    factors = np.geomspace(1.0 / grid_width, grid_width, num_grid)
    best: dict[str, float] | None = None
    for fac in factors:
        M = max(1, min(int(round(M0 * fac)), dimension))
        N = max(1, int(round(C / M)))
        risk = risk_proxy(M, N, source_prefix, source, mu, dimension, delta)
        row = {"C": float(C), "M": float(M), "N": float(N), "risk": risk}
        if best is None or risk < best["risk"]:
            best = row
    assert best is not None
    return best


def run(args: argparse.Namespace) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    idx = np.arange(1, args.dimension + 1, dtype=np.float64)
    lam = idx ** (-args.a)
    lam /= lam[0]
    source = idx ** (-args.b)
    source_prefix = prefix_sums(source)
    C_values = np.geomspace(args.c_min, args.c_max, args.num_c)
    all_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []

    for theta in parse_float_list(args.theta_values):
        mu = lam / np.sqrt(lam**theta + args.rho)
        alpha = args.a * (1.0 - theta / 2.0)
        for delta in parse_float_list(args.delta_values):
            rows = []
            for C in C_values:
                best = optimize_for_compute(C, source_prefix, source, mu, args.dimension, delta, args.grid_width, args.num_grid, alpha)
                best["theta"] = theta
                best["delta"] = delta
                best["alpha"] = alpha
                rows.append(best)
                all_rows.append(best)
            C_arr = np.array([r["C"] for r in rows])
            R_arr = np.array([r["risk"] for r in rows])
            M_arr = np.array([r["M"] for r in rows])
            N_arr = np.array([r["N"] for r in rows])
            pred_risk = -(args.b - 1.0) / (alpha + 1.0) if delta == 0 else float("nan")
            pred_M = 1.0 / (alpha + 1.0) if delta == 0 else float("nan")
            pred_N = alpha / (alpha + 1.0) if delta == 0 else float("nan")
            summary = {
                "theta": theta,
                "delta": delta,
                "alpha": alpha,
                "risk_slope_obs": fit_slope(C_arr, R_arr),
                "risk_slope_pred_no_decay": pred_risk,
                "M_slope_obs": fit_slope(C_arr, M_arr),
                "M_slope_pred_no_decay": pred_M,
                "N_slope_obs": fit_slope(C_arr, N_arr),
                "N_slope_pred_no_decay": pred_N,
                "final_risk": float(R_arr[-1]),
                "final_M": float(M_arr[-1]),
                "final_N": float(N_arr[-1]),
            }
            summary_rows.append(summary)
            print(
                f"theta={theta:g} delta={delta:g} risk_slope={summary['risk_slope_obs']:.4f} "
                f"pred={summary['risk_slope_pred_no_decay']:.4f} M={summary['M_slope_obs']:.4f} N={summary['N_slope_obs']:.4f}"
            )
    return all_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(outdir: Path, summary_rows: list[dict[str, float]], no_plots: bool) -> None:
    if no_plots or plt is None or not summary_rows:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in summary_rows if r["delta"] == 0]
    if not rows:
        return
    theta = np.array([r["theta"] for r in rows])
    obs = np.array([r["risk_slope_obs"] for r in rows])
    pred = np.array([r["risk_slope_pred_no_decay"] for r in rows])
    fig = plt.figure()
    plt.plot(theta, obs, marker="o", label="observed")
    plt.plot(theta, pred, marker="x", label="predicted")
    plt.xlabel("visible profile theta")
    plt.ylabel("compute-risk slope")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "risk_slope_vs_theta.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results/compute_optimal_visible"))
    parser.add_argument("--dimension", type=int, default=500_000)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--rho", type=float, default=1e-16)
    parser.add_argument("--theta-values", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--delta-values", type=str, default="0")
    parser.add_argument("--c-min", type=float, default=1e5)
    parser.add_argument("--c-max", type=float, default=1e9)
    parser.add_argument("--num-c", type=int, default=18)
    parser.add_argument("--grid-width", type=float, default=20.0)
    parser.add_argument("--num-grid", type=int, default=80)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.dimension = 100_000
        args.c_min = 1e4
        args.c_max = 1e7
        args.num_c = 10
        args.num_grid = 40
        args.theta_values = "0,0.5,1"
        args.outdir = Path("experiments/results/compute_optimal_visible_quick")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)
    rows, summary = run(args)
    write_csv(args.outdir / "compute_optimal_rows.csv", rows)
    write_csv(args.outdir / "compute_optimal_summary.csv", summary)
    maybe_plot(args.outdir, summary, args.no_plots)
    print(f"Wrote results to {args.outdir}")


if __name__ == "__main__":
    main()
