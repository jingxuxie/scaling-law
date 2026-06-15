"""AdamW weight-decay compute-ceiling experiment.

This deterministic experiment complements compute_optimal_visible_spectrum.py. It
checks the theorem that fixed AdamW weight decay creates a horizon ceiling,
while compute-scaled weight decay can preserve the no-decay compute exponent.

The risk proxy is the visible-spectrum proxy with

    alpha = a * (1 - theta / 2),
    K(n) ~ n ** (1 / alpha),
    n_eff = min(N, 1 / delta(C)).

The script optimizes over M under C ~= M N and reports fitted compute-risk
slopes for several weight-decay schedules.

Example:
    python experiments/weight_decay_compute_sweep.py --quick

Main run:
    python experiments/weight_decay_compute_sweep.py \
      --a 1.5 --b 1.4 --theta 1.0 \
      --schedule-exponents 0,0.1,0.2,0.3,0.4,0.5 \
      --c-min 1e4 --c-max 1e10 --num-c 40 \
      --outdir experiments/results/weight_decay_compute
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


def risk_proxy(M: int, N: int, alpha: float, b: float, delta: float) -> float:
    # Continuous proxy: approximation + bias/optimization + variance.
    M = max(1, int(M))
    N = max(1, int(N))
    n_eff = min(float(N), float("inf") if delta <= 0 else 1.0 / delta)
    K = n_eff ** (1.0 / alpha)
    approx = M ** (1.0 - b)
    bias = K ** (1.0 - b)
    variance = K / N
    return float(approx + bias + variance)


def optimize_C(C: float, alpha: float, b: float, delta: float, grid_width: float, grid_size: int) -> dict[str, float]:
    # Use unified exponent m=max(alpha,b) for the center of the search grid.
    m = max(alpha, b)
    M0 = C ** (1.0 / (m + 1.0))
    factors = np.geomspace(1.0 / grid_width, grid_width, grid_size)
    best = None
    for fac in factors:
        M = max(1, int(round(M0 * fac)))
        N = max(1, int(round(C / M)))
        risk = risk_proxy(M, N, alpha, b, delta)
        row = {
            "C": float(C),
            "M": float(M),
            "N": float(N),
            "delta": float(delta),
            "risk": risk,
            "n_eff": min(float(N), float("inf") if delta <= 0 else 1.0 / delta),
        }
        if best is None or risk < best["risk"]:
            best = row
    assert best is not None
    return best


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(outdir: Path, summary: list[dict[str, float]], no_plots: bool) -> None:
    if no_plots or plt is None:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = np.array([r["schedule_exponent"] for r in summary])
    obs = np.array([r["risk_slope_obs"] for r in summary])
    pred = np.array([r["risk_slope_pred"] for r in summary])
    fig = plt.figure()
    plt.plot(x, obs, marker="o", label="observed")
    plt.plot(x, pred, marker="x", label="predicted cap")
    plt.xlabel("weight-decay schedule exponent s in delta(C)=C^{-s}")
    plt.ylabel("compute-risk slope")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "risk_slope_vs_weight_decay_schedule.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results/weight_decay_compute"))
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--schedule-exponents", type=str, default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--delta-prefactor", type=float, default=1.0)
    parser.add_argument("--c-min", type=float, default=1e4)
    parser.add_argument("--c-max", type=float, default=1e10)
    parser.add_argument("--num-c", type=int, default=40)
    parser.add_argument("--grid-width", type=float, default=50.0)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.c_min = 1e4
        args.c_max = 1e8
        args.num_c = 18
        args.grid_size = 160
        args.schedule_exponents = "0,0.2,0.4"
        args.outdir = Path("experiments/results/weight_decay_compute_quick")

    alpha = args.a * (1.0 - args.theta / 2.0)
    m = max(alpha, args.b)
    no_decay_exp = (args.b - 1.0) / (m + 1.0)
    critical_s = alpha / (m + 1.0)

    C_values = np.geomspace(args.c_min, args.c_max, args.num_c)
    all_rows: list[dict[str, float]] = []
    summary: list[dict[str, float]] = []

    for s in parse_float_list(args.schedule_exponents):
        rows = []
        for C in C_values:
            delta = args.delta_prefactor * (C ** (-s)) if s > 0 else args.delta_prefactor
            best = optimize_C(C, alpha, args.b, delta, args.grid_width, args.grid_size)
            best.update({"schedule_exponent": s, "alpha": alpha, "b": args.b, "theta": args.theta})
            rows.append(best)
            all_rows.append(best)
        C_arr = np.array([r["C"] for r in rows])
        R_arr = np.array([r["risk"] for r in rows])
        M_arr = np.array([r["M"] for r in rows])
        N_arr = np.array([r["N"] for r in rows])
        # Predicted risk slope is negative in log risk vs log compute.
        cap_exp = min(no_decay_exp, s * (args.b - 1.0) / alpha) if s > 0 else 0.0
        summary.append(
            {
                "schedule_exponent": s,
                "theta": args.theta,
                "alpha": alpha,
                "critical_s": critical_s,
                "risk_slope_obs": fit_slope(C_arr, R_arr),
                "risk_slope_pred": -cap_exp,
                "M_slope_obs": fit_slope(C_arr, M_arr),
                "N_slope_obs": fit_slope(C_arr, N_arr),
                "no_decay_risk_exponent": no_decay_exp,
                "final_risk": float(R_arr[-1]),
            }
        )
        print(
            f"s={s:g} obs={summary[-1]['risk_slope_obs']:.4f} "
            f"pred={summary[-1]['risk_slope_pred']:.4f} critical={critical_s:.4f}"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)
    write_csv(args.outdir / "weight_decay_compute_rows.csv", all_rows)
    write_csv(args.outdir / "weight_decay_compute_summary.csv", summary)
    maybe_plot(args.outdir, summary, args.no_plots)
    print(f"Wrote results to {args.outdir}")


if __name__ == "__main__":
    main()
