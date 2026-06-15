"""Stochastic training grid for visible-profile exponents.

This experiment is a controlled stochastic-training version of the
q_eff = theta/2 interpolation theorem. It trains diagonal Gaussian linear
regression using fixed visible-profile preconditioners

    P_theta,i = (lambda_i**theta + rho)**(-1/2), theta in [0, 1].

The cases theta=0 and theta=1 correspond to scalar/flat preconditioning and
aligned Adam/RMSProp-like spectral preconditioning, respectively. Intermediate
values test the continuum predicted by the visible-spectrum theorem.

Examples
--------
Quick smoke test:

    python experiments/theta_profile_training_grid.py --quick

Main local run:

    python experiments/theta_profile_training_grid.py \
      --dimension 2048 \
      --theta-values 0,0.25,0.5,0.75,1 \
      --lr-values 0.001,0.003,0.005,0.01,0.03 \
      --checkpoints 250,500,1000,2000,4000,8000 \
      --trials 5 \
      --batch-size 16 \
      --outdir experiments/results/theta_profile_training_grid

Outputs
-------
training_curves.csv       per-trial and per-checkpoint risks
summary.csv               mean risk and fitted slope for every theta/lr
best_by_final_risk.csv    best learning rate for each theta
config.json               run configuration
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


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_problem(d: int, a: float, b: float, sigma: float, rng: np.random.Generator):
    idx = np.arange(1, d + 1, dtype=np.float64)
    lam = idx ** (-a)
    lam /= lam[0]
    source = idx ** (-b)
    signs = rng.choice(np.array([-1.0, 1.0]), size=d)
    w_star = signs * np.sqrt(source / lam)
    return lam, source, w_star, sigma


def sample_gradient(
    rng: np.random.Generator,
    w: np.ndarray,
    w_star: np.ndarray,
    lam: np.ndarray,
    sigma: float,
    batch_size: int,
) -> np.ndarray:
    x = rng.normal(size=(batch_size, len(lam))) * np.sqrt(lam)[None, :]
    noise = rng.normal(scale=sigma, size=batch_size)
    residual = x @ (w - w_star) - noise
    return (x.T @ residual) / batch_size


def excess_risk(w: np.ndarray, w_star: np.ndarray, lam: np.ndarray) -> float:
    e = w - w_star
    return float(np.sum(lam * e * e))


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float = 0.2, hi: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 4:
        return float("nan")
    a = int(lo * len(x))
    b = max(a + 3, int(hi * len(x)))
    b = min(b, len(x))
    X = np.vstack([np.ones(b - a), np.log(x[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(y[a:b]), rcond=None)
    return float(coef[1])


def train_one(
    lam: np.ndarray,
    w_star: np.ndarray,
    sigma: float,
    theta: float,
    lr: float,
    rho: float,
    checkpoints: list[int],
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    w = np.zeros_like(w_star)
    p = 1.0 / np.sqrt(lam ** theta + rho)
    max_step = max(checkpoints)
    checkpoint_set = set(checkpoints)
    rows: list[dict[str, float | int]] = []
    for step in range(1, max_step + 1):
        g = sample_gradient(rng, w, w_star, lam, sigma, batch_size)
        w -= lr * p * g
        if step in checkpoint_set:
            rows.append(
                {
                    "step": step,
                    "excess_risk": excess_risk(w, w_star, lam),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(curves: list[dict[str, object]], a: float, outdir: Path) -> list[dict[str, object]]:
    keys = sorted(set((float(r["theta"]), float(r["lr"])) for r in curves))
    summary: list[dict[str, object]] = []
    for theta, lr in keys:
        subset = [r for r in curves if float(r["theta"]) == theta and float(r["lr"]) == lr]
        steps = sorted(set(int(r["step"]) for r in subset))
        mean_risks = []
        for step in steps:
            vals = np.array([float(r["excess_risk"]) for r in subset if int(r["step"]) == step])
            mean_risks.append(float(vals.mean()))
            alpha_eff = a * (1.0 - theta / 2.0)
            summary.append(
                {
                    "theta": theta,
                    "lr": lr,
                    "step": step,
                    "risk_mean": float(vals.mean()),
                    "risk_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "risk_slope_loglog": float("nan"),
                    "q_eff_pred": theta / 2.0,
                    "alpha_eff_pred": alpha_eff,
                    "count_slope_pred": 1.0 / alpha_eff,
                }
            )
        slope = fit_slope(np.array(steps, dtype=float), np.array(mean_risks))
        for r in summary:
            if float(r["theta"]) == theta and float(r["lr"]) == lr:
                r["risk_slope_loglog"] = slope
    write_csv(outdir / "summary.csv", summary)

    final_step = max(int(r["step"]) for r in summary)
    best_rows = []
    for theta in sorted(set(float(r["theta"]) for r in summary)):
        finals = [r for r in summary if float(r["theta"]) == theta and int(r["step"]) == final_step]
        best = min(finals, key=lambda r: float(r["risk_mean"]))
        best_rows.append(dict(best))
    write_csv(outdir / "best_by_final_risk.csv", best_rows)
    return summary


def maybe_plot(summary: list[dict[str, object]], outdir: Path, no_plots: bool) -> None:
    if no_plots or plt is None or not summary:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    final_step = max(int(r["step"]) for r in summary)
    best_lrs = {}
    for theta in sorted(set(float(r["theta"]) for r in summary)):
        finals = [r for r in summary if float(r["theta"]) == theta and int(r["step"]) == final_step]
        best = min(finals, key=lambda r: float(r["risk_mean"]))
        best_lrs[theta] = float(best["lr"])

    fig = plt.figure()
    for theta, lr in best_lrs.items():
        rows = [r for r in summary if float(r["theta"]) == theta and float(r["lr"]) == lr]
        steps = np.array([float(r["step"]) for r in rows])
        risks = np.array([float(r["risk_mean"]) for r in rows])
        plt.loglog(steps, risks, marker="o", label=f"theta={theta:g}, lr={lr:g}")
    plt.xlabel("updates")
    plt.ylabel("mean excess risk")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "best_theta_risk_curves.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    thetas = np.array(sorted(best_lrs.keys()))
    final_risks = []
    for theta in thetas:
        rows = [
            r
            for r in summary
            if float(r["theta"]) == theta
            and float(r["lr"]) == best_lrs[theta]
            and int(r["step"]) == final_step
        ]
        final_risks.append(float(rows[0]["risk_mean"]))
    plt.plot(thetas, final_risks, marker="o")
    plt.xlabel("visible profile exponent theta")
    plt.ylabel("best final mean excess risk")
    plt.tight_layout()
    fig.savefig(plot_dir / "best_final_risk_vs_theta.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="experiments/results/theta_profile_training_grid")
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1e-8)
    parser.add_argument("--theta-values", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--lr-values", type=str, default="0.001,0.003,0.005,0.01,0.03")
    parser.add_argument("--checkpoints", type=str, default="250,500,1000,2000,4000,8000")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.outdir = "experiments/results/theta_profile_training_quick"
        args.dimension = 512
        args.theta_values = "0,0.5,1"
        args.lr_values = "0.003,0.01"
        args.checkpoints = "250,500,1000"
        args.trials = 2
        args.batch_size = 8

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    checkpoints = sorted(set(parse_int_list(args.checkpoints)))
    theta_values = parse_float_list(args.theta_values)
    lr_values = parse_float_list(args.lr_values)
    base_rng = np.random.default_rng(args.seed)

    curves: list[dict[str, object]] = []
    for trial in range(args.trials):
        problem_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        lam, source, w_star, sigma = make_problem(args.dimension, args.a, args.b, args.sigma, problem_rng)
        for theta in theta_values:
            for lr in lr_values:
                train_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
                rows = train_one(lam, w_star, sigma, theta, lr, args.rho, checkpoints, args.batch_size, train_rng)
                for r in rows:
                    r.update(
                        {
                            "trial": trial,
                            "theta": theta,
                            "lr": lr,
                            "a": args.a,
                            "b": args.b,
                            "dimension": args.dimension,
                            "q_eff_pred": theta / 2.0,
                        }
                    )
                curves.extend(rows)
                print(f"trial={trial} theta={theta:g} lr={lr:g} final_risk={rows[-1]['excess_risk']:.6g}", flush=True)

    write_csv(outdir / "training_curves.csv", curves)
    summary = summarize(curves, args.a, outdir)
    maybe_plot(summary, outdir, args.no_plots)
    print(f"Wrote results to {outdir}")
    print(f"  {outdir / 'training_curves.csv'}")
    print(f"  {outdir / 'summary.csv'}")
    print(f"  {outdir / 'best_by_final_risk.csv'}")


if __name__ == "__main__":
    main()
