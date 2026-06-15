"""Stochastic training experiment for optimizer-dependent scaling laws.

This script simulates actual one-pass mini-batch training in diagonal Gaussian
linear regression and records excess risk over training.  It complements the
 deterministic filter experiments by checking whether real SGD/RMSProp/Adam/AdamW
 updates follow the predicted scaling trends.

Example quick run:

    python experiments/stochastic_training_scaling.py --quick

More useful local run:

    python experiments/stochastic_training_scaling.py \
        --dimension 2048 \
        --steps 4000 \
        --checkpoints 100,200,400,800,1600,3200,4000 \
        --trials 8 \
        --batch-size 16 \
        --optimizers sgd,oracle,rmsprop,adam,adamw \
        --outdir experiments/results/stochastic_main

Outputs:
    training_curves.csv      per-trial risk and q_eff diagnostics
    summary.csv              mean/std risks and fitted slopes by optimizer
    config.json              run configuration
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


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def make_problem(d: int, a: float, b: float, sigma: float, rng: np.random.Generator):
    idx = np.arange(1, d + 1, dtype=np.float64)
    lam = idx ** (-a)
    lam /= lam[0]
    source = idx ** (-b)
    signs = rng.choice(np.array([-1.0, 1.0]), size=d)
    w_star = signs * np.sqrt(source / lam)
    return lam, source, w_star, sigma


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


def estimate_v_slope(v: np.ndarray, lam: np.ndarray, eps: float) -> tuple[float, float]:
    """Return slope log(v) vs log(lambda), and q_eff=slope/2."""
    active = np.isfinite(v) & (v > 100.0 * eps) & (lam > 0)
    if active.sum() < 20:
        return float("nan"), float("nan")
    x = np.log(lam[active])
    y = np.log(v[active])
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = len(x)
    lo = int(0.1 * n)
    hi = max(lo + 10, int(0.9 * n))
    X = np.vstack([np.ones(hi - lo), x[lo:hi]]).T
    coef, *_ = np.linalg.lstsq(X, y[lo:hi], rcond=None)
    theta_hat = float(coef[1])
    return theta_hat, 0.5 * theta_hat


def sample_batch_gradient(
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


def estimate_frozen_preconditioner(
    rng: np.random.Generator,
    lam: np.ndarray,
    w_star: np.ndarray,
    sigma: float,
    batch_size: int,
    burnin_batches: int,
    rho: float,
) -> np.ndarray:
    """Estimate an initialization-time RMSProp preconditioner.

    The estimate is normalized by its median ratio to lambda so `rho` has the
    same interpretation as the oracle damped spectral floor.
    """
    w0 = np.zeros(len(lam), dtype=np.float64)
    v = np.zeros(len(lam), dtype=np.float64)
    batches = max(1, burnin_batches)
    for _ in range(batches):
        g = sample_batch_gradient(rng, w0, w_star, lam, sigma, batch_size)
        v += g * g
    v /= batches
    active = np.isfinite(v) & np.isfinite(lam) & (v > 0) & (lam > 0)
    if active.any():
        scale = float(np.median(v[active] / lam[active]))
        if scale > 0:
            v = v / scale
    return 1.0 / np.sqrt(v + rho)


def normalize_optimizer_name(name: str) -> str:
    aliases = {
        "oracle_rmsprop": "oracle",
        "oracle_preconditioner": "oracle",
    }
    return aliases.get(name.strip().lower(), name.strip().lower())


def train_one(
    optimizer: str,
    lam: np.ndarray,
    w_star: np.ndarray,
    sigma: float,
    checkpoints: list[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    d = len(lam)
    w = np.zeros(d, dtype=np.float64)
    m = np.zeros(d, dtype=np.float64)
    v = np.zeros(d, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []

    checkpoint_set = set(checkpoints)
    max_step = max(checkpoints)
    beta1 = args.beta1
    beta2 = args.beta2
    eps = args.epsilon

    if optimizer == "sgd":
        lr = args.lr_sgd
    elif optimizer == "oracle":
        lr = args.lr_adaptive
        oracle_p = 1.0 / np.sqrt(lam + args.rho)
    elif optimizer == "frozen_rmsprop":
        lr = args.lr_adaptive
        frozen_p = estimate_frozen_preconditioner(
            rng,
            lam,
            w_star,
            sigma,
            args.batch_size,
            args.burnin_batches,
            args.rho,
        )
    else:
        lr = args.lr_adaptive

    for step in range(1, max_step + 1):
        g = sample_batch_gradient(rng, w, w_star, lam, sigma, args.batch_size)
        if args.grad_clip > 0:
            norm = float(np.linalg.norm(g))
            if norm > args.grad_clip:
                g = g * (args.grad_clip / (norm + 1e-12))

        if optimizer == "sgd":
            w -= lr * g
        elif optimizer == "oracle":
            w -= lr * oracle_p * g
        elif optimizer == "frozen_rmsprop":
            w -= lr * frozen_p * g
        elif optimizer == "rmsprop":
            v = beta2 * v + (1.0 - beta2) * (g * g)
            w -= lr * g / np.sqrt(v + eps)
        elif optimizer == "adam":
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            mhat = m / (1.0 - beta1**step)
            vhat = v / (1.0 - beta2**step)
            w -= lr * mhat / np.sqrt(vhat + eps)
        elif optimizer == "adamw":
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            mhat = m / (1.0 - beta1**step)
            vhat = v / (1.0 - beta2**step)
            w *= 1.0 - lr * args.weight_decay
            w -= lr * mhat / np.sqrt(vhat + eps)
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")

        if step in checkpoint_set:
            risk = excess_risk(w, w_star, lam)
            if optimizer in {"rmsprop", "adam", "adamw"}:
                vv = v / max(1e-12, 1.0 - beta2**step)
                v_slope, q_eff = estimate_v_slope(vv, lam, eps)
            elif optimizer in {"oracle", "frozen_rmsprop"}:
                v_slope, q_eff = 1.0, 0.5
            else:
                v_slope, q_eff = 0.0, 0.0
            rows.append(
                {
                    "optimizer": optimizer,
                    "step": step,
                    "excess_risk": risk,
                    "v_slope": v_slope,
                    "q_eff": q_eff,
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


def summarize(rows: list[dict[str, object]], outdir: Path) -> list[dict[str, object]]:
    optimizers = sorted(set(str(r["optimizer"]) for r in rows))
    summary_rows: list[dict[str, object]] = []
    for opt in optimizers:
        opt_rows = [r for r in rows if r["optimizer"] == opt]
        steps = sorted(set(int(r["step"]) for r in opt_rows))
        mean_risks = []
        for s in steps:
            vals = np.array([float(r["excess_risk"]) for r in opt_rows if int(r["step"]) == s])
            qs = np.array([float(r["q_eff"]) for r in opt_rows if int(r["step"]) == s])
            mean_risks.append(float(vals.mean()))
            summary_rows.append(
                {
                    "optimizer": opt,
                    "step": s,
                    "risk_mean": float(vals.mean()),
                    "risk_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "q_eff_mean": float(np.nanmean(qs)),
                }
            )
        slope = fit_slope(np.array(steps, dtype=float), np.array(mean_risks))
        for r in summary_rows:
            if r["optimizer"] == opt:
                r["risk_slope_loglog"] = slope
    write_csv(outdir / "summary.csv", summary_rows)
    return summary_rows


def maybe_plot(summary_rows: list[dict[str, object]], outdir: Path, no_plots: bool) -> None:
    if no_plots or plt is None:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    opts = sorted(set(str(r["optimizer"]) for r in summary_rows))
    fig = plt.figure()
    for opt in opts:
        rs = [r for r in summary_rows if r["optimizer"] == opt]
        steps = np.array([float(r["step"]) for r in rs])
        risk = np.array([float(r["risk_mean"]) for r in rs])
        plt.loglog(steps, risk, marker="o", label=opt)
    plt.xlabel("updates")
    plt.ylabel("mean excess risk")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "risk_curves.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    for opt in opts:
        rs = [r for r in summary_rows if r["optimizer"] == opt]
        if not any(np.isfinite(float(r["q_eff_mean"])) for r in rs):
            continue
        steps = np.array([float(r["step"]) for r in rs])
        q = np.array([float(r["q_eff_mean"]) for r in rs])
        plt.semilogx(steps, q, marker="o", label=opt)
    plt.xlabel("updates")
    plt.ylabel("mean q_eff diagnostic")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "qeff_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="experiments/results/stochastic_main")
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoints", type=str, default=None)
    parser.add_argument("--n-values", type=str, default=None, help="Alias for --checkpoints; sets --steps to max value if --steps is omitted.")
    parser.add_argument("--trials", "--reps", dest="trials", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--optimizers", "--algorithms", dest="optimizers", type=str, default="sgd,oracle,rmsprop,adam,adamw")
    parser.add_argument("--lr-sgd", type=float, default=0.05)
    parser.add_argument("--lr-adaptive", type=float, default=0.005)
    parser.add_argument("--rho", type=float, default=1e-8)
    parser.add_argument("--epsilon", "--eps", dest="epsilon", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--burnin-batches", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.dimension = 512
        args.steps = 800
        args.checkpoints = "100,200,400,800"
        args.trials = 2
        args.batch_size = 8
        args.outdir = "experiments/results/stochastic_quick"

    if args.checkpoints is None:
        args.checkpoints = args.n_values or "100,200,400,800,1600,2000"
    if args.steps is None:
        args.steps = max(parse_int_list(args.checkpoints)) if args.n_values else 2000

    checkpoints = sorted(set(parse_int_list(args.checkpoints) + [args.steps]))
    checkpoints = [s for s in checkpoints if 1 <= s <= args.steps]
    optimizers = [normalize_optimizer_name(opt) for opt in parse_str_list(args.optimizers)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    all_rows: list[dict[str, object]] = []
    base_rng = np.random.default_rng(args.seed)
    for trial in range(args.trials):
        problem_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        lam, source, w_star, sigma = make_problem(args.dimension, args.a, args.b, args.sigma, problem_rng)
        for opt in optimizers:
            train_rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
            rows = train_one(opt, lam, w_star, sigma, checkpoints, args, train_rng)
            for r in rows:
                r["trial"] = trial
                r["a"] = args.a
                r["b"] = args.b
                r["dimension"] = args.dimension
            all_rows.extend(rows)
            print(f"trial={trial} optimizer={opt} final_risk={rows[-1]['excess_risk']:.6g}")

    write_csv(outdir / "training_curves.csv", all_rows)
    summary_rows = summarize(all_rows, outdir)
    maybe_plot(summary_rows, outdir, args.no_plots)
    print(f"Wrote results to {outdir}")
    print(f"  {outdir / 'training_curves.csv'}")
    print(f"  {outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
