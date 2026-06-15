"""Stochastic alignment scaling experiment.

This experiment complements ``stochastic_training_scaling.py``.  It trains real
SGD/RMSProp/Adam/AdamW updates in Gaussian linear regression while varying the
optimizer-coordinate basis:

  aligned : optimizer coordinates are covariance eigenvectors
  band    : random rotations inside comparable-eigenvalue bands
  haar    : global Haar/random orthogonal rotation
  flat    : Hadamard/flat rotation when dimension is a power of two

The theory predicts that diagonal adaptive methods should have q_eff≈1/2 in
aligned and band-limited coordinates, but q_eff≈0 in flat/global coordinates.
The script reports both excess-risk slopes and a direct q_eff diagnostic from the
effective covariance P^{1/2} Sigma P^{1/2}.

Examples
--------
Quick smoke test:

    python experiments/stochastic_alignment_scaling.py --quick

Main local run:

    python experiments/stochastic_alignment_scaling.py \
        --dimension 512 \
        --cases aligned,band,flat,haar \
        --optimizers sgd,coord_oracle,rmsprop,adam,adamw,spectral_oracle \
        --checkpoints 250,500,1000,2000,4000,8000 \
        --trials 5 \
        --batch-size 16 \
        --outdir experiments/results/stochastic_alignment_main
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def parse_list(text: str, typ=str):
    return [typ(x.strip()) for x in text.split(",") if x.strip()]


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float = 0.25, hi: float = 1.0) -> float:
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


def fit_power_slope(vals: np.ndarray, lo: float = 0.10, hi: float = 0.80) -> float:
    vals = np.sort(np.asarray(vals, dtype=float))[::-1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) < 20:
        return float("nan")
    n = len(vals)
    a = int(lo * n)
    b = max(a + 10, int(hi * n))
    b = min(b, n)
    ranks = np.arange(1, n + 1, dtype=float)
    X = np.vstack([np.ones(b - a), np.log(ranks[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(vals[a:b]), rcond=None)
    return float(coef[1])


def random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    z = rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def hadamard(n: int) -> np.ndarray:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard case requires dimension to be a power of two")
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(n)


def make_bands(lam: np.ndarray, kappa: float) -> list[slice]:
    bands: list[slice] = []
    start = 0
    n = len(lam)
    while start < n:
        end = start + 1
        base = lam[start]
        while end < n and base / lam[end] <= kappa:
            end += 1
        bands.append(slice(start, end))
        start = end
    return bands


def block_random_rotation(rng: np.random.Generator, lam: np.ndarray, kappa: float) -> np.ndarray:
    d = len(lam)
    Q = np.zeros((d, d), dtype=float)
    for sl in make_bands(lam, kappa):
        Q[sl, sl] = random_orthogonal(rng, sl.stop - sl.start)
    return Q


def make_case(case: str, d: int, a: float, b: float, sigma_noise: float, rng: np.random.Generator, band_kappa: float):
    idx = np.arange(1, d + 1, dtype=float)
    lam = idx ** (-a)
    lam /= lam[0]
    source = idx ** (-b)
    signs = rng.choice(np.array([-1.0, 1.0]), size=d)
    w_star_eig = signs * np.sqrt(source / lam)

    if case == "aligned":
        Q = np.eye(d)
    elif case == "band":
        Q = block_random_rotation(rng, lam, band_kappa)
    elif case == "haar":
        Q = random_orthogonal(rng, d)
    elif case == "flat":
        Q = hadamard(d)
    else:
        raise ValueError(f"unknown case: {case}")

    # Optimizer-coordinate covariance is Sigma = Q^T diag(lam) Q, and
    # w*_opt = Q^T w*_eig because z = Q^T x_eig.
    w_star = Q.T @ w_star_eig
    sigma_mat = (Q.T * lam) @ Q
    diag_sigma = np.diag(sigma_mat).copy()
    return lam, Q, sigma_mat, diag_sigma, w_star, sigma_noise


def sample_gradient(rng, w, w_star, lam, Q, sigma_noise, batch_size):
    d = len(lam)
    x_eig = rng.normal(size=(batch_size, d)) * np.sqrt(lam)[None, :]
    z = x_eig @ Q
    noise = rng.normal(scale=sigma_noise, size=batch_size)
    residual = z @ (w - w_star) - noise
    return (z.T @ residual) / batch_size


def excess_risk(w: np.ndarray, w_star: np.ndarray, lam: np.ndarray, Q: np.ndarray) -> float:
    e_opt = w - w_star
    e_eig = Q @ e_opt
    return float(np.sum(lam * e_eig * e_eig))


def q_eff_from_preconditioner(p: np.ndarray, sigma_mat: np.ndarray, alpha_orig: float) -> float:
    if not np.all(np.isfinite(p)) or np.min(p) <= 0:
        return float("nan")
    sqrtp = np.sqrt(p)
    eff = (sqrtp[:, None] * sigma_mat) * sqrtp[None, :]
    eig_eff = np.linalg.eigvalsh(eff)[::-1]
    alpha_eff = -fit_power_slope(eig_eff)
    if not np.isfinite(alpha_eff) or alpha_orig <= 0:
        return float("nan")
    return float(1.0 - alpha_eff / alpha_orig)


def train_one(case: str, optimizer: str, args: argparse.Namespace, trial_seed: int):
    rng = np.random.default_rng(trial_seed)
    lam, Q, sigma_mat, diag_sigma, w_star, sigma_noise = make_case(
        case, args.dimension, args.a, args.b, args.sigma, rng, args.band_kappa
    )
    alpha_orig = -fit_power_slope(np.linalg.eigvalsh(sigma_mat)[::-1])
    d = args.dimension
    w = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    checkpoints = sorted(set(parse_list(args.checkpoints, int)))
    max_step = max(checkpoints)
    rows = []

    coord_p = 1.0 / np.sqrt(diag_sigma + args.rho)
    spectral_p_mat = Q.T @ np.diag(1.0 / np.sqrt(lam + args.rho)) @ Q

    for step in range(1, max_step + 1):
        g = sample_gradient(rng, w, w_star, lam, Q, sigma_noise, args.batch_size)
        if optimizer == "sgd":
            w -= args.lr_sgd * g
        elif optimizer == "coord_oracle":
            w -= args.lr_adaptive * coord_p * g
        elif optimizer == "spectral_oracle":
            w -= args.lr_adaptive * (spectral_p_mat @ g)
        elif optimizer == "rmsprop":
            v = args.beta2 * v + (1.0 - args.beta2) * (g * g)
            w -= args.lr_adaptive * g / np.sqrt(v + args.epsilon)
        elif optimizer == "adam":
            m = args.beta1 * m + (1.0 - args.beta1) * g
            v = args.beta2 * v + (1.0 - args.beta2) * (g * g)
            mhat = m / (1.0 - args.beta1**step)
            vhat = v / (1.0 - args.beta2**step)
            w -= args.lr_adaptive * mhat / np.sqrt(vhat + args.epsilon)
        elif optimizer == "adamw":
            m = args.beta1 * m + (1.0 - args.beta1) * g
            v = args.beta2 * v + (1.0 - args.beta2) * (g * g)
            mhat = m / (1.0 - args.beta1**step)
            vhat = v / (1.0 - args.beta2**step)
            w *= 1.0 - args.lr_adaptive * args.weight_decay
            w -= args.lr_adaptive * mhat / np.sqrt(vhat + args.epsilon)
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")

        if step in checkpoints:
            risk = excess_risk(w, w_star, lam, Q)
            if optimizer == "sgd":
                q_eff = 0.0
            elif optimizer == "spectral_oracle":
                q_eff = 0.5
            elif optimizer == "coord_oracle":
                q_eff = q_eff_from_preconditioner(coord_p, sigma_mat, alpha_orig)
            elif optimizer in {"rmsprop", "adam", "adamw"}:
                vhat = v / max(1e-12, 1.0 - args.beta2**step)
                p = 1.0 / np.sqrt(vhat + args.epsilon)
                q_eff = q_eff_from_preconditioner(p, sigma_mat, alpha_orig)
            else:
                q_eff = float("nan")
            rows.append(
                {
                    "case": case,
                    "optimizer": optimizer,
                    "step": step,
                    "excess_risk": risk,
                    "q_eff": q_eff,
                    "alpha_orig": alpha_orig,
                    "diag_condition": float(diag_sigma.max() / diag_sigma.min()),
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
    summary = []
    cases = sorted(set(str(r["case"]) for r in rows))
    opts = sorted(set(str(r["optimizer"]) for r in rows))
    for case in cases:
        for opt in opts:
            rs = [r for r in rows if r["case"] == case and r["optimizer"] == opt]
            if not rs:
                continue
            steps = sorted(set(int(r["step"]) for r in rs))
            mean_risks = []
            for step in steps:
                vals = np.array([float(r["excess_risk"]) for r in rs if int(r["step"]) == step])
                qs = np.array([float(r["q_eff"]) for r in rs if int(r["step"]) == step])
                mean_risks.append(float(vals.mean()))
                summary.append(
                    {
                        "case": case,
                        "optimizer": opt,
                        "step": step,
                        "risk_mean": float(vals.mean()),
                        "risk_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "q_eff_mean": float(np.nanmean(qs)),
                    }
                )
            slope = fit_slope(np.array(steps, dtype=float), np.array(mean_risks))
            for row in summary:
                if row["case"] == case and row["optimizer"] == opt:
                    row["risk_slope_loglog"] = slope
    write_csv(outdir / "summary.csv", summary)
    return summary


def maybe_plot(summary, outdir: Path, no_plots: bool) -> None:
    if no_plots or plt is None:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(set(str(r["case"]) for r in summary))
    for case in cases:
        fig = plt.figure()
        for opt in sorted(set(str(r["optimizer"]) for r in summary)):
            rs = [r for r in summary if r["case"] == case and r["optimizer"] == opt]
            if not rs:
                continue
            steps = np.array([float(r["step"]) for r in rs])
            risk = np.array([float(r["risk_mean"]) for r in rs])
            plt.loglog(steps, risk, marker="o", label=opt)
        plt.xlabel("updates")
        plt.ylabel("mean excess risk")
        plt.title(case)
        plt.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / f"risk_{case}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results/stochastic_alignment_main"))
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--cases", type=str, default="aligned,band,flat,haar")
    parser.add_argument("--optimizers", type=str, default="sgd,coord_oracle,rmsprop,adam,adamw,spectral_oracle")
    parser.add_argument("--checkpoints", type=str, default="250,500,1000,2000,4000,8000")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-sgd", type=float, default=0.05)
    parser.add_argument("--lr-adaptive", type=float, default=0.005)
    parser.add_argument("--rho", type=float, default=1e-8)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--band-kappa", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.dimension = 256
        args.cases = "aligned,band,haar"
        args.optimizers = "sgd,coord_oracle,rmsprop,adam"
        args.checkpoints = "200,400,800,1600"
        args.trials = 2
        args.batch_size = 8
        args.outdir = Path("experiments/results/stochastic_alignment_quick")

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    cases = parse_list(args.cases, str)
    optimizers = parse_list(args.optimizers, str)
    base_rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for trial in range(args.trials):
        for case in cases:
            for opt in optimizers:
                seed = int(base_rng.integers(0, 2**31 - 1))
                out = train_one(case, opt, args, seed)
                for r in out:
                    r["trial"] = trial
                    r["a"] = args.a
                    r["b"] = args.b
                    r["dimension"] = args.dimension
                rows.extend(out)
                print(f"trial={trial} case={case} opt={opt} final_risk={out[-1]['excess_risk']:.6g} q_eff={out[-1]['q_eff']:.3f}")

    write_csv(args.outdir / "training_curves.csv", rows)
    summary = summarize(rows, args.outdir)
    maybe_plot(summary, args.outdir, args.no_plots)
    print(f"Wrote results to {args.outdir}")
    print(f"  {args.outdir / 'training_curves.csv'}")
    print(f"  {args.outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
