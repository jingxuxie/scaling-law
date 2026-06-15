"""Random-feature visibility experiment.

This script is the next bridge beyond diagonal/alignment toy models.  It creates
feature-coordinate covariances Sigma = S H S^T for several feature maps, measures
how much spectral information is visible to coordinatewise second moments, and
optionally runs a small stochastic-training comparison.

The central diagnostic is

    q_eff = 1 - alpha_eff / alpha_orig,

where alpha_orig is the fitted eigenvalue exponent of Sigma and alpha_eff is the
fitted eigenvalue exponent of P^{1/2} Sigma P^{1/2}, with
P_j=(Sigma_jj+rho)^(-1/2).

Expected outcomes:
  aligned / bandlimited features: q_eff close to 1/2
  global Gaussian features:       q_eff close to 0

Example quick run:

    python experiments/random_feature_visibility_experiment.py --quick

Main diagnostic run:

    python experiments/random_feature_visibility_experiment.py \
      --dimension 512 \
      --ambient-dim 4096 \
      --cases aligned,bandlimited_gaussian,global_gaussian,flat \
      --a-values 1.25,1.5,2.0 \
      --rho-values 1e-12,1e-8 \
      --outdir experiments/results/random_feature_visibility

Small stochastic-training run:

    python experiments/random_feature_visibility_experiment.py \
      --dimension 256 \
      --ambient-dim 2048 \
      --cases aligned,bandlimited_gaussian,global_gaussian \
      --run-training \
      --steps 4000 \
      --trials 3
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


def parse_list(text: str, typ=float):
    return [typ(x.strip()) for x in text.split(",") if x.strip()]


def fit_power_slope(values: np.ndarray, lo: float = 0.10, hi: float = 0.80) -> float:
    vals = np.sort(np.asarray(values, dtype=float))[::-1]
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


def block_random_covariance(rng: np.random.Generator, lam: np.ndarray, kappa: float) -> np.ndarray:
    d = len(lam)
    sigma = np.zeros((d, d), dtype=float)
    for sl in make_bands(lam, kappa):
        block = lam[sl]
        q = random_orthogonal(rng, len(block))
        sigma[sl, sl] = q.T @ np.diag(block) @ q
    return sigma


def make_covariance(
    case: str,
    lam: np.ndarray,
    ambient_lam: np.ndarray,
    rng: np.random.Generator,
    band_kappa: float,
) -> tuple[np.ndarray, str]:
    d = len(lam)
    if case == "aligned":
        return np.diag(lam), "eigenbasis features"
    if case == "flat":
        q = random_orthogonal(rng, d)
        return q.T @ np.diag(lam) @ q, "global orthogonal/flat-like features"
    if case == "bandlimited_gaussian":
        # Orthogonal block mixing is a clean proxy for random features whose rows
        # are supported on narrow spectral bands.
        return block_random_covariance(rng, lam, band_kappa), f"bandlimited Gaussian proxy kappa={band_kappa:g}"
    if case == "global_gaussian":
        # Rectangular isotropic Gaussian sketch.  This mixes spectral scales
        # globally and usually flattens coordinate variances.
        m = d
        D = len(ambient_lam)
        S = rng.normal(size=(m, D)) / math.sqrt(D)
        sigma = (S * ambient_lam) @ S.T
        return sigma, f"global Gaussian sketch m={m}, D={D}"
    raise ValueError(f"unknown case: {case}")


def effective_covariance(sigma: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    diag = np.diag(sigma).copy()
    p = 1.0 / np.sqrt(diag + rho)
    sqrtp = np.sqrt(p)
    eff = (sqrtp[:, None] * sigma) * sqrtp[None, :]
    return eff, p


def diagnostics(case: str, sigma: np.ndarray, rho: float, a: float, description: str) -> dict[str, object]:
    eff, p = effective_covariance(sigma, rho)
    eig = np.linalg.eigvalsh(sigma)[::-1]
    eig_eff = np.linalg.eigvalsh(eff)[::-1]
    alpha_orig = -fit_power_slope(eig)
    alpha_eff = -fit_power_slope(eig_eff)
    q_eff = 1.0 - alpha_eff / alpha_orig if alpha_orig > 0 else float("nan")
    diag = np.diag(sigma).copy()
    diag_slope = fit_power_slope(diag)
    theta_diag = -diag_slope / alpha_orig if alpha_orig > 0 and np.isfinite(diag_slope) else float("nan")
    Pmat = np.diag(p)
    comm = Pmat @ sigma - sigma @ Pmat
    denom = max(float(np.linalg.norm(Pmat @ sigma, ord=2)), 1e-300)
    comm_ratio = float(np.linalg.norm(comm, ord=2) / denom)
    return {
        "case": case,
        "description": description,
        "a_input": a,
        "rho": rho,
        "alpha_orig_hat": alpha_orig,
        "alpha_eff_hat": alpha_eff,
        "q_eff_hat": q_eff,
        "diag_profile_slope": diag_slope,
        "theta_diag_hat": theta_diag,
        "diag_condition": float(diag.max() / diag.min()),
        "commutator_ratio": comm_ratio,
    }


def make_target_from_covariance(sigma: np.ndarray, b: float, rng: np.random.Generator) -> np.ndarray:
    # Construct a target in the eigenbasis of sigma with source energies i^{-b}.
    eig, V = np.linalg.eigh(sigma)
    order = np.argsort(eig)[::-1]
    eig = eig[order]
    V = V[:, order]
    idx = np.arange(1, len(eig) + 1, dtype=float)
    source = idx ** (-b)
    coeff = np.sqrt(source / np.maximum(eig, 1e-300))
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(eig))
    return V @ (signs * coeff)


def sample_gradient(
    rng: np.random.Generator,
    sigma: np.ndarray,
    w: np.ndarray,
    w_star: np.ndarray,
    noise_std: float,
    batch_size: int,
) -> np.ndarray:
    x = rng.multivariate_normal(np.zeros(len(w)), sigma, size=batch_size)
    noise = rng.normal(scale=noise_std, size=batch_size)
    residual = x @ (w - w_star) - noise
    return (x.T @ residual) / batch_size


def risk(w: np.ndarray, w_star: np.ndarray, sigma: np.ndarray) -> float:
    e = w - w_star
    return float(e.T @ sigma @ e)


def train_case(
    rng: np.random.Generator,
    case: str,
    sigma: np.ndarray,
    rho: float,
    b: float,
    steps: int,
    batch_size: int,
    lr_sgd: float,
    lr_adam: float,
    beta2: float,
    eps: float,
    noise_std: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    w_star = make_target_from_covariance(sigma, b, rng)
    checkpoints = sorted(set([steps // 4, steps // 2, steps]))
    diag = np.diag(sigma).copy()
    p_oracle_coord = 1.0 / np.sqrt(diag + rho)
    for opt in ["sgd", "coord_adam_proxy", "spectral_oracle"]:
        w = np.zeros(len(diag), dtype=float)
        v = np.zeros(len(diag), dtype=float)
        if opt == "spectral_oracle":
            eig, V = np.linalg.eigh(sigma)
            eig = np.maximum(eig, 1e-300)
            Pspec = V @ np.diag(1.0 / np.sqrt(eig + rho)) @ V.T
        for t in range(1, steps + 1):
            g = sample_gradient(rng, sigma, w, w_star, noise_std, batch_size)
            if opt == "sgd":
                w -= lr_sgd * g
            elif opt == "coord_adam_proxy":
                v = beta2 * v + (1.0 - beta2) * (g * g)
                vhat = v / (1.0 - beta2**t)
                w -= lr_adam * g / np.sqrt(vhat + eps)
            elif opt == "spectral_oracle":
                w -= lr_adam * (Pspec @ g)
            if t in checkpoints:
                if opt == "coord_adam_proxy":
                    eff, _ = effective_covariance(sigma, rho)
                    alpha_orig = -fit_power_slope(np.linalg.eigvalsh(sigma)[::-1])
                    alpha_eff = -fit_power_slope(np.linalg.eigvalsh(eff)[::-1])
                    q_eff = 1.0 - alpha_eff / alpha_orig
                elif opt == "spectral_oracle":
                    q_eff = 0.5
                else:
                    q_eff = 0.0
                rows.append({"case": case, "optimizer": opt, "step": t, "risk": risk(w, w_star, sigma), "q_eff": q_eff})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results/random_feature_visibility"))
    parser.add_argument("--cases", type=str, default="aligned,bandlimited_gaussian,global_gaussian,flat")
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--ambient-dim", type=int, default=4096)
    parser.add_argument("--a-values", type=str, default="1.25,1.5,2.0")
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--rho-values", type=str, default="1e-12,1e-8")
    parser.add_argument("--band-kappa", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-sgd", type=float, default=0.05)
    parser.add_argument("--lr-adam", type=float, default=0.003)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    args = parser.parse_args()

    if args.quick:
        args.dimension = 256
        args.ambient_dim = 2048
        args.a_values = "1.5"
        args.rho_values = "1e-8"
        args.steps = 800
        args.trials = 1
        args.outdir = Path("experiments/results/random_feature_visibility_quick")

    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    diag_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    for a in parse_list(args.a_values, float):
        idx = np.arange(1, args.dimension + 1, dtype=float)
        lam = idx ** (-a)
        lam /= lam[0]
        ambient_idx = np.arange(1, args.ambient_dim + 1, dtype=float)
        ambient_lam = ambient_idx ** (-a)
        ambient_lam /= ambient_lam[0]
        for rho in parse_list(args.rho_values, float):
            for case in cases:
                sigma, desc = make_covariance(case, lam, ambient_lam, rng, args.band_kappa)
                row = diagnostics(case, sigma, rho, a, desc)
                diag_rows.append(row)
                print(f"case={case:22s} a={a:g} rho={rho:g} q_eff={row['q_eff_hat']:.3f} comm={row['commutator_ratio']:.3f}")
                if args.run_training:
                    for trial in range(args.trials):
                        trng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                        rows = train_case(trng, case, sigma, rho, args.b, args.steps, args.batch_size, args.lr_sgd, args.lr_adam, args.beta2, args.epsilon, args.noise_std)
                        for rr in rows:
                            rr.update({"trial": trial, "a": a, "rho": rho})
                        train_rows.extend(rows)

    write_csv(args.outdir / "visibility_diagnostics.csv", diag_rows)
    if train_rows:
        write_csv(args.outdir / "training_curves.csv", train_rows)
    print(f"Wrote diagnostics to {args.outdir / 'visibility_diagnostics.csv'}")


if __name__ == "__main__":
    main()
