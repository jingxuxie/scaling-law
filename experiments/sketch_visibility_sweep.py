"""Sketch / feature visibility experiments for partial Adam/RMSProp exponents.

This script tests the next theory stage: optimizer-dependent exponents are
controlled by the spectral information visible to coordinatewise second moments.

For each covariance representation Sigma, the script computes:
  d_j = Sigma_jj                         coordinate variance profile
  P_j = (d_j + rho)^(-1/2)               RMSProp/Adam diagonal preconditioner
  Sigma_eff = P^{1/2} Sigma P^{1/2}      transformed covariance

It then fits:
  alpha_orig    from eigenvalues of Sigma
  alpha_eff     from eigenvalues of Sigma_eff
  q_eff_hat     = 1 - alpha_eff / alpha_orig
  count slopes  K(n)=#{lambda_eff >= 1/n}
  diag-profile slope and commutator diagnostics

Cases:
  aligned          : optimizer coordinates are eigenvectors; q_eff ~ 1/2
  flat             : scalar preconditioner; q_eff ~ 0
  diagonal_profile : synthetic profile d_i=lambda_i^theta; q_eff ~ theta/2
  band             : random rotations inside comparable-eigenvalue bands; q_eff ~ 1/2
  haar             : full random orthogonal mixing; usually q_eff ~ 0
  gaussian_sketch  : rectangular Gaussian sketch; usually flattens coordinate variances

Examples:
  python experiments/sketch_visibility_sweep.py --quick
  python experiments/sketch_visibility_sweep.py --cases aligned,diagonal_profile,band,haar,gaussian_sketch --dimension 1024 --ambient-dim 4096
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
    n = len(vals)
    if n < 20:
        return float("nan")
    a = int(lo * n)
    b = max(a + 10, int(hi * n))
    b = min(b, n)
    ranks = np.arange(1, n + 1, dtype=float)
    X = np.vstack([np.ones(b - a), np.log(ranks[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(vals[a:b]), rcond=None)
    return float(coef[1])


def learned_count(mu: np.ndarray, n_values: np.ndarray) -> np.ndarray:
    mu = np.sort(mu[np.isfinite(mu) & (mu > 0)])[::-1]
    return np.array([np.searchsorted(-mu, -1.0 / n, side="right") for n in n_values], dtype=float)


def fit_count_slope(mu: np.ndarray, n_values: np.ndarray) -> float:
    k = learned_count(mu, n_values)
    mask = k > 0
    if mask.sum() < 5:
        return float("nan")
    X = np.vstack([np.ones(mask.sum()), np.log(n_values[mask])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(k[mask]), rcond=None)
    return float(coef[1])


def random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    z = rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def make_bands(lam: np.ndarray, kappa: float) -> list[slice]:
    bands: list[slice] = []
    n = len(lam)
    start = 0
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
    sigma = np.zeros((d, d), dtype=float)
    for sl in make_bands(lam, kappa):
        block_lam = lam[sl]
        q = random_orthogonal(rng, len(block_lam))
        sigma[sl, sl] = q.T @ np.diag(block_lam) @ q
    return sigma


def covariance_for_case(
    case: str,
    lam: np.ndarray,
    rng: np.random.Generator,
    rho: float,
    theta: float,
    band_kappa: float,
    ambient_dim: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return Sigma, optional external profile d, and description.

    If external_profile is not None, it is used instead of diag(Sigma) to build P.
    This supports the synthetic diagonal_profile case d_i=lambda_i^theta.
    """
    d = len(lam)
    if case == "aligned":
        sigma = np.diag(lam)
        return sigma, np.diag(sigma).copy(), "eigenbasis coordinates"
    if case == "flat":
        sigma = np.diag(lam)
        profile = np.full(d, float(np.mean(lam)))
        return sigma, profile, "scalar/flat diagonal profile"
    if case == "diagonal_profile":
        sigma = np.diag(lam)
        profile = lam ** theta
        return sigma, profile, f"synthetic visible profile theta={theta:g}"
    if case == "band":
        sigma = block_random_rotation(rng, lam, band_kappa)
        return sigma, np.diag(sigma).copy(), f"block rotations, kappa={band_kappa:g}"
    if case == "haar":
        q = random_orthogonal(rng, d)
        sigma = q.T @ np.diag(lam) @ q
        return sigma, np.diag(sigma).copy(), "full Haar orthogonal rotation"
    if case == "gaussian_sketch":
        # Rectangular sketch S: m x D.  Use ambient power-law spectrum, then z=Sx.
        m = d
        D = ambient_dim
        idx = np.arange(1, D + 1, dtype=float)
        # Infer a from first two lam entries when possible.
        a_hat = -math.log(lam[1] / lam[0]) / math.log(2.0) if len(lam) > 1 else 1.5
        ambient_lam = idx ** (-a_hat)
        ambient_lam /= ambient_lam[0]
        S = rng.normal(size=(m, D)) / math.sqrt(D)
        sigma = (S * ambient_lam) @ S.T
        return sigma, np.diag(sigma).copy(), f"Gaussian sketch m={m}, ambient={D}"
    raise ValueError(f"unknown case: {case}")


def effective_covariance(sigma: np.ndarray, profile: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    p = 1.0 / np.sqrt(profile + rho)
    sqrtp = np.sqrt(p)
    eff = (sqrtp[:, None] * sigma) * sqrtp[None, :]
    return eff, p


def op_norm_symmetric(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord=2))


def run_one(
    case: str,
    d: int,
    a: float,
    rho: float,
    theta: float,
    band_kappa: float,
    ambient_dim: int,
    n_values: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    idx = np.arange(1, d + 1, dtype=float)
    lam = idx ** (-a)
    lam /= lam[0]
    sigma, profile, desc = covariance_for_case(case, lam, rng, rho, theta, band_kappa, ambient_dim)
    eff, p = effective_covariance(sigma, profile, rho)

    eig_orig = np.linalg.eigvalsh(sigma)[::-1]
    eig_eff = np.linalg.eigvalsh(eff)[::-1]

    alpha_orig = -fit_power_slope(eig_orig)
    alpha_eff = -fit_power_slope(eig_eff)
    q_eff = 1.0 - alpha_eff / alpha_orig if alpha_orig > 0 else float("nan")
    count_orig = fit_count_slope(eig_orig, n_values)
    count_eff = fit_count_slope(eig_eff, n_values)
    diag_slope = fit_power_slope(profile)
    theta_sorted = -diag_slope / alpha_orig if alpha_orig > 0 else float("nan")

    Pmat = np.diag(p)
    comm = Pmat @ sigma - sigma @ Pmat
    denom = max(op_norm_symmetric(Pmat @ sigma), 1e-300)
    comm_ratio = op_norm_symmetric(comm) / denom

    return {
        "case": case,
        "description": desc,
        "dimension": d,
        "ambient_dim": ambient_dim if case == "gaussian_sketch" else d,
        "a": a,
        "rho": rho,
        "theta_input": theta if case == "diagonal_profile" else float("nan"),
        "band_kappa": band_kappa if case == "band" else float("nan"),
        "alpha_orig_hat": alpha_orig,
        "alpha_eff_hat": alpha_eff,
        "q_eff_hat": q_eff,
        "count_orig_slope_hat": count_orig,
        "count_eff_slope_hat": count_eff,
        "diag_profile_slope": diag_slope,
        "theta_sorted_diag_hat": theta_sorted,
        "diag_condition": float(np.max(profile) / np.min(profile)),
        "commutator_ratio": comm_ratio,
        "pred_q_eff_simple": (theta / 2.0 if case == "diagonal_profile" else (0.5 if case in {"aligned", "band"} else 0.0)),
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(outdir: Path, rows: list[dict[str, float | str]], no_plots: bool) -> None:
    if no_plots or plt is None:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cases = [str(r["case"]) for r in rows]
    q = np.array([float(r["q_eff_hat"]) for r in rows])
    pred = np.array([float(r["pred_q_eff_simple"]) for r in rows])
    x = np.arange(len(rows))
    fig = plt.figure(figsize=(max(8, len(rows) * 0.5), 4))
    plt.scatter(x, q, label="observed q_eff")
    plt.scatter(x, pred, marker="x", label="simple prediction")
    plt.xticks(x, cases, rotation=60, ha="right")
    plt.ylabel("q_eff")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "q_eff_by_case.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results/sketch_visibility"))
    parser.add_argument("--cases", type=str, default="aligned,flat,diagonal_profile,band,haar,gaussian_sketch")
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--ambient-dim", type=int, default=4096)
    parser.add_argument("--a-values", type=str, default="1.25,1.5,2.0")
    parser.add_argument("--rho-values", type=str, default="1e-12,1e-8")
    parser.add_argument("--theta-values", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--band-kappa", type=float, default=2.0)
    parser.add_argument("--n-min", type=float, default=10.0)
    parser.add_argument("--n-max", type=float, default=10000.0)
    parser.add_argument("--num-n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.dimension = min(args.dimension, 256)
        args.ambient_dim = min(args.ambient_dim, 2048)
        args.a_values = "1.5"
        args.rho_values = "1e-8"
        args.theta_values = "0,0.5,1"
        args.num_n = 25
        args.outdir = Path("experiments/results/sketch_visibility_quick")

    rng = np.random.default_rng(args.seed)
    n_values = np.geomspace(args.n_min, args.n_max, args.num_n)
    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    rows: list[dict[str, float | str]] = []
    for a in parse_list(args.a_values, float):
        for rho in parse_list(args.rho_values, float):
            for case in cases:
                theta_values = parse_list(args.theta_values, float) if case == "diagonal_profile" else [float("nan")]
                for theta in theta_values:
                    row = run_one(case, args.dimension, a, rho, theta, args.band_kappa, args.ambient_dim, n_values, rng)
                    rows.append(row)
                    print(
                        f"case={case:16s} a={a:g} rho={rho:g} theta={theta if np.isfinite(theta) else '-'} "
                        f"q_eff={row['q_eff_hat']:.3f} pred={row['pred_q_eff_simple']:.3f} "
                        f"comm={row['commutator_ratio']:.3f}"
                    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "sketch_visibility_sweep.csv", rows)
    with (args.outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)
    maybe_plot(args.outdir, rows, args.no_plots)
    print(f"Wrote {len(rows)} rows to {args.outdir / 'sketch_visibility_sweep.csv'}")


if __name__ == "__main__":
    main()
