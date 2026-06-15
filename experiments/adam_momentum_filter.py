"""Sanity check for Adam first-moment momentum as a temporal filter.

The theorem in notes/07_adam_momentum_filter.md says that beta1 momentum does
not change the spectral learned-mode count. It only changes constants and the
stability region. This script tests the deterministic population filter with a
fixed RMSProp/Adam-like preconditioner

    A_i = (lambda_i + rho)^(-1/2),  mu_i = A_i lambda_i.

For each beta1, the empirical half-learning cutoff should be a constant multiple
of the no-momentum cutoff {N * gamma * mu_i >= 1}.
"""

from __future__ import annotations

import argparse
import numpy as np


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float = 0.15, hi: float = 0.85) -> float:
    """Fit y ~ intercept + slope*x on a central range."""
    n = len(x)
    a = int(lo * n)
    b = int(hi * n)
    X = np.vstack([np.ones(b - a), x[a:b]]).T
    coef, *_ = np.linalg.lstsq(X, y[a:b], rcond=None)
    return float(coef[1])


def simulate_filter(mu: np.ndarray, gamma: float, steps: int, beta1: float) -> np.ndarray:
    """Return |e_N/e_0| for the population Adam-momentum recursion."""
    e = np.ones_like(mu)
    m = np.zeros_like(mu)
    for _ in range(steps):
        grad_eff = mu * e
        m = beta1 * m + (1.0 - beta1) * grad_eff
        e = e - gamma * m
    return np.abs(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--betas", type=str, default="0,0.5,0.9,0.95")
    args = parser.parse_args()

    idx = np.arange(1, args.dimension + 1, dtype=np.float64)
    lam = idx ** (-args.a)
    lam /= lam[0]
    mu = lam / np.sqrt(lam + args.rho)

    learned_pred = args.steps * args.gamma * mu >= 1.0
    k_pred = int(np.count_nonzero(learned_pred))
    print(f"dimension={args.dimension}")
    print(f"steps={args.steps}")
    print(f"gamma={args.gamma}")
    print(f"rho={args.rho}")
    print(f"predicted_cutoff_K={{N gamma mu >= 1}}={k_pred}")

    beta_values = [float(x) for x in args.betas.split(",")]
    base = simulate_filter(mu, args.gamma, args.steps, beta1=0.0)

    for beta in beta_values:
        filt = simulate_filter(mu, args.gamma, args.steps, beta1=beta)
        k_half = int(np.count_nonzero(filt <= 0.5))
        k_10 = int(np.count_nonzero(filt <= 0.1))
        active = learned_pred & (base > 1e-12) & (filt > 1e-12)
        med_ratio_to_beta0 = (
            float(np.median(filt[active] / base[active])) if active.any() else float("nan")
        )
        central = (filt > 1e-8) & (filt < 0.9)
        if central.sum() > 50:
            slope_log_filter_vs_log_mu = fit_slope(
                np.log(mu[central]), np.log(filt[central]), 0.0, 1.0
            )
        else:
            slope_log_filter_vs_log_mu = float("nan")
        print(
            f"beta1={beta:.2f} "
            f"K_half={k_half} ratio_half_to_pred={k_half / max(k_pred, 1):.3f} "
            f"K_10={k_10} ratio_10_to_pred={k_10 / max(k_pred, 1):.3f} "
            f"median_filter_ratio_to_beta0={med_ratio_to_beta0:.3f} "
            f"slope_log_filter_vs_log_mu={slope_log_filter_vs_log_mu:.3f}"
        )


if __name__ == "__main__":
    main()
