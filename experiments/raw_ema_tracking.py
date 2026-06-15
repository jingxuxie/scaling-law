"""Sanity check for raw RMSProp EMA tracking.

This script isolates the new theorem's probabilistic mechanism.  It simulates
heavy-tailed squared-gradient observations with the same product-Gaussian tail
shape as noise-dominated Gaussian linear regression,

    g_{t,i}^2 = c_t * lambda_i * Y_{t,i},   E[Y_{t,i}] = 1,

where Y is a product of two chi-square(1) variables.  Ordinary RMSProp EMA
should track d_t lambda_i, where d_t is the matching EMA of c_t.  Consequently
A_i=(v_i+eps)^(-1/2) should have effective exponent q_eff close to 1/2.
"""

from __future__ import annotations

import argparse
import numpy as np


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float:
    """Fit y ~ intercept + slope*x on the central index range."""
    n = len(x)
    a = int(lo * n)
    b = int(hi * n)
    X = np.vstack([np.ones(b - a), x[a:b]]).T
    coef, *_ = np.linalg.lstsq(X, y[a:b], rcond=None)
    return float(coef[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--beta2", type=float, default=0.995)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    idx = np.arange(1, args.dimension + 1, dtype=np.float64)
    lam = idx ** (-args.a)
    lam /= lam[0]
    log_lam = np.log(lam)

    v = np.zeros(args.dimension, dtype=np.float64)
    d = 0.0
    beta = args.beta2

    # Slowly varying residual scale c_t.  The EMA d_t is the correct comparison,
    # not the instantaneous c_t, because RMSProp averages past gradients.
    for t in range(args.steps):
        phase = 2.0 * np.pi * t / max(args.steps - 1, 1)
        c_t = 1.0 + 0.25 * np.sin(phase)
        # Product-Gaussian squared-gradient multiplier, mean one.
        y_noise = rng.chisquare(df=1.0, size=args.dimension) * rng.chisquare(
            df=1.0, size=args.dimension
        )
        z = c_t * lam * y_noise
        v = beta * v + (1.0 - beta) * z
        d = beta * d + (1.0 - beta) * c_t

    ratio = v / (d * lam)
    slope_v = fit_slope(log_lam, np.log(v + 1e-300))

    precond = 1.0 / np.sqrt(v + args.epsilon)
    # Only fit where the epsilon floor is inactive.
    active = v > 100.0 * args.epsilon
    if active.sum() > 100:
        q_eff = -fit_slope(log_lam[active], np.log(precond[active]))
    else:
        q_eff = float("nan")

    print(f"dimension={args.dimension}")
    print(f"steps={args.steps}")
    print(f"beta2={args.beta2}")
    print(f"effective_window≈{1.0 / (1.0 - args.beta2):.1f}")
    print(f"slope_log_v_vs_log_lambda={slope_v:.4f}")
    print(f"q_eff_from_preconditioner={q_eff:.4f}")
    print(f"median_v_over_dlambda={np.median(ratio):.4f}")
    print(f"q05_v_over_dlambda={np.quantile(ratio, 0.05):.4f}")
    print(f"q95_v_over_dlambda={np.quantile(ratio, 0.95):.4f}")


if __name__ == "__main__":
    main()
