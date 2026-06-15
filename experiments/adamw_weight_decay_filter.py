"""Sanity check for AdamW decoupled weight decay as a shrinkage filter.

The theorem in notes/08_adamw_weight_decay_filter.md says that AdamW has the
same damped q=1/2 preconditioned spectrum as Adam, but decoupled weight decay
adds a separate shrinkage cutoff.  For fixed effective eigenvalues

    mu_i = lambda_i (lambda_i + rho)^(-1/2),

population AdamW learns coordinate i only if both

    N * gamma * mu_i >= O(1)       (enough time)
    mu_i >= O(lambda_wd)           (not over-shrunk)

hold.  Equivalently, the cutoff is controlled by

    mu_i >= max(1 / (N * gamma), lambda_wd).
"""

from __future__ import annotations

import argparse
import numpy as np


def exact_adamw_bias_factor(
    mu: np.ndarray, gamma: float, steps: int, wd: float
) -> np.ndarray:
    """Return |e_N / w_star| for the deterministic decoupled-WD recursion.

    The scalar recursion is

        e_{t+1} = (1 - gamma * (mu_i + wd)) e_t - gamma * wd * w_star,

    with w_0=0, so e_0=-w_star.  The closed form is

        e_N / w_star = -[wd/(mu+wd) + mu/(mu+wd) r^N],
        r = 1 - gamma * (mu+wd).
    """
    denom = mu + wd
    r = 1.0 - gamma * denom
    if np.any(np.abs(r) >= 1.0):
        raise ValueError("unstable scalar recursion: reduce gamma or wd")
    if wd == 0.0:
        return np.abs(r**steps)
    return np.abs(wd / denom + (mu / denom) * (r**steps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--wds", type=str, default="0,1e-4,1e-3,1e-2")
    args = parser.parse_args()

    idx = np.arange(1, args.dimension + 1, dtype=np.float64)
    lam = idx ** (-args.a)
    lam /= lam[0]
    mu = lam / np.sqrt(lam + args.rho)

    n = args.steps * args.gamma
    print(f"dimension={args.dimension}")
    print(f"steps={args.steps}")
    print(f"gamma={args.gamma}")
    print(f"n=N*gamma={n:.3f}")
    print(f"rho={args.rho}")

    for wd in [float(x) for x in args.wds.split(",")]:
        theta = max(1.0 / n, wd)
        k_time = int(np.count_nonzero(mu >= 1.0 / n))
        k_wd = int(np.count_nonzero(mu >= wd)) if wd > 0 else args.dimension
        k_pred = int(np.count_nonzero(mu >= theta))
        filt = exact_adamw_bias_factor(mu, args.gamma, args.steps, wd)
        k_half = int(np.count_nonzero(filt <= 0.5))
        k_10 = int(np.count_nonzero(filt <= 0.1))
        floor = wd / (mu + wd) if wd > 0 else np.zeros_like(mu)
        k_floor_half = int(np.count_nonzero(floor <= 0.5))
        print(
            f"wd={wd:.1e} "
            f"theta=max(1/n,wd)={theta:.3e} "
            f"K_time={k_time} K_wd={k_wd} K_pred={k_pred} "
            f"K_half={k_half} ratio_half_to_pred={k_half / max(k_pred, 1):.3f} "
            f"K_10={k_10} ratio_10_to_pred={k_10 / max(k_pred, 1):.3f} "
            f"K_floor_half={k_floor_half}"
        )


if __name__ == "__main__":
    main()
