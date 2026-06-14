"""Synthetic diagonal-H experiment for damped spectral preconditioning.

This script is intentionally small and dependency-light. It estimates the risk of
one-pass preconditioned SGD in a diagonal Gaussian linear model and can be used to
check the two predicted learned-mode regimes:

    K(n) ~ n^{1/[a(1-q)]}              for n << rho^{-(1-q)}
    K(n) ~ rho^{-q/a} n^{1/a}          for n >> rho^{-(1-q)}

Run, for example:

    python experiments/synthetic_damped_preconditioning.py --a 1.5 --b 1.5 --q 0.5 --rho 1e-4
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    d: int = 5000
    a: float = 1.5
    b: float = 1.5
    q: float = 0.5
    rho: float = 1e-4
    sigma: float = 1.0
    gamma: float = 0.05
    n_samples: int = 2000
    seed: int = 0


def make_problem(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    i = np.arange(1, cfg.d + 1, dtype=float)
    lam = i ** (-cfg.a)
    lam = lam / lam.sum()  # trace-normalize for numerical stability

    # Source condition: lambda_i * E[w_i^2] ~ i^{-b}.
    source = i ** (-cfg.b)
    w_std = np.sqrt(source / lam)
    w_star = rng.normal(size=cfg.d) * w_std
    return rng, lam, w_star


def run_once(cfg: Config) -> float:
    rng, lam, w_star = make_problem(cfg)
    p = (lam + cfg.rho) ** (-cfg.q)
    w = np.zeros(cfg.d)

    # Basic one-pass SGD; for speed, use dense diagonal Gaussian samples.
    for t in range(cfg.n_samples):
        # geometric step-decay block length N/log N
        block = max(1, int(cfg.n_samples / max(1.0, math.log(cfg.n_samples))))
        ell = t // block
        step = cfg.gamma / (2.0 ** ell)

        x = rng.normal(size=cfg.d) * np.sqrt(lam)
        y = float(x @ w_star + cfg.sigma * rng.normal())
        g_scalar = float(x @ w - y)
        w -= step * p * g_scalar * x

    err = w - w_star
    excess = float(np.sum(lam * err * err))
    return excess


def predicted_k(cfg: Config) -> float:
    n_eff = cfg.n_samples / max(1.0, math.log(cfg.n_samples))
    n = n_eff * cfg.gamma
    alpha = cfg.a * (1.0 - cfg.q)
    knee = cfg.rho ** (-(1.0 - cfg.q))
    if n <= knee:
        return n ** (1.0 / alpha)
    return (cfg.rho ** (-cfg.q / cfg.a)) * (n ** (1.0 / cfg.a))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=5000)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.5)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    risks = []
    for seed in range(args.trials):
        cfg = Config(
            d=args.d,
            a=args.a,
            b=args.b,
            q=args.q,
            rho=args.rho,
            sigma=args.sigma,
            gamma=args.gamma,
            n_samples=args.n_samples,
            seed=seed,
        )
        risks.append(run_once(cfg))

    cfg = Config(
        d=args.d,
        a=args.a,
        b=args.b,
        q=args.q,
        rho=args.rho,
        sigma=args.sigma,
        gamma=args.gamma,
        n_samples=args.n_samples,
    )
    print({
        "mean_excess_risk": float(np.mean(risks)),
        "std_excess_risk": float(np.std(risks)),
        "predicted_K": float(predicted_k(cfg)),
        "n_eff_gamma": float((args.n_samples / max(1.0, math.log(args.n_samples))) * args.gamma),
    })


if __name__ == "__main__":
    main()
