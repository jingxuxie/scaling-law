"""Deterministic exponent-level experiments for optimizer-dependent scaling laws.

This script does not simulate stochastic training.  Instead it evaluates the
spectral filters proved in the notes and fits log-log slopes.  This is the
fastest way to test the exponent predictions before running expensive Monte
Carlo training.

Examples
--------
python experiments/exponent_level_filters.py --quick
python experiments/exponent_level_filters.py --dimension 200000 --n-min 10 --n-max 1000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np


@dataclass
class FitResult:
    name: str
    observed: float
    predicted: float


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float = 0.15, hi: float = 0.85) -> float:
    """Fit log y = intercept + slope log x over a central index range."""
    assert x.ndim == y.ndim == 1
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 5:
        return float("nan")
    a = int(lo * n)
    b = max(a + 3, int(hi * n))
    X = np.vstack([np.ones(b - a), np.log(x[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(y[a:b]), rcond=None)
    return float(coef[1])


def learned_count(mu: np.ndarray, n_values: np.ndarray, threshold_multiplier: float = 1.0) -> np.ndarray:
    """K(n) = #{i: mu_i >= threshold_multiplier / n}."""
    thresholds = threshold_multiplier / n_values
    return np.searchsorted(-mu, -thresholds, side="right").astype(float)


def bias_filter(mu: np.ndarray, s: np.ndarray, n_values: np.ndarray) -> np.ndarray:
    return np.array([np.sum(s / (1.0 + n * mu)) for n in n_values], dtype=float)


def variance_filter(mu: np.ndarray, n_eff: float, n_values: np.ndarray) -> np.ndarray:
    return np.array([np.sum(np.minimum(1.0, (n * mu) ** 2)) / n_eff for n in n_values], dtype=float)


def adamw_bias_filter(mu: np.ndarray, s: np.ndarray, n_values: np.ndarray, wd: float) -> np.ndarray:
    denom = mu + wd
    shrink = (wd / denom) ** 2 if wd > 0 else np.zeros_like(mu)
    opt_weight = (mu / denom) ** 2
    return np.array(
        [
            np.sum(s * (shrink + opt_weight / (1.0 + n * denom)))
            for n in n_values
        ],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=200_000)
    parser.add_argument("--quick", action="store_true", help="Use smaller dimension for fast smoke tests.")
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--rho", type=float, default=1e-12)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--n-min", type=float, default=10.0)
    parser.add_argument("--n-max", type=float, default=1000.0)
    parser.add_argument("--num-n", type=int, default=30)
    args = parser.parse_args()

    if args.quick:
        args.dimension = min(args.dimension, 50_000)
        args.num_n = min(args.num_n, 24)

    idx = np.arange(1, args.dimension + 1, dtype=np.float64)
    lam = idx ** (-args.a)
    lam /= lam[0]
    source = idx ** (-args.b)

    n_values = np.geomspace(args.n_min, args.n_max, args.num_n)

    mu_sgd = lam
    mu_adam = lam / np.sqrt(lam + args.rho)

    k_sgd = learned_count(mu_sgd, n_values)
    k_adam = learned_count(mu_adam, n_values)

    b_sgd = bias_filter(mu_sgd, source, n_values)
    b_adam = bias_filter(mu_adam, source, n_values)

    # AdamW learned count uses theta=max(1/n, wd), equivalently T_eff=min(n,1/wd).
    t_eff = np.minimum(n_values, 1.0 / args.wd)
    k_adamw = learned_count(mu_adam, t_eff)
    b_adamw = adamw_bias_filter(mu_adam, source, n_values, args.wd)

    results = [
        FitResult("SGD learned-count slope", fit_slope(n_values, k_sgd), 1.0 / args.a),
        FitResult("Adam/RMSProp learned-count slope", fit_slope(n_values, k_adam), 2.0 / args.a),
        FitResult("SGD bias slope", fit_slope(n_values, b_sgd), -(args.b - 1.0) / args.a),
        FitResult("Adam/RMSProp bias slope", fit_slope(n_values, b_adam), -2.0 * (args.b - 1.0) / args.a),
    ]

    # Fit AdamW before and after the weight-decay horizon when both ranges exist.
    before = n_values <= 0.8 / args.wd
    after = n_values >= 1.2 / args.wd
    if before.sum() >= 5:
        results.append(
            FitResult(
                "AdamW learned-count slope before WD ceiling",
                fit_slope(n_values[before], k_adamw[before], 0.0, 1.0),
                2.0 / args.a,
            )
        )
        results.append(
            FitResult(
                "AdamW bias slope before WD ceiling",
                fit_slope(n_values[before], b_adamw[before], 0.0, 1.0),
                -2.0 * (args.b - 1.0) / args.a,
            )
        )
    if after.sum() >= 5:
        results.append(
            FitResult(
                "AdamW learned-count slope after WD ceiling",
                fit_slope(n_values[after], k_adamw[after], 0.0, 1.0),
                0.0,
            )
        )

    print("deterministic_filter_exponent_experiment")
    print(f"dimension={args.dimension}")
    print(f"a={args.a}")
    print(f"b={args.b}")
    print(f"rho={args.rho}")
    print(f"wd={args.wd}")
    print(f"n_range=[{args.n_min}, {args.n_max}]")
    print("")
    print("name,observed,predicted,abs_error")
    for r in results:
        print(f"{r.name},{r.observed:.4f},{r.predicted:.4f},{abs(r.observed-r.predicted):.4f}")

    print("")
    print("sample_counts")
    stride = max(1, len(n_values) // 6)
    for n, ks, ka, kaw in zip(n_values[::stride], k_sgd[::stride], k_adam[::stride], k_adamw[::stride]):
        print(f"n={n:.2f}, K_sgd={ks:.0f}, K_adam={ka:.0f}, K_adamw={kaw:.0f}")


if __name__ == "__main__":
    main()
