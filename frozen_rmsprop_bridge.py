"""Sanity checks for the frozen-RMSProp bridge theorem.

This script verifies the mechanism proved in notes/03_frozen_rmsprop_bridge.md:
for diagonal Gaussian linear regression, gradient second moments at a fixed anchor
satisfy

    E[g_i(anchor)^2] = lambda_i (c_e + 2 lambda_i e_i^2) \asymp c_e lambda_i.

Consequently a frozen RMSProp preconditioner

    A_i = (vhat_i + epsilon)^(-1/2)

is spectrally equivalent to the oracle damped q=1/2 preconditioner

    c_e^(-1/2) (lambda_i + epsilon / c_e)^(-1/2).

Run:
    python experiments/frozen_rmsprop_bridge.py --plot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class BridgeResult:
    dimension: int
    a: float
    b: float
    sigma: float
    epsilon: float
    burnin_samples: int
    mom_blocks: int
    mom_block_size: int
    c_e: float
    rho: float
    slope_log_vhat_vs_log_lambda: float
    slope_log_vstar_vs_log_lambda: float
    median_mu_ratio: float
    q10_mu_ratio: float
    q90_mu_ratio: float
    median_v_ratio: float
    q10_v_ratio: float
    q90_v_ratio: float


def make_problem(d: int, a: float, b: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lambdas, source energies s_i, and w_star."""
    rng = np.random.default_rng(seed)
    idx = np.arange(1, d + 1, dtype=float)
    lambdas = idx ** (-a)
    source = idx ** (-b)  # s_i = lambda_i * w_i^2
    signs = rng.choice(np.array([-1.0, 1.0]), size=d)
    w_star = signs * np.sqrt(source / lambdas)
    return lambdas, source, w_star


def median_of_means(values: np.ndarray, blocks: int) -> np.ndarray:
    """Coordinate-wise median of block means for an array of shape (n, d)."""
    n, d = values.shape
    if n % blocks != 0:
        raise ValueError("number of samples must be divisible by number of blocks")
    block_size = n // blocks
    block_means = values.reshape(blocks, block_size, d).mean(axis=1)
    return np.median(block_means, axis=0)


def fit_log_slope(x: np.ndarray, y: np.ndarray, start_frac: float = 0.05, end_frac: float = 0.80) -> float:
    """Fit slope of log(y) against log(x) over an interior spectral range."""
    d = len(x)
    lo = max(1, int(start_frac * d))
    hi = max(lo + 5, int(end_frac * d))
    mask = (x[lo:hi] > 0) & (y[lo:hi] > 0)
    coeff = np.polyfit(np.log(x[lo:hi][mask]), np.log(y[lo:hi][mask]), deg=1)
    return float(coeff[0])


def run_bridge_check(
    d: int,
    a: float,
    b: float,
    sigma: float,
    epsilon: float,
    burnin_samples: int,
    mom_blocks: int,
    seed: int,
    plot: bool,
    output_dir: Path,
) -> BridgeResult:
    rng = np.random.default_rng(seed + 1)
    lambdas, source, w_star = make_problem(d=d, a=a, b=b, seed=seed)

    # Anchor at zero: e = -w_star.
    e = -w_star
    c_e = sigma**2 + float(np.sum(lambdas * e**2))
    rho = epsilon / c_e

    x = rng.normal(size=(burnin_samples, d)) * np.sqrt(lambdas)[None, :]
    xi = rng.normal(scale=sigma, size=burnin_samples)
    residual = x @ e - xi
    gradients = residual[:, None] * x
    y_values = gradients**2

    vhat = median_of_means(y_values, blocks=mom_blocks)
    vstar = lambdas * (c_e + 2.0 * lambdas * e**2)

    # The theorem predicts vstar / (c_e * lambda_i) in [1, 3].
    v_ratio = vhat / (c_e * lambdas)

    # Frozen preconditioner versus oracle damped q=1/2 preconditioner.
    a_frozen = 1.0 / np.sqrt(vhat + epsilon)
    a_oracle = (c_e ** -0.5) / np.sqrt(lambdas + rho)
    mu_frozen = lambdas * a_frozen
    mu_oracle = lambdas * a_oracle
    mu_ratio = mu_frozen / mu_oracle

    slope_vhat = fit_log_slope(lambdas, vhat)
    slope_vstar = fit_log_slope(lambdas, vstar)

    result = BridgeResult(
        dimension=d,
        a=a,
        b=b,
        sigma=sigma,
        epsilon=epsilon,
        burnin_samples=burnin_samples,
        mom_blocks=mom_blocks,
        mom_block_size=burnin_samples // mom_blocks,
        c_e=c_e,
        rho=rho,
        slope_log_vhat_vs_log_lambda=slope_vhat,
        slope_log_vstar_vs_log_lambda=slope_vstar,
        median_mu_ratio=float(np.median(mu_ratio)),
        q10_mu_ratio=float(np.quantile(mu_ratio, 0.10)),
        q90_mu_ratio=float(np.quantile(mu_ratio, 0.90)),
        median_v_ratio=float(np.median(v_ratio)),
        q10_v_ratio=float(np.quantile(v_ratio, 0.10)),
        q90_v_ratio=float(np.quantile(v_ratio, 0.90)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_rmsprop_bridge_metrics.json").write_text(json.dumps(asdict(result), indent=2))

    if plot:
        import matplotlib.pyplot as plt

        idx = np.arange(1, d + 1)

        plt.figure()
        plt.loglog(idx, vhat, label="MoM estimate")
        plt.loglog(idx, vstar, label="population value")
        plt.loglog(idx, c_e * lambdas, label="c_e lambda_i")
        plt.xlabel("coordinate i")
        plt.ylabel("gradient second moment")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "gradient_second_moment.png", dpi=200)
        plt.close()

        plt.figure()
        plt.semilogx(idx, v_ratio)
        plt.axhline(1.0, linewidth=1)
        plt.axhline(3.0, linewidth=1)
        plt.xlabel("coordinate i")
        plt.ylabel("vhat_i / (c_e lambda_i)")
        plt.tight_layout()
        plt.savefig(output_dir / "vhat_ratio.png", dpi=200)
        plt.close()

        plt.figure()
        plt.semilogx(idx, mu_ratio)
        plt.axhline(1.0, linewidth=1)
        plt.xlabel("coordinate i")
        plt.ylabel("mu_frozen_i / mu_oracle_i")
        plt.tight_layout()
        plt.savefig(output_dir / "frozen_vs_oracle_mu_ratio.png", dpi=200)
        plt.close()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=1.3)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--burnin-samples", type=int, default=512)
    parser.add_argument("--mom-blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/frozen_rmsprop_bridge"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.burnin_samples % args.mom_blocks != 0:
        raise SystemExit("--burnin-samples must be divisible by --mom-blocks")
    result = run_bridge_check(
        d=args.d,
        a=args.a,
        b=args.b,
        sigma=args.sigma,
        epsilon=args.epsilon,
        burnin_samples=args.burnin_samples,
        mom_blocks=args.mom_blocks,
        seed=args.seed,
        plot=args.plot,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
