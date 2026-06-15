"""Coordinate-alignment experiment for adaptive diagonal preconditioning.

The coordinate-alignment theorem says that Adam/RMSProp behaves like spectral
q=1/2 preconditioning only when optimizer coordinates reveal the covariance
eigendirections.  In flat/rotated coordinates, the coordinate second moments
diag(Sigma) can be nearly flat, so diagonal adaptivity is close to a scalar
preconditioner and the effective spectrum keeps the original exponent.

This script compares:
  aligned coordinates:  Sigma = diag(lambda_i)
  flat Hadamard coordinates: diag(Sigma) = trace(H)/d exactly
  random orthogonal coordinates: optional finite-dimensional diagnostic
"""

from __future__ import annotations

import argparse
import numpy as np


def fit_power_slope(values: np.ndarray, lo: float = 0.10, hi: float = 0.80) -> float:
    """Fit log sorted_values = intercept + slope log rank."""
    vals = np.sort(values)[::-1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n = len(vals)
    a = int(lo * n)
    b = max(a + 5, int(hi * n))
    ranks = np.arange(1, n + 1, dtype=np.float64)
    X = np.vstack([np.ones(b - a), np.log(ranks[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(vals[a:b]), rcond=None)
    return float(coef[1])


def random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    """Haar-ish orthogonal matrix from QR."""
    z = rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-random", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    d = args.dimension
    idx = np.arange(1, d + 1, dtype=np.float64)
    lam = idx ** (-args.a)
    lam /= lam[0]

    # Original/SGD spectrum.
    slope_original = fit_power_slope(lam)

    # Aligned Adam/RMSProp: P_i=(lambda_i+rho)^(-1/2), effective eigenvalues
    # are lambda_i / sqrt(lambda_i+rho).
    mu_aligned = lam / np.sqrt(lam + args.rho)
    slope_aligned = fit_power_slope(mu_aligned)

    # Perfectly flat orthogonal coordinates, e.g. normalized Hadamard when it
    # exists.  diag(Q^T diag(lambda) Q)=trace(H)/d, so the diagonal preconditioner
    # is scalar and the spectrum is unchanged up to a constant.
    coord_var_flat = np.full(d, float(np.mean(lam)))
    scalar_p = 1.0 / np.sqrt(coord_var_flat[0] + args.rho)
    mu_flat = scalar_p * lam
    slope_flat = fit_power_slope(mu_flat)

    print("coordinate_alignment_experiment")
    print(f"dimension={d}")
    print(f"a={args.a}")
    print(f"rho={args.rho}")
    print("")
    print("case,eigen_slope,predicted")
    print(f"original_SGD,{slope_original:.4f},{-args.a:.4f}")
    print(f"aligned_RMSProp_Adam,{slope_aligned:.4f},{-args.a/2:.4f}")
    print(f"flat_Hadamard_coordinates,{slope_flat:.4f},{-args.a:.4f}")
    print("")
    print("coordinate_second_moment_diagnostics")
    print(f"aligned_diag_slope={fit_power_slope(lam):.4f}")
    print(f"flat_diag_min_over_max={coord_var_flat.min()/coord_var_flat.max():.4f}")

    if not args.skip_random:
        q = random_orthogonal(rng, d)
        # Sigma = Q^T diag(lam) Q.
        sigma = (q.T * lam) @ q
        diag_sigma = np.diag(sigma).copy()
        p = 1.0 / np.sqrt(diag_sigma + args.rho)
        # Effective covariance P^{1/2} Sigma P^{1/2}, where P=diag(p).
        sqrtp = np.sqrt(p)
        eff = (sqrtp[:, None] * sigma) * sqrtp[None, :]
        eig_eff = np.linalg.eigvalsh(eff)[::-1]
        slope_random = fit_power_slope(eig_eff)
        print(f"random_orthogonal_effective,{slope_random:.4f},near {-args.a:.4f}")
        print(f"random_diag_min={diag_sigma.min():.4e}")
        print(f"random_diag_median={np.median(diag_sigma):.4e}")
        print(f"random_diag_max={diag_sigma.max():.4e}")
        print(f"random_diag_condition={diag_sigma.max()/diag_sigma.min():.3f}")
        print(f"random_diag_slope={fit_power_slope(diag_sigma):.4f}")


if __name__ == "__main__":
    main()
