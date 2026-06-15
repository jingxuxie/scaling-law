"""Band-limited Gaussian feature alignment experiment.

This script compares three feature-coordinate regimes with the same population
power-law spectrum:

1. aligned/eigenbasis features: Adam/RMSProp effective slope should be -a/2;
2. band-limited Gaussian features: effective slope should also be close to -a/2;
3. global Gaussian features: effective slope should be close to -a when the
   coordinate variances are flat enough.

It tests the theorem in notes/14_bandlimited_gaussian_features.md.
"""

from __future__ import annotations

import argparse
import numpy as np


def fit_power_slope(values: np.ndarray, lo: float = 0.10, hi: float = 0.80) -> float:
    vals = np.sort(np.asarray(values, dtype=float))[::-1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n = len(vals)
    if n < 20:
        return float("nan")
    a = int(lo * n)
    b = max(a + 10, int(hi * n))
    b = min(b, n)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    X = np.vstack([np.ones(b - a), np.log(ranks[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(vals[a:b]), rcond=None)
    return float(coef[1])


def make_geometric_bands(d: int, min_band: int = 32) -> list[np.ndarray]:
    """Contiguous dyadic-ish bands."""
    bands: list[np.ndarray] = []
    start = 0
    width = min_band
    while start < d:
        end = min(d, start + width)
        bands.append(np.arange(start, end))
        start = end
        width *= 2
    return bands


def block_gaussian_effective_spectrum(
    lam: np.ndarray,
    rho: float,
    row_fraction: float,
    rng: np.random.Generator,
    min_band: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return effective eigenvalues and coordinate variances for band-limited features."""
    eigs: list[np.ndarray] = []
    diag_vars: list[np.ndarray] = []
    for band in make_geometric_bands(len(lam), min_band=min_band):
        d_l = len(band)
        if d_l < 4:
            continue
        m_l = max(4, min(d_l - 1, int(row_fraction * d_l)))
        h_l = lam[band]
        s = rng.normal(scale=1.0 / np.sqrt(d_l), size=(m_l, d_l))
        sigma = (s * h_l[None, :]) @ s.T
        dcoord = np.diag(sigma).copy()
        p = 1.0 / np.sqrt(dcoord + rho)
        sqrtp = np.sqrt(p)
        eff = (sqrtp[:, None] * sigma) * sqrtp[None, :]
        vals = np.linalg.eigvalsh(eff)
        eigs.append(vals[vals > 1e-14])
        diag_vars.append(dcoord)
    return np.concatenate(eigs), np.concatenate(diag_vars)


def global_gaussian_effective_spectrum(
    lam: np.ndarray,
    rho: float,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    d = len(lam)
    s = rng.normal(scale=1.0 / np.sqrt(d), size=(m, d))
    sigma = (s * lam[None, :]) @ s.T
    dcoord = np.diag(sigma).copy()
    p = 1.0 / np.sqrt(dcoord + rho)
    sqrtp = np.sqrt(p)
    eff = (sqrtp[:, None] * sigma) * sqrtp[None, :]
    vals = np.linalg.eigvalsh(eff)
    return vals[vals > 1e-14], dcoord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--a", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=1e-10)
    parser.add_argument("--row-fraction", type=float, default=0.5)
    parser.add_argument("--min-band", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    idx = np.arange(1, args.dimension + 1, dtype=np.float64)
    lam = idx ** (-args.a)
    lam /= lam[0]

    # Aligned/eigenbasis effective spectrum.
    mu_aligned = lam / np.sqrt(lam + args.rho)

    # Band-limited Gaussian features.
    eig_band, diag_band = block_gaussian_effective_spectrum(
        lam, args.rho, args.row_fraction, rng, args.min_band
    )

    # Global Gaussian features with comparable number of feature coordinates.
    m_global = len(eig_band)
    eig_global, diag_global = global_gaussian_effective_spectrum(lam, args.rho, m_global, rng)

    print("bandlimited_feature_alignment_experiment")
    print(f"dimension={args.dimension}")
    print(f"a={args.a}")
    print(f"rho={args.rho}")
    print(f"row_fraction={args.row_fraction}")
    print(f"num_band_features={len(eig_band)}")
    print(f"num_global_features={m_global}")
    print("")
    print("case,effective_eigen_slope,predicted")
    print(f"aligned,{fit_power_slope(mu_aligned):.4f},{-args.a/2:.4f}")
    print(f"bandlimited_gaussian,{fit_power_slope(eig_band):.4f},{-args.a/2:.4f}")
    print(f"global_gaussian,{fit_power_slope(eig_global):.4f},near {-args.a:.4f}")
    print("")
    print("coordinate_variance_diagnostics")
    print(f"band_diag_condition={diag_band.max()/diag_band.min():.4f}")
    print(f"band_diag_slope={fit_power_slope(diag_band):.4f}")
    print(f"global_diag_condition={diag_global.max()/diag_global.min():.4f}")
    print(f"global_diag_slope={fit_power_slope(diag_global):.4f}")


if __name__ == "__main__":
    main()
