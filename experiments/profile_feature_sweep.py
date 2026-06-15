"""Profile-exponent feature sweep for partial Adam/RMSProp alignment.

This script measures the coordinate-variance profile exponent theta and the
resulting effective spectral exponent in several feature-coordinate models:

  * synthetic profiles d_i = lambda_i^theta;
  * aligned/eigenbasis coordinates;
  * flat/global coordinates;
  * band-limited orthogonal rotations;
  * band-limited Gaussian features;
  * global Gaussian features.

The clean interpolation theorem predicts q_eff = theta/2 when feature
coordinates preserve spectral bands.  The Gaussian-feature cases are diagnostic:
they can deviate because the feature map also changes the covariance spectrum,
not only the coordinate-variance profile.

Example:
    python experiments/profile_feature_sweep.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def fit_power_slope(values: np.ndarray, lo: float = 0.10, hi: float = 0.80) -> float:
    vals = np.sort(np.asarray(values, dtype=float))[::-1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) < 20:
        return float("nan")
    n = len(vals)
    a = int(lo * n)
    b = max(a + 10, int(hi * n))
    b = min(b, n)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    X = np.vstack([np.ones(b - a), np.log(ranks[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(vals[a:b]), rcond=None)
    return float(coef[1])


def estimate_theta_from_diag(diag_vars: np.ndarray, a: float) -> float:
    slope = fit_power_slope(diag_vars)
    return float(max(0.0, min(1.5, -slope / a))) if np.isfinite(slope) else float("nan")


def qeff_from_effective_spectrum(eigs: np.ndarray, a: float) -> tuple[float, float]:
    slope = fit_power_slope(eigs)
    if not np.isfinite(slope):
        return float("nan"), float("nan")
    alpha_eff = -slope
    q_eff = 1.0 - alpha_eff / a
    return float(q_eff), float(alpha_eff)


def make_geometric_bands(d: int, min_band: int = 32) -> list[np.ndarray]:
    bands: list[np.ndarray] = []
    start = 0
    width = min_band
    while start < d:
        end = min(d, start + width)
        bands.append(np.arange(start, end))
        start = end
        width *= 2
    return bands


def random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    z = rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def bandlimited_orthogonal_effective_spectrum(
    lam: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    min_band: int,
) -> tuple[np.ndarray, np.ndarray]:
    eigs: list[np.ndarray] = []
    diag_vars: list[np.ndarray] = []
    for band in make_geometric_bands(len(lam), min_band=min_band):
        h = lam[band]
        q = random_orthogonal(rng, len(band))
        sigma = (q.T * h) @ q
        dcoord = np.diag(sigma).copy()
        p = 1.0 / np.sqrt(dcoord + rho)
        eff = (np.sqrt(p)[:, None] * sigma) * np.sqrt(p)[None, :]
        eigs.append(np.linalg.eigvalsh(eff))
        diag_vars.append(dcoord)
    return np.concatenate(eigs), np.concatenate(diag_vars)


def block_gaussian_effective_spectrum(
    lam: np.ndarray,
    rho: float,
    row_fraction: float,
    rng: np.random.Generator,
    min_band: int,
) -> tuple[np.ndarray, np.ndarray]:
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
        eff = (np.sqrt(p)[:, None] * sigma) * np.sqrt(p)[None, :]
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
    eff = (np.sqrt(p)[:, None] * sigma) * np.sqrt(p)[None, :]
    vals = np.linalg.eigvalsh(eff)
    return vals[vals > 1e-14], dcoord


def add_row(rows: list[dict[str, object]], case: str, a: float, rho: float, eigs: np.ndarray, diag_vars: np.ndarray, theta_pred: float | None, note: str) -> None:
    theta_hat = estimate_theta_from_diag(diag_vars, a)
    q_eff_obs, alpha_eff_obs = qeff_from_effective_spectrum(eigs, a)
    pred_theta = theta_hat if theta_pred is None else theta_pred
    rows.append(
        {
            "case": case,
            "note": note,
            "a": a,
            "rho": rho,
            "num_features": len(eigs),
            "diag_slope": fit_power_slope(diag_vars),
            "theta_hat": theta_hat,
            "theta_pred": pred_theta,
            "q_eff_obs": q_eff_obs,
            "q_eff_pred": 0.5 * pred_theta if np.isfinite(pred_theta) else float("nan"),
            "alpha_eff_obs": alpha_eff_obs,
            "alpha_eff_pred": a * (1.0 - 0.5 * pred_theta) if np.isfinite(pred_theta) else float("nan"),
            "effective_slope_obs": fit_power_slope(eigs),
            "diag_condition": float(np.max(diag_vars) / np.min(diag_vars)),
        }
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(path: Path, rows: list[dict[str, object]], no_plots: bool) -> None:
    if no_plots or plt is None or not rows:
        return
    plot_dir = path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pred = np.array([float(r["q_eff_pred"]) for r in rows])
    obs = np.array([float(r["q_eff_obs"]) for r in rows])
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.any():
        fig = plt.figure()
        plt.scatter(pred[mask], obs[mask])
        lo = min(float(pred[mask].min()), float(obs[mask].min()))
        hi = max(float(pred[mask].max()), float(obs[mask].max()))
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel("predicted q_eff")
        plt.ylabel("observed q_eff")
        plt.tight_layout()
        fig.savefig(plot_dir / "qeff_observed_vs_predicted.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="experiments/results/profile_feature_main")
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--a-values", type=str, default="1.25,1.5,2.0")
    parser.add_argument("--theta-values", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--rho", type=float, default=1e-10)
    parser.add_argument("--row-fraction", type=float, default=0.5)
    parser.add_argument("--min-band", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.dimension = 1024
        args.a_values = "1.5"
        args.theta_values = "0,0.5,1.0"
        args.outdir = "experiments/results/profile_feature_quick"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for a in parse_float_list(args.a_values):
        idx = np.arange(1, args.dimension + 1, dtype=np.float64)
        lam = idx ** (-a)
        lam /= lam[0]

        for theta in parse_float_list(args.theta_values):
            diag_vars = lam**theta
            eigs = lam / np.sqrt(diag_vars + args.rho)
            add_row(rows, f"synthetic_theta_{theta:g}", a, args.rho, eigs, diag_vars, theta, "controlled profile")

        add_row(rows, "aligned", a, args.rho, lam / np.sqrt(lam + args.rho), lam, 1.0, "theory-clean")
        flat_diag = np.full_like(lam, float(np.mean(lam)))
        add_row(rows, "flat", a, args.rho, lam / np.sqrt(flat_diag + args.rho), flat_diag, 0.0, "theory-clean")

        eig_bo, diag_bo = bandlimited_orthogonal_effective_spectrum(lam, args.rho, rng, args.min_band)
        add_row(rows, "bandlimited_orthogonal", a, args.rho, eig_bo, diag_bo, 1.0, "theory-clean")

        eig_bg, diag_bg = block_gaussian_effective_spectrum(lam, args.rho, args.row_fraction, rng, args.min_band)
        add_row(rows, "bandlimited_gaussian", a, args.rho, eig_bg, diag_bg, None, "diagnostic: spectrum also changes")

        eig_global, diag_global = global_gaussian_effective_spectrum(lam, args.rho, len(eig_bg), rng)
        add_row(rows, "global_gaussian", a, args.rho, eig_global, diag_global, None, "diagnostic: global mixing")

    write_csv(outdir / "profile_feature_sweep.csv", rows)
    maybe_plot(outdir, rows, args.no_plots)
    for r in rows:
        print(
            f"case={r['case']} a={r['a']} theta_hat={float(r['theta_hat']):.3f} "
            f"q_obs={float(r['q_eff_obs']):.3f} q_pred={float(r['q_eff_pred']):.3f} "
            f"alpha_obs={float(r['alpha_eff_obs']):.3f} alpha_pred={float(r['alpha_eff_pred']):.3f}"
        )
    print(f"Wrote results to {outdir}")


if __name__ == "__main__":
    main()
