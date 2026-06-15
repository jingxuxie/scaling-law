"""Transformed-source diagnostics for visible-spectrum experiments.

The visible-spectrum theorem requires not only an effective spectrum but also a
source condition in the transformed covariance basis.  This script measures that
condition for the same coordinate systems used in stochastic_alignment_scaling.py.

For each case, it computes

    Sigma_eff = P^{1/2} Sigma P^{1/2},
    u_star = P^{-1/2} w_star,
    s_tilde_i = mu_i <v_i, u_star>^2,

where (mu_i, v_i) are eigenpairs of Sigma_eff.  It then fits the slope of
s_tilde_i ordered by decreasing mu_i.  In aligned or invariant band cases, the
source exponent should be close to the original b.  In globally mixed cases, this
diagnostic reveals whether target/source rotation is influencing risk curves.

Quick run:

    python experiments/source_condition_diagnostic.py --quick

Main run:

    python experiments/source_condition_diagnostic.py \
      --dimension 512 \
      --cases aligned,band,flat,haar \
      --a-values 1.25,1.5,2.0 \
      --b-values 1.2,1.4,1.8 \
      --outdir experiments/results/source_diagnostic
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import stochastic_alignment_scaling as base

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def parse_list(text: str, typ=str):
    return [typ(x.strip()) for x in text.split(",") if x.strip()]


def fit_slope_against_rank(values: np.ndarray, lo: float = 0.10, hi: float = 0.80) -> float:
    vals = np.asarray(values, dtype=float)
    mask = np.isfinite(vals) & (vals > 0)
    vals = vals[mask]
    if len(vals) < 20:
        return float("nan")
    n = len(vals)
    a = int(lo * n)
    b = max(a + 10, int(hi * n))
    b = min(b, n)
    ranks = np.arange(1, n + 1, dtype=float)
    X = np.vstack([np.ones(b - a), np.log(ranks[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(vals[a:b]), rcond=None)
    return float(coef[1])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_one(case: str, d: int, a: float, b: float, rho: float, seed: int, band_kappa: float) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    lam, Q, sigma_mat, diag_sigma, w_star, _ = base.make_case(case, d, a, b, 0.0, rng, band_kappa)

    # Coordinatewise RMSProp/Adam proxy.
    p = 1.0 / np.sqrt(diag_sigma + rho)
    sqrtp = np.sqrt(p)
    sigma_eff = (sqrtp[:, None] * sigma_mat) * sqrtp[None, :]

    # u_star = P^{-1/2} w_star.  Since P=diag(p), P^{-1/2}=diag(1/sqrt(p)).
    u_star = w_star / sqrtp

    mu, V = np.linalg.eigh(sigma_eff)
    order = np.argsort(mu)[::-1]
    mu = mu[order]
    V = V[:, order]
    coeff = V.T @ u_star
    s_tilde = mu * coeff * coeff

    original_source = np.arange(1, d + 1, dtype=float) ** (-b)
    source_slope = fit_slope_against_rank(s_tilde)
    q_eff = base.q_eff_from_preconditioner(p, sigma_mat, a)
    eig_slope = -base.fit_power_slope(mu)

    # Compare total source mass; it should be invariant up to numerical error.
    original_mass = float(w_star.T @ sigma_mat @ w_star)
    transformed_mass = float(u_star.T @ sigma_eff @ u_star)

    return {
        "case": case,
        "dimension": d,
        "a": a,
        "b": b,
        "rho": rho,
        "q_eff": q_eff,
        "effective_eigen_slope": eig_slope,
        "source_slope_obs": source_slope,
        "source_slope_pred": -b,
        "source_slope_abs_error": abs(source_slope + b) if np.isfinite(source_slope) else float("nan"),
        "diag_condition": float(diag_sigma.max() / diag_sigma.min()),
        "total_source_mass_original": original_mass,
        "total_source_mass_transformed": transformed_mass,
        "total_source_mass_rel_error": abs(transformed_mass - original_mass) / max(1e-300, abs(original_mass)),
        "top_source_fraction_10pct": float(np.sum(s_tilde[: max(1, d // 10)]) / np.sum(s_tilde)),
    }


def maybe_plot(outdir: Path, rows: list[dict[str, object]], no_plots: bool) -> None:
    if no_plots or plt is None or not rows:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    cases = sorted(set(str(r["case"]) for r in rows))
    for case in cases:
        xs = [float(r["b"]) for r in rows if str(r["case"]) == case]
        ys = [-float(r["source_slope_obs"]) for r in rows if str(r["case"]) == case]
        plt.scatter(xs, ys, label=case)
    lo = min(float(r["b"]) for r in rows)
    hi = max(float(r["b"]) for r in rows)
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("target source exponent b")
    plt.ylabel("observed transformed source exponent")
    plt.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "source_exponent_observed_vs_target.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="experiments/results/source_diagnostic")
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--cases", type=str, default="aligned,band,flat,haar")
    parser.add_argument("--a-values", type=str, default="1.25,1.5,2.0")
    parser.add_argument("--b-values", type=str, default="1.2,1.4,1.8")
    parser.add_argument("--rho", type=float, default=1e-8)
    parser.add_argument("--band-kappa", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.dimension = 256
        args.cases = "aligned,band,flat,haar"
        args.a_values = "1.5"
        args.b_values = "1.4"
        args.outdir = "experiments/results/source_diagnostic_quick"

    rows: list[dict[str, object]] = []
    seed_rng = np.random.default_rng(args.seed)
    for case in parse_list(args.cases, str):
        for a in parse_list(args.a_values, float):
            for b in parse_list(args.b_values, float):
                seed = int(seed_rng.integers(0, 2**31 - 1))
                row = diagnostic_one(case, args.dimension, a, b, args.rho, seed, args.band_kappa)
                rows.append(row)
                print(
                    f"case={case:8s} a={a:g} b={b:g} q={row['q_eff']:.3f} "
                    f"source_exp={-row['source_slope_obs']:.3f} target={b:.3f}"
                )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    write_csv(outdir / "source_diagnostic.csv", rows)
    maybe_plot(outdir, rows, args.no_plots)
    print(f"Wrote {len(rows)} rows to {outdir / 'source_diagnostic.csv'}")


if __name__ == "__main__":
    main()
