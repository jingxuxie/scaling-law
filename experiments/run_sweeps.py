"""Run local experiment sweeps for optimizer-dependent scaling laws.

This is the main script to run after cloning the repo.  It performs two kinds of
experiments:

1. Deterministic exponent-level filter sweeps over (a, b, rho, weight decay).
   These evaluate the spectral filters proved in the notes and fit log-log
   slopes for learned-mode counts and bias filters.

2. Coordinate-alignment sweeps comparing aligned coordinates, flat/Hadamard-like
   coordinates, and optionally random orthogonal coordinates.

Examples
--------
Quick smoke test:

    python experiments/run_sweeps.py --quick --mode all

Larger deterministic sweep:

    python experiments/run_sweeps.py \
        --mode filters \
        --dimension 300000 \
        --a-values 1.25,1.5,2.0 \
        --b-values 1.2,1.4,1.8 \
        --rho-values 1e-16,1e-12,1e-8 \
        --wd-values 0,1e-4,1e-3,1e-2 \
        --n-min 10 --n-max 10000 --num-n 50

Coordinate-alignment sweep with random rotations:

    python experiments/run_sweeps.py \
        --mode alignment \
        --alignment-dimension 512 \
        --a-values 1.25,1.5,2.0 \
        --rho-values 1e-12,1e-8 \
        --include-random

Outputs are written to experiments/results/<run_name>/ by default.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:  # plotting is optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def fit_slope(x: np.ndarray, y: np.ndarray, lo: float = 0.15, hi: float = 0.85) -> float:
    """Fit log(y) = c + slope log(x) over a central portion of the data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 5:
        return float("nan")
    a = int(lo * n)
    b = max(a + 3, int(hi * n))
    b = min(b, n)
    X = np.vstack([np.ones(b - a), np.log(x[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(y[a:b]), rcond=None)
    return float(coef[1])


def spectrum_and_source(dimension: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(1, dimension + 1, dtype=np.float64)
    lam = idx ** (-a)
    lam /= lam[0]
    source = idx ** (-b)
    return lam, source


def learned_count(mu: np.ndarray, n_values: np.ndarray) -> np.ndarray:
    """K(n)=#{i: mu_i >= 1/n}; assumes mu is sorted decreasing."""
    thresholds = 1.0 / n_values
    return np.searchsorted(-mu, -thresholds, side="right").astype(float)


def bias_filter(mu: np.ndarray, source: np.ndarray, n_values: np.ndarray) -> np.ndarray:
    out = np.empty_like(n_values, dtype=float)
    for k, n in enumerate(n_values):
        out[k] = float(np.sum(source / (1.0 + n * mu)))
    return out


def adamw_bias_filter(mu: np.ndarray, source: np.ndarray, n_values: np.ndarray, wd: float) -> np.ndarray:
    if wd == 0.0:
        return bias_filter(mu, source, n_values)
    denom = mu + wd
    shrink = (wd / denom) ** 2
    opt_weight = (mu / denom) ** 2
    out = np.empty_like(n_values, dtype=float)
    for k, n in enumerate(n_values):
        out[k] = float(np.sum(source * (shrink + opt_weight / (1.0 + n * denom))))
    return out


def adam_theory_count(n_values: np.ndarray, a: float, rho: float) -> np.ndarray:
    """Theory K_{rho,1/2}(n) up to constants."""
    if rho <= 0:
        return n_values ** (2.0 / a)
    knee = rho ** (-0.5)
    return np.where(
        n_values <= knee,
        n_values ** (2.0 / a),
        rho ** (-1.0 / (2.0 * a)) * n_values ** (1.0 / a),
    )


def safe_rel_error(obs: float, pred: float) -> float:
    if not np.isfinite(obs) or not np.isfinite(pred):
        return float("nan")
    return abs(obs - pred)


def run_filter_sweeps(args: argparse.Namespace, outdir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    n_values = np.geomspace(args.n_min, args.n_max, args.num_n)

    for a, b, rho, wd in itertools.product(
        parse_float_list(args.a_values),
        parse_float_list(args.b_values),
        parse_float_list(args.rho_values),
        parse_float_list(args.wd_values),
    ):
        lam, source = spectrum_and_source(args.dimension, a, b)
        mu_sgd = lam
        mu_adam = lam / np.sqrt(lam + rho)

        k_sgd = learned_count(mu_sgd, n_values)
        k_adam = learned_count(mu_adam, n_values)
        t_eff = np.minimum(n_values, np.inf if wd == 0.0 else 1.0 / wd)
        k_adamw = learned_count(mu_adam, t_eff)

        b_sgd = bias_filter(mu_sgd, source, n_values)
        b_adam = bias_filter(mu_adam, source, n_values)
        b_adamw = adamw_bias_filter(mu_adam, source, n_values, wd)

        # Predicted slopes are fit on the same n-grid using the asymptotic formula,
        # so crossings of damping or weight-decay knees are handled gracefully.
        kth_sgd = n_values ** (1.0 / a)
        kth_adam = adam_theory_count(n_values, a, rho)
        kth_adamw = adam_theory_count(t_eff, a, rho)
        bth_sgd = kth_sgd ** (1.0 - b)
        bth_adam = kth_adam ** (1.0 - b)
        bth_adamw = kth_adamw ** (1.0 - b)

        row: dict[str, float | str] = {
            "a": a,
            "b": b,
            "rho": rho,
            "wd": wd,
            "dimension": args.dimension,
            "n_min": args.n_min,
            "n_max": args.n_max,
            "num_n": args.num_n,
            "rho_knee_n": float("inf") if rho == 0 else rho ** (-0.5),
            "wd_knee_n": float("inf") if wd == 0 else 1.0 / wd,
            "sgd_count_slope_obs": fit_slope(n_values, k_sgd),
            "sgd_count_slope_pred": fit_slope(n_values, kth_sgd),
            "adam_count_slope_obs": fit_slope(n_values, k_adam),
            "adam_count_slope_pred": fit_slope(n_values, kth_adam),
            "adamw_count_slope_obs": fit_slope(n_values, k_adamw),
            "adamw_count_slope_pred": fit_slope(n_values, kth_adamw),
            "sgd_bias_slope_obs": fit_slope(n_values, b_sgd),
            "sgd_bias_slope_pred": fit_slope(n_values, bth_sgd),
            "adam_bias_slope_obs": fit_slope(n_values, b_adam),
            "adam_bias_slope_pred": fit_slope(n_values, bth_adam),
            "adamw_bias_slope_obs": fit_slope(n_values, b_adamw),
            "adamw_bias_slope_pred": fit_slope(n_values, bth_adamw),
        }
        for key in list(row):
            if key.endswith("_obs"):
                pred_key = key[:-4] + "_pred"
                row[key[:-4] + "_abs_error"] = safe_rel_error(float(row[key]), float(row[pred_key]))
        rows.append(row)
        print(
            "filter",
            f"a={a}",
            f"b={b}",
            f"rho={rho:g}",
            f"wd={wd:g}",
            f"adam_count={row['adam_count_slope_obs']:.3f}/{row['adam_count_slope_pred']:.3f}",
            f"adamw_count={row['adamw_count_slope_obs']:.3f}/{row['adamw_count_slope_pred']:.3f}",
        )

    csv_path = outdir / "filter_sweep.csv"
    write_csv(csv_path, rows)
    maybe_plot_filter_summary(outdir, rows, args.no_plots)
    return rows


def fit_power_slope(values: np.ndarray, lo: float = 0.10, hi: float = 0.80) -> float:
    vals = np.sort(np.asarray(values, dtype=float))[::-1]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n = len(vals)
    if n < 10:
        return float("nan")
    a = int(lo * n)
    b = max(a + 5, int(hi * n))
    b = min(b, n)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    X = np.vstack([np.ones(b - a), np.log(ranks[a:b])]).T
    coef, *_ = np.linalg.lstsq(X, np.log(vals[a:b]), rcond=None)
    return float(coef[1])


def random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    z = rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def run_alignment_sweeps(args: argparse.Namespace, outdir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    rng = np.random.default_rng(args.seed)

    for d, a, rho in itertools.product(
        parse_int_list(args.alignment_dimensions),
        parse_float_list(args.a_values),
        parse_float_list(args.rho_values),
    ):
        idx = np.arange(1, d + 1, dtype=np.float64)
        lam = idx ** (-a)
        lam /= lam[0]

        mu_aligned = lam / np.sqrt(lam + rho)
        flat_scalar = 1.0 / math.sqrt(float(np.mean(lam)) + rho)
        mu_flat = flat_scalar * lam

        row: dict[str, float | str] = {
            "dimension": d,
            "a": a,
            "rho": rho,
            "original_slope_obs": fit_power_slope(lam),
            "original_slope_pred": -a,
            "aligned_slope_obs": fit_power_slope(mu_aligned),
            "aligned_slope_pred": -a / 2.0,
            "flat_slope_obs": fit_power_slope(mu_flat),
            "flat_slope_pred": -a,
            "random_slope_obs": float("nan"),
            "random_slope_pred": -a,
            "random_diag_condition": float("nan"),
            "random_diag_slope": float("nan"),
        }

        if args.include_random:
            q = random_orthogonal(rng, d)
            sigma = (q.T * lam) @ q
            diag_sigma = np.diag(sigma).copy()
            p = 1.0 / np.sqrt(diag_sigma + rho)
            sqrtp = np.sqrt(p)
            eff = (sqrtp[:, None] * sigma) * sqrtp[None, :]
            eig_eff = np.linalg.eigvalsh(eff)[::-1]
            row["random_slope_obs"] = fit_power_slope(eig_eff)
            row["random_diag_condition"] = float(diag_sigma.max() / diag_sigma.min())
            row["random_diag_slope"] = fit_power_slope(diag_sigma)

        rows.append(row)
        print(
            "alignment",
            f"d={d}",
            f"a={a}",
            f"rho={rho:g}",
            f"aligned={row['aligned_slope_obs']:.3f}/{row['aligned_slope_pred']:.3f}",
            f"flat={row['flat_slope_obs']:.3f}/{row['flat_slope_pred']:.3f}",
            f"random={row['random_slope_obs']:.3f}",
        )

    csv_path = outdir / "alignment_sweep.csv"
    write_csv(csv_path, rows)
    maybe_plot_alignment_summary(outdir, rows, args.no_plots)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_filter_summary(outdir: Path, rows: list[dict[str, float | str]], no_plots: bool) -> None:
    if no_plots or plt is None or not rows:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("sgd_count_slope", "SGD count"),
        ("adam_count_slope", "Adam/RMSProp count"),
        ("adamw_count_slope", "AdamW count"),
        ("sgd_bias_slope", "SGD bias"),
        ("adam_bias_slope", "Adam/RMSProp bias"),
        ("adamw_bias_slope", "AdamW bias"),
    ]
    for stem, title in metrics:
        obs = np.array([float(r[f"{stem}_obs"]) for r in rows], dtype=float)
        pred = np.array([float(r[f"{stem}_pred"]) for r in rows], dtype=float)
        mask = np.isfinite(obs) & np.isfinite(pred)
        if mask.sum() == 0:
            continue
        fig = plt.figure()
        plt.scatter(pred[mask], obs[mask])
        lo = float(min(pred[mask].min(), obs[mask].min()))
        hi = float(max(pred[mask].max(), obs[mask].max()))
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel("predicted slope")
        plt.ylabel("observed slope")
        plt.title(title)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{stem}_observed_vs_predicted.png", dpi=160)
        plt.close(fig)


def maybe_plot_alignment_summary(outdir: Path, rows: list[dict[str, float | str]], no_plots: bool) -> None:
    if no_plots or plt is None or not rows:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = ["original", "aligned", "flat", "random"]
    fig = plt.figure()
    xs: list[float] = []
    ys: list[float] = []
    tick_labels: list[str] = []
    for j, label in enumerate(labels):
        vals = np.array([float(r.get(f"{label}_slope_obs", float("nan"))) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        xs.extend([j] * len(vals))
        ys.extend(vals.tolist())
        tick_labels.append(label)
    if xs:
        plt.scatter(xs, ys)
        plt.xticks(range(len(labels)), labels)
        plt.ylabel("fitted eigenvalue slope")
        plt.title("Coordinate alignment slopes")
        plt.tight_layout()
        fig.savefig(plot_dir / "alignment_slopes.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["filters", "alignment", "all"], default="all")
    parser.add_argument("--run-name", type=str, default="", help="Output subdirectory name under experiments/results.")
    parser.add_argument("--outdir", type=str, default="", help="Explicit output directory. Overrides --run-name.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller, faster sweep.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dimension", type=int, default=200_000)
    parser.add_argument("--a-values", type=str, default="1.25,1.5,2.0")
    parser.add_argument("--b-values", type=str, default="1.2,1.4,1.8")
    parser.add_argument("--rho-values", type=str, default="1e-16,1e-12,1e-8")
    parser.add_argument("--wd-values", type=str, default="0,1e-4,1e-3,1e-2")
    parser.add_argument("--n-min", type=float, default=10.0)
    parser.add_argument("--n-max", type=float, default=10_000.0)
    parser.add_argument("--num-n", type=int, default=50)

    parser.add_argument("--alignment-dimensions", type=str, default="512")
    parser.add_argument("--include-random", action="store_true", help="Run expensive QR/eigendecomp random-rotation diagnostics.")

    args = parser.parse_args()

    if args.quick:
        args.dimension = min(args.dimension, 50_000)
        args.a_values = "1.5"
        args.b_values = "1.4"
        args.rho_values = "1e-12"
        args.wd_values = "0,1e-2"
        args.n_min = 10.0
        args.n_max = 1000.0
        args.num_n = 24
        args.alignment_dimensions = "256"
        args.include_random = True

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        run_name = args.run_name or ("quick" if args.quick else "sweep")
        outdir = Path("experiments") / "results" / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    with (outdir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    all_rows: dict[str, int] = {}
    if args.mode in {"filters", "all"}:
        rows = run_filter_sweeps(args, outdir)
        all_rows["filter_rows"] = len(rows)
    if args.mode in {"alignment", "all"}:
        rows = run_alignment_sweeps(args, outdir)
        all_rows["alignment_rows"] = len(rows)

    with (outdir / "summary.json").open("w") as f:
        json.dump(all_rows, f, indent=2, sort_keys=True)

    print("")
    print(f"Wrote results to: {outdir}")
    for path in sorted(outdir.glob("*.csv")):
        print(f"  {path}")
    if (outdir / "plots").exists():
        print(f"  {outdir / 'plots'}")


if __name__ == "__main__":
    main()
