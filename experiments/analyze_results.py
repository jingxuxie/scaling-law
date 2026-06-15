"""Analyze sweep CSVs for optimizer-dependent scaling-law experiments.

Usage
-----
python experiments/analyze_results.py \
  --filter-csv experiments/results/filter_main/filter_sweep.csv \
  --alignment-csv experiments/results/alignment_main/alignment_sweep.csv

The script prints aggregate error statistics and flags rows where apparent
mismatch is expected because the row is outside the clean theorem range or the
n-range hits finite-dimensional saturation.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


def f(x: str) -> float:
    if x == "inf":
        return float("inf")
    return float(x)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def summarize_errors(rows: list[dict[str, str]], metric_prefixes: list[str]) -> None:
    for prefix in metric_prefixes:
        key = f"{prefix}_abs_error"
        vals = [f(r[key]) for r in rows if key in r and math.isfinite(f(r[key]))]
        if not vals:
            continue
        print(
            f"{prefix:28s} mean={mean(vals):.4f} median={median(vals):.4f} "
            f"max={max(vals):.4f} n={len(vals)}"
        )


def adam_theory_count(n: float, a: float, rho: float) -> float:
    if rho <= 0:
        return n ** (2.0 / a)
    knee = rho ** (-0.5)
    if n <= knee:
        return n ** (2.0 / a)
    return rho ** (-1.0 / (2.0 * a)) * n ** (1.0 / a)


def analyze_filter(path: Path) -> None:
    rows = read_csv(path)
    print("\n== Filter sweep ==")
    print(f"path={path}")
    print(f"rows={len(rows)}")
    summarize_errors(
        rows,
        [
            "sgd_count_slope",
            "adam_count_slope",
            "adamw_count_slope",
            "sgd_bias_slope",
            "adam_bias_slope",
            "adamw_bias_slope",
        ],
    )

    print("\nRows with large Adam bias error and likely reason:")
    flagged = []
    for r in rows:
        a = f(r["a"])
        b = f(r["b"])
        rho = f(r["rho"])
        dim = f(r["dimension"])
        nmax = f(r["n_max"])
        err = f(r["adam_bias_slope_abs_error"])
        alpha = a / 2.0
        kmax = adam_theory_count(nmax, a, rho)
        reasons = []
        if b >= alpha + 1.0:
            reasons.append("outside clean Adam-bias theorem: b >= a/2 + 1")
        if b <= 1.25:
            reasons.append("slow source-tail convergence: b close to 1")
        if kmax > 0.25 * dim:
            reasons.append("finite-dimensional saturation: K(n_max) not << dimension")
        if err > 0.10:
            flagged.append((err, r, reasons))
    flagged.sort(reverse=True, key=lambda x: x[0])
    for err, r, reasons in flagged[:12]:
        print(
            f"err={err:.3f} a={r['a']} b={r['b']} rho={r['rho']} wd={r['wd']} "
            f"obs={f(r['adam_bias_slope_obs']):.3f} pred={f(r['adam_bias_slope_pred']):.3f} "
            f"reason={'; '.join(reasons) or 'needs inspection'}"
        )

    clean = []
    for r in rows:
        a = f(r["a"])
        b = f(r["b"])
        rho = f(r["rho"])
        dim = f(r["dimension"])
        nmax = f(r["n_max"])
        alpha = a / 2.0
        kmax = adam_theory_count(nmax, a, rho)
        if (1.25 < b < alpha + 1.0) and (kmax < 0.25 * dim):
            clean.append(r)
    if clean:
        vals = [f(r["adam_bias_slope_abs_error"]) for r in clean]
        print(
            "\nClean Adam-bias subset "
            f"(1.25<b<a/2+1 and K(n_max)<0.25M): n={len(clean)} "
            f"mean_err={mean(vals):.4f} median_err={median(vals):.4f} max_err={max(vals):.4f}"
        )
    else:
        print("\nNo rows met the strict clean Adam-bias subset criterion.")


def analyze_alignment(path: Path) -> None:
    rows = read_csv(path)
    print("\n== Alignment sweep ==")
    print(f"path={path}")
    print(f"rows={len(rows)}")
    for prefix in ["original", "aligned", "flat", "random"]:
        obs_key = f"{prefix}_slope_obs"
        pred_key = f"{prefix}_slope_pred"
        vals = [abs(f(r[obs_key]) - f(r[pred_key])) for r in rows if obs_key in r]
        if vals:
            print(
                f"{prefix:10s} slope_abs_error mean={mean(vals):.4f} "
                f"median={median(vals):.4f} max={max(vals):.4f}"
            )
    print("\nRandom-rotation diagnostics:")
    for r in rows:
        if "random_diag_condition" in r:
            print(
                f"d={r['dimension']} a={r['a']} rho={r['rho']} "
                f"random_slope={f(r['random_slope_obs']):.3f} pred={f(r['random_slope_pred']):.3f} "
                f"diag_cond={f(r['random_diag_condition']):.2f} diag_slope={f(r['random_diag_slope']):.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-csv", type=Path, default=Path("experiments/results/filter_main/filter_sweep.csv"))
    parser.add_argument("--alignment-csv", type=Path, default=Path("experiments/results/alignment_main/alignment_sweep.csv"))
    args = parser.parse_args()
    if args.filter_csv.exists():
        analyze_filter(args.filter_csv)
    else:
        print(f"missing filter CSV: {args.filter_csv}")
    if args.alignment_csv.exists():
        analyze_alignment(args.alignment_csv)
    else:
        print(f"missing alignment CSV: {args.alignment_csv}")


if __name__ == "__main__":
    main()
