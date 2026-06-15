"""Sanity check for the online RMSProp spectral-tracking theorem.

The theorem in notes/04_online_rmsprop_tracking.md predicts that online RMSProp's
EMA satisfies

    v_{t,i} \approx cbar_t * lambda_i,

so log(v_t) versus log(lambda) should have slope close to 1. Consequently the
preconditioner A_i=(v_i+eps)^(-1/2) should have active-range log slope close to
-1/2, i.e. q_eff close to 1/2.

The script uses mini-batch RMSProp in diagonal Gaussian linear regression.  The
mini-batch average is not essential for the mechanism; it simply makes the
sanity check less noisy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Config:
    d: int = 512
    steps: int = 1600
    batch_size: int = 256
    a: float = 1.6
    b: float = 1.25
    sigma: float = 0.2
    beta2: float = 0.99
    lr: float = 0.02
    epsilon: float = 1e-8
    seed: int = 0
    eval_every: int = 200
    output_dir: str = "outputs"


def make_problem(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(1, cfg.d + 1, dtype=float)
    lambdas = idx ** (-cfg.a)
    source_energy = idx ** (-cfg.b)  # s_i = lambda_i * (w_i*)^2
    signs = np.where(np.arange(cfg.d) % 2 == 0, 1.0, -1.0)
    w_star = signs * np.sqrt(source_energy / lambdas)
    return lambdas, w_star


def fit_slope(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    xx = np.log(x[mask])
    yy = np.log(y[mask])
    slope, _intercept = np.polyfit(xx, yy, 1)
    return float(slope)


def risk(w: np.ndarray, w_star: np.ndarray, lambdas: np.ndarray, sigma: float) -> float:
    e = w - w_star
    return float(np.sum(lambdas * e * e) + sigma**2)


def run(cfg: Config) -> dict[str, float | str]:
    rng = np.random.default_rng(cfg.seed)
    lambdas, w_star = make_problem(cfg)
    sqrt_lambdas = np.sqrt(lambdas)

    w = np.zeros(cfg.d)
    v = np.zeros(cfg.d)
    cbar = 0.0

    rows: list[tuple[int, float, float, float, float, float]] = []

    for t in range(1, cfg.steps + 1):
        x = rng.normal(size=(cfg.batch_size, cfg.d)) * sqrt_lambdas
        noise = cfg.sigma * rng.normal(size=cfg.batch_size)
        residual = x @ (w - w_star) - noise
        grad_samples = residual[:, None] * x

        grad = grad_samples.mean(axis=0)
        second_moment = (grad_samples * grad_samples).mean(axis=0)

        v = cfg.beta2 * v + (1.0 - cfg.beta2) * second_moment
        c_t = cfg.sigma**2 + float(np.sum(lambdas * (w - w_star) ** 2))
        cbar = cfg.beta2 * cbar + (1.0 - cfg.beta2) * c_t

        preconditioner = 1.0 / np.sqrt(v + cfg.epsilon)
        w = w - cfg.lr * preconditioner * grad

        if t % cfg.eval_every == 0 or t == cfg.steps:
            # Active range: avoid coordinates where epsilon dominates v.
            active = v > 20.0 * cfg.epsilon
            if int(active.sum()) < 20:
                active = np.ones(cfg.d, dtype=bool)

            slope_v = fit_slope(lambdas, v + cfg.epsilon, active)
            slope_A = fit_slope(lambdas, preconditioner, active)
            q_eff = -slope_A
            ratio = np.median(v[active] / (cbar * lambdas[active]))
            rows.append(
                (
                    t,
                    slope_v,
                    slope_A,
                    q_eff,
                    ratio,
                    risk(w, w_star, lambdas, cfg.sigma),
                )
            )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "online_rmsprop_tracking.csv"
    header = "step,slope_log_v_vs_log_lambda,slope_log_A_vs_log_lambda,q_eff,median_v_over_cbar_lambda,risk\n"
    np.savetxt(csv_path, np.array(rows), delimiter=",", header=header.strip(), comments="")

    last = rows[-1]
    return {
        "final_step": last[0],
        "slope_log_v_vs_log_lambda": last[1],
        "slope_log_A_vs_log_lambda": last[2],
        "q_eff": last[3],
        "median_v_over_cbar_lambda": last[4],
        "risk": last[5],
        "csv_path": str(csv_path),
    }


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=Config.d)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--a", type=float, default=Config.a)
    parser.add_argument("--b", type=float, default=Config.b)
    parser.add_argument("--sigma", type=float, default=Config.sigma)
    parser.add_argument("--beta2", type=float, default=Config.beta2)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--epsilon", type=float, default=Config.epsilon)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--eval-every", type=int, default=Config.eval_every)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    args = parser.parse_args()
    return Config(
        d=args.d,
        steps=args.steps,
        batch_size=args.batch_size,
        a=args.a,
        b=args.b,
        sigma=args.sigma,
        beta2=args.beta2,
        lr=args.lr,
        epsilon=args.epsilon,
        seed=args.seed,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = run(cfg)
    for key, value in result.items():
        print(f"{key}: {value}")
