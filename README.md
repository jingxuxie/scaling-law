# Optimizer-dependent scaling laws in linear regression

This repository is a working research notebook for extending the NeurIPS 2024 linear-regression scaling-law theory from vanilla one-pass SGD to Adam/RMSProp-like adaptive preconditioning.

## Current proof notes

- `notes/01_fixed_spectral_preconditioning.md`: fixed spectral preconditioning baseline, `P_q = H^{-q}`.
- `notes/02_damped_spectral_preconditioning.md`: damped spectral preconditioning, `P_{rho,q}=(H+rho I)^{-q}`, including the two-slope effective spectrum, bias/variance spectral sums, and Adam/RMSProp `q=1/2` corollary.
- `notes/03_frozen_rmsprop_bridge.md`: frozen-RMSProp bridge theorem. It proves that, in diagonal Gaussian linear regression, coordinate gradient second moments satisfy `E[g_i^2] \asymp c_e lambda_i`, so a frozen RMSProp preconditioner is spectrally equivalent to `(H+rho I)^(-1/2)` with `rho = epsilon / c_e`.
- `notes/04_online_rmsprop_tracking.md`: online RMSProp spectral-tracking theorem. It proves that the online second-moment EMA tracks `d_t lambda_i`, where `d_t` is the EMA of `sigma^2 + ||w_t-w*||_H^2`, so RMSProp tracks a time-varying damped `q=1/2` preconditioner with `rho_t = epsilon / d_t`.
- `notes/05_online_rmsprop_sandwich.md`: conditional online RMSProp sandwich theorem. If online second moments track their conditional means and the residual scale stays in a constant band, online RMSProp has the same scaling filters as frozen damped `q=1/2` spectral preconditioning. A robust decoupled online estimator is proved to satisfy the tracking event.
- `notes/06_raw_ema_tracking.md`: raw-EMA tracking theorem for ordinary RMSProp. It proves that the unmodified EMA tracks `d_t lambda_i` with high probability under an effective-window/leverage condition, despite product-Gaussian squared-gradient tails. This removes the main tracking assumption from the sandwich theorem and again yields the damped `q=1/2` scaling law.
- `notes/07_adam_momentum_filter.md`: Adam first-moment momentum theorem. It proves that `beta1` momentum is a temporal filter: it changes constants and stability margins but leaves the spectral learned-mode count and the emergent `q_eff=1/2` second-moment preconditioning law unchanged.
- `notes/08_adamw_weight_decay_filter.md`: AdamW decoupled-weight-decay theorem. It proves that AdamW equals Adam's damped `q_eff=1/2` preconditioner plus a separate shrinkage filter, with active cutoff `mu_i >= max(1/n, lambda_wd)` and effective horizon `T_eff = min(n, lambda_wd^{-1})` after scalar normalization.
- `notes/09_matching_filter_lower_bounds.md`: matching spectral lower-bound theorem. It proves that the bias and variance filters used throughout the notes are sharp for two-slope power-law spectra, including Adam/RMSProp and AdamW learned-mode counts.
- `notes/10_coordinate_alignment.md`: coordinate-alignment theorem. It proves that diagonal adaptive methods are spectral only when coordinatewise gradient second moments retain spectral information; aligned coordinates give `q_eff=1/2`, while flat/bounded coordinate variances make RMSProp/Adam essentially scalar at exponent level.

## Manuscript draft materials

- `manuscript/theorem_stack.md`: consolidated theorem stack and current main conclusion for a paper draft.

## Experiments

- `experiments/synthetic_damped_preconditioning.py`: oracle damped spectral-preconditioning sanity check.
- `experiments/frozen_rmsprop_bridge.py`: verifies the gradient-second-moment bridge `v_i \propto lambda_i` and compares frozen RMSProp to the oracle `q=1/2` preconditioner.
- `experiments/online_rmsprop_tracking.py`: checks that online RMSProp has `slope(log v_t, log lambda) \approx 1`, `q_eff \approx 1/2`, and `v_t \approx d_t lambda` in diagonal Gaussian regression.
- `experiments/raw_ema_tracking.py`: isolates the raw EMA concentration mechanism with product-Gaussian squared-gradient noise and verifies that `v_t \approx d_t lambda` and `q_eff \approx 1/2`.
- `experiments/adam_momentum_filter.py`: verifies that Adam `beta1` momentum leaves the learned-mode cutoff `N gamma mu_i \asymp 1` unchanged up to constants for a fixed damped `q=1/2` preconditioner.
- `experiments/adamw_weight_decay_filter.py`: verifies that AdamW decoupled weight decay saturates the learned-mode count at the cutoff `mu_i >= max(1/(N gamma), lambda_wd)`.
- `experiments/exponent_level_filters.py`: deterministic exponent-level filter experiment. It fits log-log slopes for learned counts and bias filters for SGD, Adam/RMSProp, and AdamW.
- `experiments/coordinate_alignment.py`: tests the coordinate-alignment theorem by comparing aligned, flat Hadamard, and random-orthogonal optimizer coordinates.

## Quick sanity checks

```bash
python experiments/frozen_rmsprop_bridge.py
python experiments/online_rmsprop_tracking.py
python experiments/raw_ema_tracking.py
python experiments/adam_momentum_filter.py
python experiments/adamw_weight_decay_filter.py
python experiments/exponent_level_filters.py --quick
python experiments/coordinate_alignment.py --dimension 512
```

Stored quick outputs:

- `experiments/results/quick_exponent_level_filters.txt`
- `experiments/results/quick_coordinate_alignment.txt`

## Current status

The diagonal Gaussian optimizer-theory chain is complete at the level of spectral upper and lower bounds:

```text
RMSProp / Adam / AdamW
  -> gradient second moments track lambda_i in aligned coordinates
  -> q_eff = 1/2 damped spectral preconditioning
  -> learned-mode count K_{rho,1/2}(n)
  -> AdamW replaces n by min(n, lambda_wd^{-1})
  -> sharp bias and variance scaling filters
```

The newest coordinate-alignment theorem shows when this mechanism survives a change of optimizer basis. The next high-impact target is to extend the proof strategy from diagonal Gaussian coordinates to Gaussian-sketched or random-feature regression, where coordinate/eigenvector alignment is partial rather than perfect.
