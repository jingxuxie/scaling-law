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

## Experiments

- `experiments/synthetic_damped_preconditioning.py`: oracle damped spectral-preconditioning sanity check.
- `experiments/frozen_rmsprop_bridge.py`: verifies the gradient-second-moment bridge `v_i \propto lambda_i` and compares frozen RMSProp to the oracle `q=1/2` preconditioner.
- `experiments/online_rmsprop_tracking.py`: checks that online RMSProp has `slope(log v_t, log lambda) \approx 1`, `q_eff \approx 1/2`, and `v_t \approx d_t lambda` in diagonal Gaussian regression.
- `experiments/raw_ema_tracking.py`: isolates the raw EMA concentration mechanism with product-Gaussian squared-gradient noise and verifies that `v_t \approx d_t lambda` and `q_eff \approx 1/2`.
- `experiments/adam_momentum_filter.py`: verifies that Adam `beta1` momentum leaves the learned-mode cutoff `N gamma mu_i \asymp 1` unchanged up to constants for a fixed damped `q=1/2` preconditioner.

## Quick sanity checks

```bash
python experiments/frozen_rmsprop_bridge.py
python experiments/online_rmsprop_tracking.py
python experiments/raw_ema_tracking.py
python experiments/adam_momentum_filter.py
```

Expected Adam-momentum sanity output is roughly:

```text
predicted_cutoff_K={N gamma mu >= 1}=O(10^2-10^3)
beta1=0.00 ratio_half_to_pred=constant
beta1=0.90 ratio_half_to_pred=similar constant
beta1=0.95 ratio_half_to_pred=similar constant
```

## Next target

The next theorem target is AdamW decoupled weight decay. The goal is to treat the update `w <- (1 - gamma * lambda_wd) w - gamma * (v + eps)^(-1/2) m` as Adam's damped `q=1/2` spectral preconditioner plus a separate shrinkage filter, derive the modified bias term, and then optimize the scaling law over `(M, N, gamma, epsilon, lambda_wd)`.
