# Optimizer-dependent scaling laws in linear regression

This repository is a working research notebook for extending the NeurIPS 2024 linear-regression scaling-law theory from vanilla one-pass SGD to Adam/RMSProp-like adaptive preconditioning.

Current proof notes:

- `notes/01_fixed_spectral_preconditioning.md`: fixed spectral preconditioning baseline, `P_q = H^{-q}`.
- `notes/02_damped_spectral_preconditioning.md`: damped spectral preconditioning, `P_{rho,q}=(H+rho I)^{-q}`, including the two-slope effective spectrum, bias/variance spectral sums, and Adam/RMSProp `q=1/2` corollary.
- `notes/03_frozen_rmsprop_bridge.md`: frozen-RMSProp bridge theorem. It proves that, in diagonal Gaussian linear regression, coordinate gradient second moments satisfy `E[g_i^2] \asymp c_e lambda_i`, so a frozen RMSProp preconditioner is spectrally equivalent to `(H+rho I)^(-1/2)` with `rho = epsilon / c_e`.
- `notes/04_online_rmsprop_tracking.md`: online RMSProp spectral-tracking theorem. It proves that the online second-moment EMA tracks `d_t lambda_i`, where `d_t` is the EMA of `sigma^2 + ||w_t-w*||_H^2`, so RMSProp tracks a time-varying damped `q=1/2` preconditioner with `rho_t = epsilon / d_t`.

Experiments:

- `experiments/synthetic_damped_preconditioning.py`: oracle damped spectral-preconditioning sanity check.
- `experiments/frozen_rmsprop_bridge.py`: verifies the gradient-second-moment bridge `v_i \propto lambda_i` and compares frozen RMSProp to the oracle `q=1/2` preconditioner.
- `experiments/online_rmsprop_tracking.py`: checks that online RMSProp has `slope(log v_t, log lambda) \approx 1`, `q_eff \approx 1/2`, and `v_t \approx d_t lambda` in diagonal Gaussian regression.

Next target: upgrade the online tracking theorem into a full online RMSProp risk theorem with matching lower bounds, first conditional on the raw-EMA tracking event and then with a direct high-probability proof for raw RMSProp. After that, extend to Adam momentum, AdamW decay, and coordinate misalignment.