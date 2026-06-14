# Optimizer-dependent scaling laws in linear regression

This repository is a working research notebook for extending the NeurIPS 2024 linear-regression scaling-law theory from vanilla one-pass SGD to Adam/RMSProp-like adaptive preconditioning.

Current proof notes:

- `notes/01_fixed_spectral_preconditioning.md`: fixed spectral preconditioning baseline, `P_q = H^{-q}`.
- `notes/02_damped_spectral_preconditioning.md`: damped spectral preconditioning, `P_{rho,q}=(H+rho I)^{-q}`, including the two-slope effective spectrum, bias/variance spectral sums, and Adam/RMSProp `q=1/2` corollary.
- `notes/03_frozen_rmsprop_bridge.md`: frozen-RMSProp bridge theorem.  It proves that, in diagonal Gaussian linear regression, coordinate gradient second moments satisfy `E[g_i^2] \asymp c_e lambda_i`, so a frozen RMSProp preconditioner is spectrally equivalent to `(H+rho I)^(-1/2)` with `rho = epsilon / c_e`.

Experiments:

- `experiments/synthetic_damped_preconditioning.py`: oracle damped spectral-preconditioning sanity check.
- `experiments/frozen_rmsprop_bridge.py`: verifies the gradient-second-moment bridge `v_i \propto lambda_i` and compares frozen RMSProp to the oracle `q=1/2` preconditioner.

Next target: prove an online RMSProp theorem by showing that the EMA preconditioner tracks a time-varying damped spectral preconditioner `(H+rho_t I)^(-1/2)`, where `rho_t = epsilon / (sigma^2 + ||w_t-w*||_H^2)`.
