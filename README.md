# Optimizer-dependent scaling laws in linear regression

This repository is a working research notebook for extending the NeurIPS 2024 linear-regression scaling-law theory from vanilla one-pass SGD to Adam/RMSProp-like adaptive preconditioning.

Current proof notes:

- `notes/01_fixed_spectral_preconditioning.md`: fixed spectral preconditioning baseline, `P_q = H^{-q}`.
- `notes/02_damped_spectral_preconditioning.md`: damped spectral preconditioning, `P_{rho,q}=(H+rho I)^{-q}`, including the two-slope effective spectrum, bias/variance spectral sums, and Adam/RMSProp `q=1/2` corollary.

Immediate next step: prove a frozen-RMSProp theorem showing that an empirical second-moment estimator satisfies `v_i + epsilon \asymp c_t lambda_i + epsilon` over the learned coordinates, which reduces frozen RMSProp to the damped `q=1/2` theorem.
