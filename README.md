# Optimizer-dependent scaling laws in linear regression

This repository is a working research notebook for extending the NeurIPS 2024 linear-regression scaling-law theory from vanilla one-pass SGD to Adam/RMSProp-like adaptive preconditioning.

## Current proof notes

- `notes/01_fixed_spectral_preconditioning.md`: fixed spectral preconditioning baseline, `P_q = H^{-q}`.
- `notes/02_damped_spectral_preconditioning.md`: damped spectral preconditioning, `P_{rho,q}=(H+rho I)^{-q}`, including the two-slope effective spectrum, bias/variance spectral sums, and Adam/RMSProp `q=1/2` corollary.
- `notes/03_frozen_rmsprop_bridge.md`: frozen-RMSProp bridge theorem. It proves that, in diagonal Gaussian linear regression, coordinate gradient second moments satisfy `E[g_i^2] \asymp c_e lambda_i`, so a frozen RMSProp preconditioner is spectrally equivalent to `(H+rho I)^(-1/2)` with `rho = epsilon / c_e`.
- `notes/04_online_rmsprop_tracking.md`: online RMSProp spectral-tracking theorem. It proves that the online second-moment EMA tracks `d_t lambda_i`, where `d_t` is the EMA of `sigma^2 + ||w_t-w*||_H^2`, so RMSProp tracks a time-varying damped `q=1/2` preconditioner with `rho_t = epsilon / d_t`.
- `notes/05_online_rmsprop_sandwich.md`: conditional online RMSProp sandwich theorem. If online second moments track their conditional means and the residual scale stays in a constant band, online RMSProp has the same scaling filters as frozen damped `q=1/2` spectral preconditioning. A robust decoupled online estimator is proved to satisfy the tracking event.
- `notes/06_raw_ema_tracking.md`: raw-EMA tracking theorem for ordinary RMSProp. It proves that the unmodified EMA tracks `d_t lambda_i` with high probability under an effective-window/leverage condition, despite product-Gaussian squared-gradient tails.
- `notes/07_adam_momentum_filter.md`: Adam first-moment momentum theorem. It proves that `beta1` momentum is a temporal filter: it changes constants and stability margins but leaves the spectral learned-mode count and the emergent `q_eff=1/2` second-moment preconditioning law unchanged.
- `notes/08_adamw_weight_decay_filter.md`: AdamW decoupled-weight-decay theorem. It proves that AdamW equals Adam's damped `q_eff=1/2` preconditioner plus a separate shrinkage filter.
- `notes/09_matching_filter_lower_bounds.md`: matching spectral lower-bound theorem. It proves that the bias and variance filters used throughout the notes are sharp for two-slope power-law spectra.
- `notes/10_coordinate_alignment.md`: coordinate-alignment theorem. It proves that diagonal adaptive methods are spectral only when coordinatewise gradient second moments retain spectral information.
- `notes/11_bandlimited_partial_alignment.md`: band-limited partial-alignment theorem. It proves that arbitrary rotations within comparable-eigenvalue bands preserve the Adam/RMSProp `q_eff=1/2` exponent up to constants.
- `notes/12_smooth_source_bias_saturation.md`: smooth-source bias saturation theorem. It explains out-of-regime bias-slope deviations in the sweeps.
- `notes/13_gaussian_sketch_global_mixing.md`: Gaussian-sketch global-mixing theorem. It proves that high-effective-rank global Gaussian sketches flatten coordinate variances, making diagonal RMSProp/Adam approximately scalar at exponent level unless the sketch is spectrally aligned or band-limited.
- `notes/14_bandlimited_gaussian_features.md`: band-limited Gaussian-feature theorem. It proves that Gaussian random features within comparable-eigenvalue bands preserve Adam/RMSProp's `q_eff=1/2` spectral gains with high probability.
- `notes/15_profile_qeff_diagnostics.md`: profile-exponent diagnostic theorem. It proves that if coordinate second moments scale as `lambda_i^theta`, then the effective exponent is `q_eff=theta/2`.
- `notes/16_bounded_preconditioner_no_exponent_change.md`: bounded-preconditioner theorem. It proves that bounded or sub-polynomial condition-number diagonal preconditioners cannot change power-law spectral exponents.
- `notes/16_source_condition_stability.md`: source-condition stability theorem. It proves that spectral preconditioners preserve source energies exactly, and invariant band decompositions preserve block source mass, justifying the transformed-source assumption in aligned and band-limited settings.
- `notes/17_compute_optimal_visible_spectrum.md`: compute-optimal visible-spectrum theorem. It derives the optimal allocation `M_*(C)`, `N_*(C)`, and compute-risk exponent as a function of the visible exponent `theta`.
- `notes/18_weight_decay_compute_ceiling.md`: AdamW compute-ceiling theorem. It proves that fixed decoupled weight decay imposes a finite effective-horizon ceiling and that weight decay must scale down with compute to preserve the no-decay exponent.

## Manuscript draft materials

- `manuscript/theorem_stack.md`: consolidated theorem stack and current main conclusion for a paper draft.

## Experiments

- `experiments/README.md`: local experiment-running instructions and recommended commands.
- `experiments/run_sweeps.py`: main local sweep runner for deterministic exponent-level filters and coordinate-alignment experiments.
- `experiments/bandlimited_feature_alignment.py`: compares aligned, band-limited Gaussian, and global Gaussian feature maps.
- `experiments/profile_feature_sweep.py`: measures coordinate-variance profile exponents and checks the prediction `q_eff≈theta/2` across synthetic, aligned, flat, band-limited, and global feature maps.
- `experiments/sketch_visibility_sweep.py`: measures effective preconditioned spectra for aligned, flat, profile, band-limited, Haar, and Gaussian-sketch feature systems.
- `experiments/stochastic_training_scaling.py`: runs actual stochastic mini-batch training for SGD, oracle preconditioning, RMSProp, Adam, and AdamW in diagonal Gaussian regression.
- `experiments/stochastic_alignment_scaling.py`: runs actual stochastic mini-batch training across aligned, band-limited, flat, and Haar coordinate systems.
- `experiments/stochastic_alignment_lr_grid.py`: runs a learning-rate grid for stochastic alignment experiments, so risk comparisons are not dominated by a single hand-picked step size.
- `experiments/source_condition_diagnostic.py`: measures transformed-source exponents after applying coordinatewise preconditioners.
- `experiments/compute_optimal_visible_spectrum.py`: optimizes the spectral risk proxy over `M,N` under compute `C≈MN` and checks the visible-spectrum compute-optimal phase diagram.
- `experiments/weight_decay_compute_sweep.py`: tests the AdamW prediction that fixed weight decay creates a compute horizon ceiling while compute-scaled weight decay preserves the no-decay exponent.
- `experiments/synthetic_damped_preconditioning.py`: oracle damped spectral-preconditioning sanity check.
- `experiments/frozen_rmsprop_bridge.py`: verifies the gradient-second-moment bridge `v_i \propto lambda_i` and compares frozen RMSProp to the oracle `q=1/2` preconditioner.
- `experiments/online_rmsprop_tracking.py`: checks that online RMSProp has `slope(log v_t, log lambda) \approx 1`, `q_eff \approx 1/2`, and `v_t \approx d_t lambda` in diagonal Gaussian regression.
- `experiments/raw_ema_tracking.py`: isolates the raw EMA concentration mechanism with product-Gaussian squared-gradient noise and verifies that `v_t \approx d_t lambda` and `q_eff \approx 1/2`.
- `experiments/adam_momentum_filter.py`: verifies that Adam `beta1` momentum leaves the learned-mode cutoff `N gamma mu_i \asymp 1` unchanged up to constants.
- `experiments/adamw_weight_decay_filter.py`: verifies that AdamW decoupled weight decay saturates the learned-mode count.
- `experiments/exponent_level_filters.py`: deterministic exponent-level filter experiment.
- `experiments/coordinate_alignment.py`: tests the coordinate-alignment theorem.

## Results

- `experiments/results/analysis_2026_06_15.md`: analysis of the first filter and alignment sweeps.
- `experiments/results/analysis_after_second_batch.md`: analysis of the stochastic-training and sketch-visibility sweeps.
- `experiments/results/analysis_after_stochastic_alignment.md`: analysis after stochastic coordinate-alignment experiments, with recommended next runs.
- `experiments/results/COMPUTE_OPTIMAL_VISIBLE_REVIEW.md`: analysis of the compute-optimal visible-spectrum results and recommended follow-up runs.

## Quick sanity checks

```bash
python experiments/frozen_rmsprop_bridge.py
python experiments/online_rmsprop_tracking.py
python experiments/raw_ema_tracking.py
python experiments/adam_momentum_filter.py
python experiments/adamw_weight_decay_filter.py
python experiments/exponent_level_filters.py --quick
python experiments/coordinate_alignment.py --dimension 512
python experiments/run_sweeps.py --quick --mode all
python experiments/bandlimited_feature_alignment.py --dimension 2048
python experiments/profile_feature_sweep.py --quick
python experiments/sketch_visibility_sweep.py --quick
python experiments/stochastic_training_scaling.py --quick
python experiments/stochastic_alignment_scaling.py --quick
python experiments/stochastic_alignment_lr_grid.py --quick
python experiments/source_condition_diagnostic.py --quick
python experiments/compute_optimal_visible_spectrum.py --quick
python experiments/weight_decay_compute_sweep.py --quick
```

## Current status

The theory and experiments now support the central mechanism:

```text
coordinatewise second moments expose spectral structure
  -> q_eff in [0, 1/2]
  -> visible-spectrum learned-mode count
  -> optimizer-dependent scaling filters
  -> compute-optimal allocation and compute-risk exponent
```

Aligned and band-limited coordinates preserve `q_eff≈1/2`; flat, Haar, and global Gaussian-sketch coordinates behave like `q_eff≈0`. Learning-rate calibrated stochastic experiments support the same diagnostic. The newest compute-optimal experiment supports the visible-spectrum allocation law and clarifies the hard-source versus variance-limited phases. The remaining empirical gaps for a high-impact paper are AdamW compute-ceiling sweeps and larger random-feature/source-condition diagnostics.
