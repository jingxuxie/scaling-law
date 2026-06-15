# Current results interpretation

This note summarizes the current experimental state after the deterministic filter, sketch visibility, stochastic training, and stochastic alignment runs.

## What is strongly supported

1. **Deterministic filter exponents.**  Learned-mode count slopes for SGD, Adam/RMSProp, and AdamW agree closely with the formulas in the notes.  Adam/RMSProp follows the active pre-damping slope and AdamW saturates at the weight-decay horizon.

2. **Visibility controls q_eff.**  In the sketch-visibility sweep:
   - aligned coordinates give `q_eff_hat` near `0.5`;
   - flat coordinates give `q_eff_hat` near `0`;
   - synthetic diagonal profiles give `q_eff_hat ≈ theta/2`;
   - band-limited rotations give `q_eff_hat` near `0.5`;
   - Haar/global rotations and isotropic Gaussian sketches give `q_eff_hat` near `0`.

3. **Actual optimizers learn the predicted q_eff diagnostic.**  In stochastic alignment training:
   - aligned and band-limited coordinates keep RMSProp/Adam/AdamW around `q_eff≈0.5`;
   - flat and Haar coordinates push RMSProp/Adam/AdamW to `q_eff≈0`;
   - SGD remains `q_eff=0` by construction.

## Important caveat

In flat/Haar stochastic training, Adam/RMSProp can still beat vanilla SGD in finite-time risk even when `q_eff≈0`.  This is not a contradiction.  It means the optimizer improves scalar normalization, stability, or constants, not the spectral scaling exponent.  The theorem in `notes/16_scalar_mixing_no_exponent_improvement.md` formalizes this point.

## Highest-priority next experiment

Run the learning-rate calibrated stochastic alignment sweep:

```bash
python experiments/lr_sweep_stochastic_alignment.py \
  --cases aligned,band,flat,haar \
  --optimizers sgd,rmsprop,adam,adamw,coord_oracle,spectral_oracle \
  --lr-sgd-values 0.01,0.03,0.05,0.1 \
  --lr-adaptive-values 0.001,0.003,0.005,0.01,0.02 \
  --dimension 512 \
  --checkpoints 250,500,1000,2000,4000,8000 \
  --trials 5 \
  --batch-size 16 \
  --outdir experiments/results/stochastic_alignment_lr_sweep
```

This will separate true spectral exponent effects from learning-rate constants.  The key columns will be:

```text
case, optimizer, lr, final_risk_mean, risk_slope_loglog, final_q_eff_mean
```

The main expected pattern is:

```text
aligned/band: RMSProp/Adam/AdamW final_q_eff_mean ≈ 0.5
flat/Haar:    RMSProp/Adam/AdamW final_q_eff_mean ≈ 0
```

After LR tuning, compare best final risk and best slopes within each case.  The paper claim should emphasize `q_eff` and spectral visibility, not only raw risk curves.
