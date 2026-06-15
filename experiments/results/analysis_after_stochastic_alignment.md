# Analysis after stochastic-alignment and sketch-visibility experiments

## Summary

The current results strongly support the mechanism:

1. deterministic filter sweeps validate the predicted learned-mode exponents;
2. sketch-visibility sweeps validate the phase diagram for `q_eff`;
3. stochastic alignment experiments show that real RMSProp/Adam/AdamW second moments have `q_eff≈1/2` in aligned and band-limited coordinates, but `q_eff≈0` in flat/Haar coordinates.

The remaining experimental gap is not whether the second moment learns the predicted shape; it does.  The gap is clean risk-scaling comparison under tuned hyperparameters and transformed-source diagnostics.

## Key evidence

### Sketch visibility

For `diagonal_profile`, the experiment verifies the interpolation law

```text
q_eff ≈ theta / 2.
```

Examples:

```text
theta=0.25 -> q_eff≈0.125
theta=0.50 -> q_eff≈0.250
theta=0.75 -> q_eff≈0.375
theta=1.00 -> q_eff≈0.500
```

Aligned and band-limited features preserve `q_eff≈1/2`.  Haar and Gaussian-sketch coordinates are near `q_eff≈0`.

### Stochastic alignment

Actual mini-batch training confirms the coordinate visibility diagnostic:

```text
aligned: Adam/RMSProp/AdamW q_eff≈0.48--0.51
band:    Adam/RMSProp/AdamW q_eff≈0.48--0.51
flat:    Adam/RMSProp/AdamW q_eff≈0
haar:    Adam/RMSProp/AdamW q_eff≈0 or slightly negative
```

So the core mechanistic claim is now empirically supported:

```text
coordinatewise adaptive methods change exponents only when second moments are spectrally informative.
```

## Caveat

Raw risk slopes in the stochastic experiments are sensitive to learning-rate choices.  In flat/Haar coordinates, adaptive methods still sometimes reduce risk faster than SGD even with `q_eff≈0`; this should be interpreted as a learning-rate/normalization effect, not spectral exponent improvement.

Therefore the next risk experiment should tune learning rates for every optimizer and coordinate case, then compare best final risk and slopes.

## Next experiments

Run:

```bash
python experiments/stochastic_alignment_lr_grid.py --quick
```

Then:

```bash
python experiments/stochastic_alignment_lr_grid.py \
  --cases aligned,band,flat,haar \
  --optimizers sgd,rmsprop,adam,adamw,coord_oracle,spectral_oracle \
  --lr-sgd-values 0.01,0.03,0.05,0.1 \
  --lr-adaptive-values 0.001,0.003,0.005,0.01 \
  --dimension 512 \
  --checkpoints 250,500,1000,2000,4000,8000 \
  --trials 5 \
  --outdir experiments/results/stochastic_alignment_lr_grid
```

Also run the source diagnostic:

```bash
python experiments/source_condition_diagnostic.py --quick
```

Then:

```bash
python experiments/source_condition_diagnostic.py \
  --dimension 512 \
  --cases aligned,band,flat,haar \
  --a-values 1.25,1.5,2.0 \
  --b-values 1.2,1.4,1.8 \
  --outdir experiments/results/source_diagnostic
```

## Theory status

The newly added source-condition theorem proves that source energies are exactly preserved under spectral preconditioners and that block source mass is preserved under invariant band decompositions.  This justifies the transformed source assumption used by the visible-spectrum scaling theorem for aligned and band-limited models.  For globally mixed coordinates, transformed source diagnostics should be reported experimentally.
