# Next experiments after current results

The current results support the deterministic filter theory, the coordinate
alignment theorem, and the sketch visibility story.  The next experiments should
focus on actual stochastic training under coordinate changes.

## 1. Stochastic coordinate-alignment training

Run:

```bash
python experiments/stochastic_alignment_scaling.py \
  --dimension 512 \
  --cases aligned,band,flat,haar \
  --optimizers sgd,coord_oracle,rmsprop,adam,adamw,spectral_oracle \
  --checkpoints 250,500,1000,2000,4000,8000 \
  --trials 5 \
  --batch-size 16 \
  --outdir experiments/results/stochastic_alignment_main
```

Expected pattern:

- aligned and band-limited cases: RMSProp/Adam q_eff diagnostic near 1/2;
- flat and Haar/global cases: coordinatewise RMSProp/Adam q_eff diagnostic near 0;
- spectral_oracle remains near 1/2 in every basis;
- SGD remains q_eff=0.

This is the most important next experiment because it checks that the alignment
mechanism affects real training curves, not only deterministic spectra.

## 2. Learning-rate calibration for stochastic scaling

The current stochastic run shows q_eff≈1/2 for RMSProp/Adam/AdamW and much
steeper risk decay than SGD.  However, the risk slopes are likely in a transient
constant-step regime.  To compare to theorem exponents, repeat stochastic runs
with a grid of learning rates and fit only stable non-saturated windows.

Suggested adaptive learning rates:

```text
0.001,0.002,0.005,0.01
```

Suggested SGD learning rates:

```text
0.02,0.05,0.1
```

## 3. Structured random features

After stochastic alignment, add feature maps whose coordinate variance profile
has an intermediate exponent theta.  The prediction from the visible-spectrum
theorem is

```text
q_eff ≈ theta / 2.
```

This is the most direct path to a high-impact result beyond diagonal models.
