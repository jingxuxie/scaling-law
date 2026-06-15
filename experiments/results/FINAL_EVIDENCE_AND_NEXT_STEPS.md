# Evidence status and remaining experiments

The latest results support the main mechanism:

```text
visible spectral information -> q_eff -> learned-mode count -> scaling law
```

## Supported by current results

1. **Theta-profile stochastic grid.**  Best tuned final risk improves monotonically as the visible exponent theta increases.  The run also records the predicted continuum

```text
theta=0    -> q_eff=0
theta=0.25 -> q_eff=0.125
theta=0.5  -> q_eff=0.25
theta=0.75 -> q_eff=0.375
theta=1    -> q_eff=0.5
```

2. **Source diagnostics.**  Aligned and flat cases preserve the source exponent essentially exactly.  Band-limited cases preserve total source mass exactly and keep the transformed source exponent close to the target, up to finite-band fluctuations.  Haar/global mixing has more source-exponent variability, so global-mixing risk curves should be interpreted with both spectrum and source diagnostics.

3. **LR-calibrated stochastic alignment.**  After tuning learning rates, aligned and band-limited coordinate systems still have q_eff about 1/2, while flat/Haar coordinate systems have q_eff about zero.  Thus learning-rate tuning improves constants but does not change the visible spectral exponent.

## Remaining high-impact experiment

Run the compute-optimal visible-spectrum experiment:

```bash
python experiments/compute_optimal_scaling.py --quick
```

Then run the main deterministic scaling-law experiment:

```bash
python experiments/compute_optimal_scaling.py \
  --a 1.5 \
  --b 1.4 \
  --theta-values 0,0.25,0.5,0.75,1 \
  --c-min 1e4 \
  --c-max 1e10 \
  --num-c 40 \
  --outdir experiments/results/compute_optimal_visible
```

This verifies the Chinchilla-style allocation law:

```text
M_* ~ C^{1/(max(alpha,b)+1)}
N_* ~ C^{max(alpha,b)/(max(alpha,b)+1)}
R_* ~ C^{-(b-1)/(max(alpha,b)+1)}
alpha = a(1 - theta/2)
```

If this passes, the empirical package is enough for a first strong paper draft.  The remaining work is manuscript organization and, optionally, a more realistic random-feature experiment.
