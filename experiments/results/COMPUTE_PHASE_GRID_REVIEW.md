# Compute-optimal phase-grid review

The compute-optimal phase-grid results support the visible-spectrum compute law.

## Current saturating regime: a=1.5, b=1.4

The critical visibility threshold is

\[
    \theta_c=2(1-b/a)\approx 0.133.
\]

Therefore only \(\theta=0\) is in the hard-source/time-limited phase.  All
tested \(\theta\ge0.25\) are in the easy-source/variance-limited phase, so the
compute-risk exponent should saturate at

\[
    (b-1)/(b+1)=0.4/2.4\approx0.1667.
\]

The observed risk exponent is approximately \(-0.1667\) for all
\(\theta\ge0.25\), matching the theory.  The allocation exponents also match:
\(M_*(C)\approx C^{0.416}\), \(N_*(C)\approx C^{0.584}\).

## Hard full-range regime: a=3.0, b=1.4

Here \(\theta_c>1\), so the whole tested \(\theta\in[0,1]\) interval remains
time-limited.  The compute-risk exponent should improve monotonically with
\(\theta\).  The observed risk exponent changes from about \(-0.102\) at
\(\theta=0\) to about \(-0.167\) at \(\theta=1\), matching the predicted trend.

## Fixed AdamW weight decay

The fixed-weight-decay runs with \(\delta=10^{-3}\) and \(\delta=10^{-4}\)
intentionally violate the no-weight-decay condition.  They should not match the
no-decay prediction columns.  The observed behavior is qualitatively correct:

1. the learned-mode exponent collapses relative to the no-decay case;
2. the risk exponent becomes much smaller in magnitude;
3. optimal model size grows faster because compute cannot be used to learn more
   modes once the cap is active;
4. reducing \(\delta\) from \(10^{-3}\) to \(10^{-4}\) partially restores the
   no-decay behavior.

The theorem in `notes/20_weight_decay_compute_scaling.md` and
`notes/20_compute_optimal_weight_decay_scaling.md` explains this and predicts
that if \(\delta(C)=C^{-s}\), then

\[
    \beta_{\rm wd}(s)
    =
    \min\left\{\frac{b-1}{\max\{\alpha,b\}+1},\frac{s(b-1)}{\alpha}\right\}.
\]

## Next run

Run the single-theta check with

```bash
python experiments/compute_optimal_weight_decay_schedule.py \
  --a 3.0 --b 1.4 --theta 1.0 \
  --s-values 0,0.05,0.1,0.2,0.4,0.8 \
  --c-min 1e4 --c-max 1e10 --num-c 40 \
  --outdir experiments/results/compute_optimal_wd_schedule
```

Run the multi-theta check with

```bash
python experiments/compute_optimal_weight_decay_schedule.py \
  --a 3.0 --b 1.4 --theta-values 0,0.5,1 \
  --s-values 0,0.05,0.1,0.2,0.4,0.8 \
  --c-min 1e4 --c-max 1e10 --num-c 40 \
  --outdir experiments/results/compute_optimal_weight_decay_schedule
```

These directly test the compute-dependent weight-decay theorem.
