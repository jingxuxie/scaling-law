# Review of `compute_optimal_visible` results

The deterministic compute-optimal experiment supports the visible-spectrum compute law.

For the reported run,

```text
a = 1.5
b = 1.4
theta in {0, 0.25, 0.5, 0.75, 1}
weight_decay = 0
C in [1e4, 1e10]
```

The summary file shows that the fitted exponents match the phase-diagram predictions.

## Main observations

1. At `theta=0`, the effective exponent is `alpha=1.5` and the phase is hard-source/time-limited.  The observed risk slope is about `-0.1667`, close to the predicted `-0.1600`.  The observed allocation exponents are `M≈C^0.416`, `N≈C^0.584`, close to the predicted `M≈C^0.4`, `N≈C^0.6`.

2. At `theta>0`, the run enters the easy-source/variance-limited phase because `b=1.4` exceeds `alpha=a(1-theta/2)` for these theta values.  The predicted compute-risk exponent saturates at `-(b-1)/(b+1)=-1/6≈-0.1667`, and the observed risk slope is `-0.166665` for all tested positive theta values.

3. The allocation exponents in the easy-source phase match the unified law with `m=max(alpha,b)=b`: `M≈C^{1/(b+1)}=C^{0.4167}`, `N≈C^{b/(b+1)}=C^{0.5833}`.  The observed exponents are about `0.4158` and `0.5842`.

4. Although the compute-risk exponent saturates for `theta>0`, the optimal effective horizon still changes with theta.  The observed `n_eff` exponent decreases from about `0.546` at `theta=0.25` to about `0.312` at `theta=1`, matching the prediction that stronger visible preconditioning needs a smaller effective learning horizon in the variance-limited phase.

## Interpretation

The run verifies an important subtlety: visible preconditioning does not always continue to improve the compute-risk exponent.  It improves the exponent until the effective spectral exponent `alpha=a(1-theta/2)` reaches the source smoothness scale `b`.  After that, the problem is variance-limited and additional visible preconditioning mostly changes constants and optimal learning-rate schedules.

This is not a failure of the theory.  It is exactly the two-phase prediction.

## Next recommended run

To show monotonic improvement of the compute-risk exponent with theta, run a harder-source setting where `b <= alpha(theta)` across more of the theta range, for example

```bash
python experiments/compute_optimal_visible_spectrum.py \
  --a 1.5 \
  --b 1.2 \
  --theta-values 0,0.25,0.5,0.75,1 \
  --rho 1e-16 \
  --dimension 500000 \
  --c-min 1e4 \
  --c-max 1e10 \
  --num-c 40 \
  --outdir experiments/results/compute_optimal_visible_b12
```

This should produce a risk exponent closer to

```text
-(b-1)/(a(1-theta/2)+1)
```

for the theta values that remain in the hard-source/time-limited phase.

Also run the AdamW ceiling experiment:

```bash
python experiments/weight_decay_compute_sweep.py \
  --a 1.5 --b 1.4 --theta 1.0 \
  --schedule-exponents 0,0.1,0.2,0.3,0.4,0.5 \
  --c-min 1e4 --c-max 1e10 --num-c 40 \
  --outdir experiments/results/weight_decay_compute
```

This checks the prediction that fixed weight decay creates a compute horizon ceiling, while weight decay scaled as `delta(C)≈C^{-s}` recovers the no-decay exponent once `s` exceeds the critical threshold.
