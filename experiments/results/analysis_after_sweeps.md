# Analysis of `experiments/results` after first local sweeps

## Files reviewed

- `experiments/results/filter_main/config.json`
- `experiments/results/filter_main/filter_sweep.csv`
- `experiments/results/filter_main/summary.json`
- `experiments/results/alignment_main/config.json`
- `experiments/results/alignment_main/alignment_sweep.csv`
- `experiments/results/alignment_main/summary.json`

## Main takeaways

The deterministic learned-mode count experiments pass strongly.

For the filter sweep, the main grid used

```text
a in {1.25, 1.5, 2.0}
b in {1.2, 1.4, 1.8}
rho in {1e-16, 1e-12, 1e-8}
weight_decay in {0, 1e-4, 1e-3, 1e-2}
M = 300000
n in [10, 10000]
```

The count exponents match the predictions across the grid:

- SGD: `K(n) ~ n^{1/a}`.
- Adam/RMSProp: `K(n) ~ n^{2/a}` before the damping knee.
- AdamW: `K(n) ~ K_{rho,1/2}(min(n, lambda_wd^{-1}))`, so count growth saturates after the weight-decay horizon.

Representative rows:

```text
a=1.25: SGD count 0.807 vs 0.800, Adam count 1.586 vs 1.600.
a=1.50: SGD count 0.681 vs 0.667, Adam count 1.334 vs 1.333.
a=2.00: SGD count 0.520 vs 0.500, Adam count 1.004 vs 1.000.
```

AdamW count slopes also match the theory very closely when the predicted slope accounts for the weight-decay horizon.  For example, at `a=2.0`, `wd=1e-3`, the AdamW count slope is around `0.845` vs predicted `0.841`, and at `wd=1e-2` it is around `0.194` vs predicted `0.192`.

## Bias slope interpretation

Bias slopes match well in the clean source range and when finite-M saturation is not too strong.  Some rows show larger discrepancy, especially when:

1. the source exponent is outside or near the clean range `1 < b < alpha + 1`, where `alpha = a/2` for Adam/RMSProp;
2. `K_{rho,1/2}(n_max)` approaches or exceeds the finite dimension `M=300000`, so the finite-dimensional saturated filter is no longer well represented by the simple infinite-dimensional slope;
3. AdamW has a nonzero shrinkage floor, so a single global log-log slope over a range crossing the weight-decay horizon can mix several regimes.

Examples:

- For `a=1.25`, `alpha=0.625`, so `alpha+1=1.625`.  The rows with `b=1.8` are outside the clean source range, and the Adam bias slope is accordingly not expected to match the simple `-2(b-1)/a` prediction.
- For `a=1.25`, the Adam learned-count exponent is large, `2/a=1.6`; at large `n`, the implied learned count can be comparable to or larger than `M`, causing finite-dimensional effects.
- For `a=2.0`, `b=1.4`, the Adam bias slopes are close: roughly `-0.414` vs predicted `-0.400`.

Conclusion: the count law is robustly validated; bias-slope experiments should be refined with validity windows that avoid saturated finite-M and transition regimes.

## Coordinate-alignment sweep

The coordinate-alignment sweep used

```text
alignment_dimensions in {256, 512, 1024}
a in {1.25, 1.5, 2.0}
rho in {1e-12, 1e-8}
include_random = true
```

It strongly supports the coordinate-alignment theorem:

- aligned coordinates give effective slope `-a/2`;
- flat/Hadamard coordinates give slope `-a`;
- random orthogonal coordinates are close to `-a`, not `-a/2`.

Representative rows:

```text
d=512, a=1.25: aligned -0.625 vs -0.625, flat -1.25 vs -1.25, random -1.265 vs -1.25.
d=512, a=1.5: aligned -0.750 vs -0.750, flat -1.50 vs -1.50, random -1.526 vs -1.50.
d=512, a=2.0: aligned -1.000 vs -1.000, flat -2.00 vs -2.00, random -2.059 vs -2.00.
```

This is an important result: diagonal Adam/RMSProp does not automatically improve exponents under arbitrary rotations.  It improves exponents when the optimizer coordinates expose spectral structure to the coordinatewise second moments.

## Next experimental improvements

1. Add a validity-aware slope fitter that separately fits:
   - active pre-damping regime;
   - damping-limited regime;
   - AdamW pre-ceiling regime;
   - AdamW post-ceiling regime;
   - non-saturated finite-M regime.

2. Add stochastic-training versions of the deterministic filter experiments.

3. Add partial-alignment experiments with block rotations of increasing band width, because the next theorem predicts that rotations within comparable-eigenvalue bands preserve `q_eff=1/2`, while broad cross-band rotations reduce the effective exponent.

4. Add an experiment that estimates an empirical diagonal-profile exponent `theta` from coordinate variances and tests the interpolated prediction `q_eff = theta/2` when off-band leakage is controlled.
