# Results review: filter and alignment sweeps

The first experiment batch is encouraging.

## What passes clearly

The learned-mode count predictions are very strong.  Representative rows show:

- for `a=1.5`, Adam/RMSProp count slope is about `1.334` versus predicted `1.333`;
- for `a=2.0`, Adam/RMSProp count slope is about `1.004` or `0.999` versus predicted `1.0`;
- AdamW count slopes track the effective-horizon prediction after replacing `n` by `min(n, lambda_wd^{-1})`.

The coordinate-alignment experiment is especially clean.  Aligned coordinates give slope `-a/2`, flat coordinates give `-a`, and random rotations are close to the flat/SGD-like slope.

## What needs care

Some Adam/RMSProp bias slopes are worse than count slopes.  This is expected in three regimes:

1. `b` near 1: source tails converge slowly, so finite-dimensional spectral sums have large corrections.
2. `b >= a/2 + 1`: this is outside the clean bias theorem for the Adam/RMSProp head exponent `alpha=a/2`.
3. `K(n_max)` close to or larger than the truncation dimension: the fitted range includes finite-model saturation.

The next analysis should use active-window fits that exclude rows outside the theorem range and exclude points for which `K(n)` is too close to the ambient dimension.

## Next empirical step

Run:

```bash
python experiments/analyze_results.py \
  --filter-csv experiments/results/filter_main/filter_sweep.csv \
  --alignment-csv experiments/results/alignment_main/alignment_sweep.csv
```

Then run a larger/saner bias-focused sweep with smaller `n_max` for small `a` and low `b`, or larger dimension when testing aggressive Adam preconditioning.
