# Analysis after stochastic-training and sketch-visibility sweeps

This note reviews the new experiment files under `experiments/results`.

## Files reviewed

- `experiments/results/stochastic_main/config.json`
- `experiments/results/stochastic_main/summary.csv`
- `experiments/results/stochastic_main/training_curves.csv`
- `experiments/results/sketch_visibility/sketch_visibility_sweep.csv`

## 1. Stochastic training: what it supports

The stochastic run used

```text
a = 1.5
b = 1.4
dimension = 4096
sigma = 0.1
batch_size = 16
checkpoints = 250,500,1000,2000,4000,8000
trials = 5
optimizers = sgd, oracle_rmsprop, frozen_rmsprop, rmsprop, adam, adamw
```

The most important result is the second-moment mechanism.  RMSProp, Adam, and
AdamW all have measured

```text
q_eff_mean ≈ 0.5
```

throughout training.  This directly supports the proof chain

\[
    v_{t,i}\propto \lambda_i
    \quad\Rightarrow\quad
    (v_{t,i}+\epsilon)^{-1/2}\propto (\lambda_i+\rho)^{-1/2}
    \quad\Rightarrow\quad
    q_{\rm eff}\approx1/2.
\]

Representative values from `summary.csv`:

```text
rmsprop: q_eff_mean ≈ 0.50 at all checkpoints
adam:    q_eff_mean ≈ 0.50 at all checkpoints
adamw:   q_eff_mean ≈ 0.50 at all checkpoints
```

The risk curves also show a clear optimizer separation: SGD is much slower, while
RMSProp/Adam/AdamW rapidly reduce excess risk.  However, the fitted stochastic
risk slopes are steeper than the deterministic spectral-filter exponents.  This
means the stochastic-training run should be treated as evidence for the
mechanism and qualitative separation, not yet as a calibrated scaling-law slope
experiment.

Likely reasons:

1. learning rates were not chosen to match the one-pass geometric schedule in the theorem;
2. the run is finite-dimensional and reaches very small risks for adaptive methods;
3. optimizer transients and batch-size effects influence fitted slopes;
4. the risk slope is fitted over a single hyperparameter configuration.

## 2. Sketch visibility: what it supports

The sketch-visibility sweep is very strong.  It tests whether the effective
preconditioned covariance

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}
\]

has exponent shift \(q_{\rm eff}\).

The results show:

```text
aligned:          q_eff ≈ 0.5
profile(theta):   q_eff ≈ theta/2
band-limited:     q_eff ≈ 0.49-0.50
global Haar:      q_eff ≈ 0
gaussian_sketch:  q_eff ≈ 0
```

Representative rows:

```text
a=1.5, rho=1e-12:
aligned          q_eff = 0.500
profile theta=.5 q_eff = 0.250
band             q_eff = 0.499
haar             q_eff = -0.019
gaussian_sketch  q_eff = -0.020
```

This supports the current paper-level claim:

\[
\text{coordinatewise Adam/RMSProp changes exponents only when feature coordinates expose spectral information.}
\]

## 3. Important diagnostic caveat

The sketch-visibility rows also show that the **sorted coordinate variance slope
alone is not enough**.

For Haar/global Gaussian sketch cases, the sorted diagonal profile can have a
moderate finite-dimensional slope, but \(q_{\rm eff}\approx0\).  The reason is
that the diagonal preconditioner is not spectrally aligned with \(\Sigma\).  It
is a non-commuting coordinate rescaling, not a spectral function of \(\Sigma\).

This motivated `notes/16_bounded_preconditioner_no_exponent_change.md`.

The reliable diagnostics are:

1. directly fit the eigenvalue slope of \(P^{1/2}\Sigma P^{1/2}\);
2. measure the condition number of the diagonal preconditioner;
3. measure the commutator ratio

\[
    \|P\Sigma-\Sigma P\|_{\rm op}/\|P\Sigma\|_{\rm op};
\]

4. use coordinate-variance profile slopes only when spectral alignment or
   spectral comparability has been established.

## 4. Does this provide enough evidence?

It provides enough evidence for the core mechanism in a theory-driven paper:

- deterministic filters validate the predicted count exponents;
- stochastic training validates \(q_{\rm eff}\approx1/2\) in actual optimizer updates;
- sketch visibility validates the aligned/band-limited/global-mixing phase diagram.

For a higher-impact paper, the remaining empirical step is a small random-feature
or sketched-linear **training** experiment, not only a covariance diagnostic.  The
most valuable next experiment would train SGD/RMSProp/Adam on:

1. aligned diagonal features;
2. band-limited random features;
3. global Gaussian sketch features;

and compare the observed risk slopes and \(q_{\rm eff}\) diagnostics.
