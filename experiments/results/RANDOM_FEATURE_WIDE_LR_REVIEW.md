# Review of random-feature wide-learning-rate results

The wide learning-rate random-feature run supports the visible-spectrum claim.

Configuration:

```text
a=1.5, b=1.4, dimension=256, ambient_dim=2048
cases=haar,gaussian_sketch
checkpoints=250,500,1000,2000,4000,8000,16000
trials=10
lr_sgd_values=0.03,0.1,0.3
lr_adaptive_values=0.0003,0.001,0.003,0.01,0.03,0.1
```

## Main findings

### Gaussian sketch

Best final risks:

```text
adam            final risk 8.35e-4, q_eff=-0.0278
adamw           final risk 1.04e-3, q_eff=-0.0276
rmsprop         final risk 1.52e-3, q_eff=-0.0316
sgd             final risk 7.37e-2, q_eff=0
coord_oracle    final risk 1.69e-3, q_eff=-0.0210
spectral_oracle final risk 1.45e-4, q_eff=0.5
```

The coordinatewise adaptive methods have near-zero effective spectral exponent, while the spectral oracle recovers `q_eff=1/2` and obtains the best final risk.

### Haar/global mixing

Best final risks:

```text
adam            final risk 1.30e-3, q_eff=-0.0243
adamw           final risk 7.84e-4, q_eff=-0.0231
rmsprop         final risk 9.50e-4, q_eff=-0.0269
sgd             final risk 2.73e-3, q_eff=0
coord_oracle    final risk 1.56e-3, q_eff=-0.0177
spectral_oracle final risk 1.16e-4, q_eff=0.5
```

Again, coordinatewise adaptivity improves finite-time constants but does not produce spectral exponent improvement.  The spectral oracle isolates the missing mechanism: access to the covariance eigenstructure.

## Interpretation

These results close the main empirical concern from the previous random-feature run.  With a wider learning-rate grid and more trials, the spectral oracle is now clearly better than coordinatewise Adam/RMSProp in globally mixed feature coordinates.

The paper should phrase the empirical lesson as:

```text
Coordinatewise adaptive optimizers can improve finite-time risk even when q_eff≈0,
but they improve the spectral scaling exponent only when their second moments are
spectrally informative.
```

This is consistent with the tuned-risk separation theorem and the visible-spectrum theorem.

## Paper-readiness assessment

The project is ready to move into manuscript writing.  Additional toy experiments are unlikely to change the core scientific story.  Any further experiments should be aimed at presentation quality, not mechanism discovery.
