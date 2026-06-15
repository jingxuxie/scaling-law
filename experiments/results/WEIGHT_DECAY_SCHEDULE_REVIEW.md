# Review: compute-optimal weight-decay schedule results

The run in `experiments/results/compute_optimal_weight_decay_schedule` tests the theorem

\[
\beta_{\rm wd}(\theta,s)
=
\min\left\{
    \frac{(b-1)s}{\alpha},
    \frac{b-1}{\max\{\alpha,b\}+1}
\right\},
\qquad
\alpha=a(1-\theta/2),
\]

for compute-dependent AdamW decay

\[
\delta(C)=C^{-s}.
\]

## Configuration

The run used

\[
a=3.0,
\qquad
b=1.4,
\qquad
\theta\in\{0,0.5,1\},
\qquad
s\in\{0,0.05,0.1,0.2,0.4,0.8\},
\]

with \(C\in[10^4,10^{10}]\), 40 compute points, and grid size 600.

## Main finding

The results are supportive.

For every tested \(\theta\), the learned-mode exponent `obs_K_exp` matches the
predicted cap exponent essentially exactly in the cap-active regime:

\[
K_\star(C)\asymp C^{s/\alpha}.
\]

The allocation exponents `obs_M_exp` and `obs_N_exp` also match their predictions
up to small grid error.  The risk exponent tracks the predicted formula, with
slightly more finite-range bias because the risk is a sum of three terms and the
cap floor is approached gradually.

## Interpretation by row type

### Fixed weight decay, `s=0`

Fixed decay gives `pred_risk_exp = 0`, because the learned-mode count is capped
at a constant and the risk has an asymptotic floor.  The observed exponent is
small but slightly negative, around `-0.006`, which is a finite-compute transient
rather than an asymptotic power law.

### Cap-active schedules

For \(0<s<s_{\rm threshold}\), the weight-decay cap controls the learned-mode
count.  The exponent is

\[
\beta_{\rm wd}=\frac{(b-1)s}{\alpha}.
\]

This is exactly what the observed `K` exponents show, and the risk exponents
increase roughly linearly with \(s/\alpha\).

### Cap-inactive schedules

When

\[
s\ge s_{\rm threshold}=\frac{\alpha}{\max\{\alpha,b\}+1},
\]

the cap is inactive and the no-weight-decay law is recovered.  For example, at
\(\theta=1\), \(\alpha=1.5\), and \(s_{\rm threshold}=0.6\).  The row with
`s=0.8` recovers the no-decay exponent closely.

## Conclusion

The experiment supports the AdamW scaling message:

\[
\boxed{
\text{fixed AdamW decay creates a compute horizon ceiling; to preserve the no-decay scaling law, decay must decrease with compute.}
}
\]

Equivalently, the effective weight-decay threshold must satisfy

\[
\boxed{
\delta(C)\lesssim C^{-\alpha/(\max\{\alpha,b\}+1)}.
}
\]

This result should be part of the paper's main optimizer-scaling story, because
it connects the visible-spectrum exponent \(\theta\), Adam/RMSProp adaptivity,
and AdamW hyperparameter scaling.
