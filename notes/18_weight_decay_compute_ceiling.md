# 18. AdamW weight decay creates a compute horizon ceiling

This note extends the compute-optimal visible-spectrum theorem to fixed AdamW-style decoupled weight decay.  The main message is that weight decay does not change the local visible-spectrum exponent before its ceiling, but it caps the effective horizon.  To preserve a compute-optimal scaling law as compute grows, weight decay must itself decay with compute.

## Setup

Let the visible-spectrum exponent be

\[
    \alpha=a(1-\theta/2),
\]

so without weight decay the learned-mode horizon is controlled by the effective time \(n\).  In the clean risk law,

\[
    R(M,n)-\sigma^2
    \asymp
    M^{1-b}+K(n)^{1-b}+K(n)/N,
\]

with

\[
    K(n)\asymp n^{1/\alpha}
\]

before damping.  AdamW decoupled weight decay adds the cutoff

\[
    n\le \delta^{-1},
\]

so the effective horizon is

\[
\boxed{
    n_{\rm eff}=\min\{n,\delta^{-1}\}.
}
\]

Thus

\[
\boxed{
    K_\delta(n)\asymp K(\min\{n,\delta^{-1}\}).
}
\]

## Theorem 1: fixed weight decay saturates compute scaling

Assume \(\delta>0\) is fixed and the model dimension \(M\) and sample size \(N\) grow with compute \(C\asymp MN\).  Then the learned-mode count is bounded by

\[
    K_\delta(n)\le K(\delta^{-1})\asymp \delta^{-1/\alpha}.
\]

Therefore the optimization/source term cannot decay below

\[
\boxed{
    K(\delta^{-1})^{1-b}
    \asymp
    \delta^{(b-1)/\alpha}.
}
\]

Consequently, for any fixed positive weight decay, the asymptotic compute-risk curve eventually has a nonzero weight-decay-induced floor:

\[
\boxed{
    \liminf_{C\to\infty}\left(R_\delta^\star(C)-\sigma^2\right)
    \gtrsim
    \delta^{(b-1)/\alpha}.
}
\]

This is not a contradiction to AdamW's finite-compute benefits.  It says that fixed weight decay is a scale-dependent regularizer and must be retuned as compute grows.

## Theorem 2: scaling weight decay to preserve the no-decay exponent

Let the no-weight-decay compute-optimal effective horizon be

\[
    n_\star(C)\asymp C^{\alpha/(m+1)},
    \qquad
    m=\max\{\alpha,b\}.
\]

If weight decay satisfies

\[
\boxed{
    \delta(C)\lesssim n_\star(C)^{-1}
    \asymp C^{-\alpha/(m+1)},
}
\]

then the AdamW ceiling is inactive at the compute optimum, and the no-decay compute exponent is preserved:

\[
\boxed{
    R_{\delta(C)}^\star(C)-\sigma^2
    \asymp
    C^{-(b-1)/(m+1)}.
}
\]

If instead

\[
    \delta(C)\asymp C^{-s}
\]

with

\[
    0<s<\alpha/(m+1),
\]

then the weight-decay ceiling is active and the best possible source/optimization decay is at most

\[
\boxed{
    \delta(C)^{(b-1)/\alpha}
    \asymp
    C^{-s(b-1)/\alpha}.
}
\]

Thus the compute-risk exponent is capped by

\[
\boxed{
    \beta_\delta(s)\le s(b-1)/\alpha.
}
\]

The no-decay exponent is recovered exactly at the threshold

\[
    s=\alpha/(m+1).
\]

## Consequence for experiments

For a fixed compute range, the transition should occur near

\[
    \delta\approx n_\star(C_{\max})^{-1}.
\]

A practical deterministic experiment should sweep

\[
    \delta\in\{0, C^{-s}:s\in[0,1]\}
\]

and check that fixed \(\delta\) bends the risk curve while scaled \(\delta(C)\) restores the visible-spectrum compute exponent.

## Consequence for the paper

This theorem turns AdamW weight decay into a scaling-law prediction:

\[
\boxed{
    \lambda_{\rm wd}\text{ is not just a constant hyperparameter; it has a compute scaling law.}
}
\]

For visible exponent \(	heta\), the critical AdamW decay scaling is

\[
\boxed{
    \lambda_{\rm wd}(C)\lesssim C^{-a(1-\theta/2)/(\max\{a(1-\theta/2),b\}+1)}.
}
\]

This is a concrete, testable prediction and should be included in the final paper as an AdamW-specific corollary.
