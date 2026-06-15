# 12. Gaussian sketches flatten coordinate second moments

The coordinate-alignment theorem shows that diagonal RMSProp/Adam acts spectrally only when coordinatewise gradient second moments retain spectral information.  This note applies that principle to Gaussian sketches.

## Setup

Let

\[
H=\operatorname{diag}(\lambda_1,\ldots,\lambda_d)
\]

and let the trainable features be sketched coordinates

\[
z=Sx,
\qquad
\Sigma=SHS^\top,
\]

where the rows \(s_j^\top\) of \(S\in\mathbb R^{M\times d}\) are independent isotropic sketch vectors.  For simplicity take

\[
s_j\sim \mathcal N(0,I_d/d).
\]

The coordinate variance seen by RMSProp/Adam is

\[
d_j=\Sigma_{jj}=s_j^\top Hs_j.
\]

By the arbitrary-coordinate second-moment lemma,

\[
\mathbb E[g_j^2\mid e]
=
c_e d_j+2(\Sigma e)_j^2,
\]

so the adaptive diagonal preconditioner is controlled by \(d_j\), not by the ordered eigenvalues \(\lambda_i\).

## Theorem 1: Gaussian sketch diagonals concentrate around trace scale

Let

\[
\bar d=\mathbb E d_j=\frac{\operatorname{tr}(H)}{d}.
\]

Define the effective rank

\[
r_{\rm eff}=\frac{\operatorname{tr}(H)^2}{\operatorname{tr}(H^2)}.
\]

There are universal constants \(c,C>0\) such that for every \(t>0\),

\[
\boxed{
\Pr\left(|d_j-ar d|>t\bar d\right)
\le
2\exp\left[-c r_{\rm eff}\min\{t^2,t\}\right].
}
\]

Consequently, if

\[
r_{\rm eff}\gtrsim \log(M/\delta),
\]

then with probability at least \(1-\delta\), simultaneously for all sketch coordinates,

\[
\boxed{
(1/2)\bar d\le d_j\le (3/2)\bar d.
}
\]

Thus the diagonal RMSProp/Adam preconditioner is scalar up to constants:

\[
(d_j+
ho)^{-1/2}
\asymp
(ar d+
ho)^{-1/2}.
\]

Therefore

\[
D_\rho^{1/2}\Sigma D_\rho^{1/2}
\asymp
(ar d+
ho)^{-1/2}

times \Sigma,
\]

and diagonal adaptivity does not change the eigenvalue power-law exponent.

### Proof

The random variable \(d_j=s_j^\top Hs_j\) is a weighted chi-square sum.  Hanson-Wright or the standard Laurent-Massart quadratic-form bound gives

\[
\Pr(|d_j-ar d|>tar d)
\le
2\exp[-c\min\{t^2\operatorname{tr}(H)^2/\operatorname{tr}(H^2),
 t\operatorname{tr}(H)/\|H\|_{\rm op}\}].
\]

Since \(\|H\|_{\rm op}\le \sqrt{\operatorname{tr}(H^2)}\), the first displayed bound follows up to constants.  A union bound over \(M\) coordinates gives the simultaneous statement.

## Theorem 2: damped scalarization under bounded diagonal spread

Even when \(r_{\rm eff}
ot\gg
gg \log M\), suppose the realized sketch diagonals satisfy

\[
0<d_-\le d_j\le d_+<\infty.
\]

With damping \(\rho>0\), the transformed covariance obeys

\[
\boxed{
(d_++\rho)^{-1/2}\Sigma
\preceq
D_\rho^{1/2}\Sigma D_\rho^{1/2}
\preceq
(d_-+
ho)^{-1/2}\Sigma.
}
\]

Thus all transformed eigenvalues differ from the original sketch covariance eigenvalues by at most the scalar condition factor

\[
\left(\frac{d_++\rho}{d_-+
ho}\right)^{1/2}.
\]

If this factor is bounded or polylogarithmic in dimension, the exponent-level learned-mode count remains SGD-like.  In particular, an unstructured Gaussian sketch does not produce the aligned Adam/RMSProp exponent improvement unless its coordinate variances encode spectral scale.

## Interpretation

A Gaussian sketch mixes all spectral scales into each coordinate.  Diagonal Adam/RMSProp sees the diagonal profile \(d_j=(SHS^\top)_{jj}\), which is exchangeable across rows and typically trace-scale.  It therefore behaves like a scalar preconditioner at exponent level.

This is the opposite of band-limited alignment: if trainable coordinates mix only comparable eigenvalues, Adam/RMSProp preserves \(q_{\rm eff}=1/2\); if trainable coordinates mix all eigenvalues isotropically, diagonal adaptivity loses spectral information and gives \(q_{\rm eff}=0\) at exponent level.

## Consequence for the original sketched-linear model

For the exact Gaussian-sketch model, the next theorem should analyze \(\Sigma=SHS^\top\)'s eigenvalue law together with the diagonal profile \(\operatorname{diag}(\Sigma)\).  The prediction is:

\[
\boxed{
\text{i.i.d. Gaussian sketch coordinates should not show the full }a\mapsto a/2\text{ Adam exponent shift.}
}
\]

To obtain Adam-like spectral flattening in a sketched/random-feature model, the feature map must retain spectral-band information in coordinatewise second moments.
