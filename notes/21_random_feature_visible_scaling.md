# 21. Random-feature visible-spectrum scaling theorem

This note packages the project for random-feature and sketched-linear models.  It states the theorem that connects a feature map's visible spectral information to the optimizer-dependent scaling law.

## Setup

Let

\[
    z=Sx,
    \qquad x\sim \mathcal N(0,H),
\]

so the trainable-coordinate covariance is

\[
    \Sigma=SHS^\top.
\]

Let \(\lambda_i(\Sigma)\asymp i^{-a}\).  A coordinatewise second-moment method such as RMSProp or Adam constructs, up to scalar residual factors,

\[
    P=\operatorname{diag}\left((\Sigma_{jj}+\rho)^{-1/2}\right).
\]

The effective covariance is

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}.
\]

## Visible spectral comparability

Assume there exists \(0\le\theta\le1\) and constants \(0<c<C<\infty\) such that

\[
\boxed{
    c(\Sigma^\theta+\rho I)^{-1/2}
    \preceq
    P
    \preceq
    C(\Sigma^\theta+\rho I)^{-1/2}.
}
\]

This is the formal version of saying that coordinatewise second moments expose a \(\theta\)-fraction of the covariance spectrum.

## Theorem 1: visible spectrum determines the optimizer exponent

Under the comparability condition above, the eigenvalues of \(\widetilde\Sigma\) satisfy

\[
\boxed{
    \lambda_i(\widetilde\Sigma)
    \asymp
    \lambda_i(\Sigma)\left(\lambda_i(\Sigma)^\theta+\rho\right)^{-1/2}.
}
\]

Therefore, before the damping knee,

\[
    \lambda_i(\widetilde\Sigma)
    \asymp
    i^{-a(1-\theta/2)}.
\]

Thus the effective optimizer exponent is

\[
\boxed{
    q_{\rm eff}=\theta/2.
}
\]

The learned-mode count is

\[
\boxed{
K_{\rho,\theta}(n)
\asymp
\begin{cases}
    n^{1/[a(1-\theta/2)]},
    & n\lesssim \rho^{-(1/\theta-1/2)},\\[4pt]
    \rho^{-1/(2a)}n^{1/a},
    & n\gtrsim \rho^{-(1/\theta-1/2)}.
\end{cases}
}
\]

with the interpretation \(K(n)\asymp n^{1/a}\) for \(\theta=0\).

### Proof

The nonzero eigenvalues of \(P^{1/2}\Sigma P^{1/2}\) equal those of \(\Sigma^{1/2}P\Sigma^{1/2}\).  By Loewner comparability,

\[
    c\Sigma^{1/2}(\Sigma^\theta+\rho I)^{-1/2}\Sigma^{1/2}
    \preceq
    \Sigma^{1/2}P\Sigma^{1/2}
    \preceq
    C\Sigma^{1/2}(\Sigma^\theta+\rho I)^{-1/2}\Sigma^{1/2}.
\]

The outer terms commute with \(\Sigma\) and have eigenvalues

\[
    \lambda_i(\Sigma)(\lambda_i(\Sigma)^\theta+\rho)^{-1/2}.
\]

The min-max principle gives the two-sided eigenvalue bound.  The power-law learned-mode count follows by solving \(\lambda_i(\widetilde\Sigma)\gtrsim n^{-1}\).

## Theorem 2: risk and compute scaling

Assume the transformed source energies satisfy the band/source condition

\[
    \widetilde s_i\asymp i^{-b}
\]

or the corresponding band-averaged version.  In the clean range

\[
    1<b<a(1-\theta/2)+1,
\]

one-pass training obeys

\[
\boxed{
    R_M-\sigma^2
    \asymp
    M^{1-b}
    +K_{\rho,\theta}(n)^{1-b}
    +\frac{\min\{M,K_{\rho,\theta}(n)\}}{N_{\rm eff}}.
}
\]

Under compute \(C\asymp MN\), let

\[
    \alpha=a(1-\theta/2),
    \qquad m=\max\{\alpha,b\}.
\]

Then

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-(b-1)/(m+1)}.
}
\]

AdamW with \(\delta(C)=C^{-s}\) has compute exponent

\[
\boxed{
    \beta_{\rm wd}(\theta,s)
    =
    \min\left\{
        \frac{(b-1)s}{a(1-\theta/2)},
        \frac{b-1}{\max\{a(1-\theta/2),b\}+1}
    \right\}.
}
\]

## Consequences for feature maps

### Aligned or band-limited random features

If rows of \(S\) mix only comparable-eigenvalue bands, then \(P\) is comparable to \((\Sigma+\rho I)^{-1/2}\).  Hence \(\theta=1\) and \(q_{\rm eff}=1/2\).

### Global isotropic Gaussian sketches

If rows of \(S\) mix all spectral scales isotropically and the coordinate variances concentrate, then \(P\asymp cI\).  Hence \(\theta=0\) and \(q_{\rm eff}=0\).

### Intermediate feature maps

If empirical diagnostics show

\[
    P\approx (\Sigma^\theta+\rho I)^{-1/2}
\]

in spectral-comparability or effective-eigenvalue terms, then the theorem predicts \(q_{\rm eff}=\theta/2\).  The script `experiments/random_feature_visibility_training.py` tests this mechanism under actual stochastic training.

## What this contributes

This theorem is the bridge from the diagonal proof to random-feature and sketched-linear models.  It shows that the relevant quantity is not merely whether an optimizer is adaptive, but whether the feature coordinates expose the covariance spectrum to coordinatewise second moments.
