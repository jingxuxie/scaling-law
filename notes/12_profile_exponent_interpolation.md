# 12. Diagonal-profile exponent interpolation theorem

The alignment notes prove two extremes and one robust intermediate case:

- aligned coordinates give `q_eff = 1/2`;
- flat/global mixing gives `q_eff = 0` at exponent level;
- arbitrary rotations within comparable-eigenvalue bands preserve `q_eff = 1/2`.

This note proves a quantitative interpolation theorem.  The effective exponent is controlled by how strongly the coordinate variance profile tracks spectral scale.

## 1. Band profile model

Let

\[
    H=\operatorname{diag}(\lambda_1,\ldots,\lambda_M),
    \qquad
    \lambda_i\asymp i^{-a},
    \qquad a>1.
\]

Let the optimizer-coordinate covariance be

\[
    \Sigma=Q^\top H Q.
\]

Partition coordinates into spectral bands \(B_\ell\).  Inside band \(B_\ell\), assume the population eigenvalues have scale \(\Lambda_\ell\):

\[
    \lambda_i\asymp_\kappa \Lambda_\ell,
    \qquad i\in B_\ell.
\]

Assume also that optimizer-coordinate variances inside the same band have scale \(\Delta_\ell\):

\[
    \Sigma_{jj}\asymp_\kappa \Delta_\ell,
    \qquad j\in B_\ell.
\]

The interpolation assumption is that coordinate variances track spectral scales by a power law:

\[
\boxed{
    \Delta_\ell\asymp \Lambda_\ell^\theta,
    \qquad 0\le \theta\le 1.
}
\]

The aligned/band-limited case has \(\theta=1\).  The flat-coordinate case has \(\theta=0\).

## 2. RMSProp/Adam preconditioner under a diagonal profile

By the coordinate-second-moment identity,

\[
    v_j\asymp d\,\Sigma_{jj},
\]

where \(d\) is a scalar residual-energy EMA.  Ignoring the scalar, the diagonal preconditioner is

\[
    D_\rho=\operatorname{diag}\left((\Sigma_{jj}+\rho)^{-1/2}\right).
\]

The transformed covariance is

\[
    \widetilde\Sigma=D_\rho^{1/2}\Sigma D_\rho^{1/2}.
\]

Within band \(B_\ell\), the diagonal entries of \(D_\rho\) are comparable to

\[
    (\Delta_\ell+\rho)^{-1/2}.
\]

Therefore the same Loewner/min-max argument as in the band-limited theorem gives, for every effective eigenvalue in band \(B_\ell\),

\[
\boxed{
    \widetilde\mu_i
    \asymp_\kappa
    \lambda_i(\Delta_\ell+\rho)^{-1/2}.
}
\]

Using \(\Delta_\ell\asymp \Lambda_\ell^\theta\) and \(\lambda_i\asymp \Lambda_\ell\), this becomes

\[
\boxed{
    \widetilde\mu_i
    \asymp_\kappa
    \lambda_i(\lambda_i^\theta+\rho)^{-1/2}.
}
\]

This is the profile-interpolated effective spectrum.

## 3. Effective exponent before the damping knee

When

\[
    \lambda_i^\theta\gg \rho,
\]

the effective eigenvalues satisfy

\[
    \widetilde\mu_i
    \asymp
    \lambda_i^{1-\theta/2}.
\]

Since \(\lambda_i\asymp i^{-a}\), this means

\[
    \widetilde\mu_i\asymp i^{-a(1-\theta/2)}.
\]

Thus the effective spectral exponent is

\[
\boxed{
    \alpha_{\rm eff}=a(1-\theta/2).
}
\]

Equivalently, the optimizer behaves like spectral preconditioning with

\[
\boxed{
    q_{\rm eff}=\theta/2.
}
\]

This interpolates between

\[
    \theta=0 \Rightarrow q_{\rm eff}=0,
    \qquad
    \theta=1 \Rightarrow q_{\rm eff}=1/2.
\]

## 4. Learned-mode count with damping

For \(\theta>0\), the damping knee is determined by

\[
    \lambda_i^\theta\asymp \rho.
\]

Since \(\lambda_i\asymp i^{-a}\), the index knee is

\[
\boxed{
    i_\rho\asymp \rho^{-1/(a\theta)}.
}
\]

At the knee,

\[
    \widetilde\mu_{i_\rho}\asymp \rho^{1/\theta-1/2}.
\]

Therefore the sample-size knee is

\[
\boxed{
    n_\rho\asymp \rho^{-(1/\theta-1/2)}.
}
\]

The learned-mode count is

\[
    K_{\rho,\theta}(n)=\#\{i:\widetilde\mu_i\gtrsim n^{-1}\}.
\]

Hence

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

For \(\theta=1\), this recovers the Adam/RMSProp formula.  For \(\theta=0\), the coordinate variances are flat, the preconditioner is scalar, and \(K_{\rho,0}(n)\asymp n^{1/a}\) up to constants.

## 5. Source-condition scaling law

Assume

\[
    s_i=\lambda_i(w_i^\star)^2\asymp i^{-b}.
\]

In the clean source range

\[
    1<b<a(1-\theta/2)+1,
\]

the sharp filter law gives

\[
\boxed{
    R_M-\sigma^2
    \asymp
    M^{1-b}
    +K_{\rho,\theta}(n)^{1-b}
    +\frac{K_{\rho,\theta}(n)}{N_{\rm eff}}.
}
\]

Before the damping knee, the bias exponent in \(n\) is

\[
\boxed{
    -\frac{b-1}{a(1-\theta/2)}.
}
\]

and the learned-count exponent is

\[
\boxed{
    \frac{1}{a(1-\theta/2)}.
}
\]

Thus measuring the coordinate-variance profile exponent \(\theta\) gives a direct prediction for the optimizer-dependent scaling exponent.

## 6. Empirical use

Given an optimizer coordinate covariance \(\Sigma\), compute

\[
    d_j=\Sigma_{jj}.
\]

Sort or band-average \((d_j)\), fit a slope against the corresponding spectral band scales, and estimate \(\theta\).  The prediction is

\[
\boxed{
    q_{\rm eff}\approx \theta/2.
}
\]

This gives a bridge between the diagonal/aligned theorem and random-feature models, where coordinates may partially but not perfectly reveal spectral scale.
