# 16. Bounded diagonal preconditioners cannot change spectral exponents

The sketch-visibility sweep shows an important diagnostic caveat. In Haar and global Gaussian-sketch coordinates, the sorted coordinate-variance profile can have a nonzero finite-dimensional slope, but the effective spectral exponent is still essentially unchanged. The reason is that the diagonal preconditioner is not spectrally aligned with the covariance, and in many globally mixed cases it is bounded in condition number. A bounded diagonal preconditioner can change constants but not power-law eigenvalue exponents.

## 1. Setup

Let \(\Sigma\succeq0\) have eigenvalues

\[
    \lambda_i(\Sigma)\asymp i^{-a},
    \qquad a>0.
\]

Let \(P\succ0\) be any positive diagonal preconditioner in the optimizer coordinates. The transformed covariance is

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}.
\]

Write

\[
    p_{\min}=\lambda_{\min}(P),
    \qquad
    p_{\max}=\lambda_{\max}(P),
    \qquad
    \kappa_P=p_{\max}/p_{\min}.
\]

## Theorem 1: bounded preconditioners preserve the eigenvalue exponent

If \(0<p_{\min}\le p_{\max}<\infty\), then for every \(i\),

\[
\boxed{
    p_{\min}\lambda_i(\Sigma)
    \le
    \lambda_i(\widetilde\Sigma)
    \le
    p_{\max}\lambda_i(\Sigma).
}
\]

Consequently, if \(\kappa_P=O(1)\) along a sequence of dimensions, then

\[
\boxed{
    \lambda_i(\widetilde\Sigma)\asymp i^{-a}.
}
\]

Thus the learned-mode count remains SGD-like:

\[
\boxed{
    K_{\widetilde\Sigma}(n)\asymp n^{1/a}.
}
\]

At exponent level,

\[
\boxed{q_{\rm eff}=0.}
\]

### Proof

Since

\[
    p_{\min} I\preceq P\preceq p_{\max} I,
\]

congruence by \(\Sigma^{1/2}\) gives

\[
    p_{\min}\Sigma
    \preceq
    \Sigma^{1/2}P\Sigma^{1/2}
    \preceq
    p_{\max}\Sigma.
\]

The matrices \(P^{1/2}\Sigma P^{1/2}\) and \(\Sigma^{1/2}P\Sigma^{1/2}\) have the same nonzero eigenvalues. The min-max principle therefore implies the stated eigenvalue sandwich.

## 2. Sub-polynomial condition numbers

The same conclusion holds at exponent level if the preconditioner condition number grows sub-polynomially with dimension. Suppose

\[
    \kappa_P(M)=M^{o(1)}.
\]

Then, over ranks \(i\le M\), the eigenvalue sandwich becomes

\[
    \lambda_i(\widetilde\Sigma)=i^{-a+o(1)}
\]

whenever \(\lambda_i(\Sigma)=i^{-a+o(1)}\). Thus no fixed positive exponent improvement can be inferred from such a preconditioner.

This explains why globally mixed coordinates can have finite-dimensional sorted diagonal slopes without producing a true Adam/RMSProp spectral exponent shift.

## 3. Implication for coordinate-variance diagnostics

Let

\[
    d_j=\Sigma_{jj}
\]

and

\[
    P_j=(d_j+\rho)^{-1/2}.
\]

A sorted power-law slope in \(d_j\) is **not sufficient** to conclude \(q_{\rm eff}=\theta/2\). That conclusion requires spectral alignment or a Loewner/spectral-comparability condition such as

\[
    P\asymp f(\Sigma)
\]

for a spectral function \(f\). Without such alignment, \(P\) may be merely a bounded diagonal rescaling in a basis that does not commute with \(\Sigma\), and then Theorem 1 says the spectral exponent is unchanged.

Thus the right empirical diagnostics are:

1. the effective eigenvalue slope of \(P^{1/2}\Sigma P^{1/2}\);
2. the condition number of \(P\);
3. a commutator or spectral-comparability diagnostic, for example

\[
    \frac{\|P\Sigma-\Sigma P\|_{\rm op}}{\|P\Sigma\|_{\rm op}};
\]

4. the coordinate-variance profile slope only when alignment or low commutator has already been established.

## 4. Relation to the sketch-visibility experiment

The sketch-visibility sweep shows:

\[
\begin{array}{lll}
\text{aligned} & q_{\rm eff}\approx1/2, & P\text{ commutes with }\Sigma,\\[3pt]
\text{band-limited} & q_{\rm eff}\approx1/2, & P\text{ is block-spectral up to constants},\\[3pt]
\text{Haar/global sketch} & q_{\rm eff}\approx0, & P\text{ is not spectrally aligned.}
\end{array}
\]

This theorem explains the third line. Even if the sorted diagonal profile has a moderate finite-dimensional slope, the preconditioner may be bounded and non-commuting, so it cannot change the covariance eigenvalue exponent.

## 5. Paper-level takeaway

For a high-impact theorem, the main object should not be the sorted diagonal of \(\Sigma\) alone. It should be **visible spectral preconditioning**:

\[
    P^{1/2}\Sigma P^{1/2},
\]

or sufficient conditions that make \(P\) comparable to a spectral function of \(\Sigma\). This is the right bridge from diagonal Gaussian regression to sketched/random-feature models.
