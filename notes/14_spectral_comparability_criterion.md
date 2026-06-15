# 14. Spectral comparability criterion for coordinatewise adaptivity

The previous notes show that diagonal Adam/RMSProp changes scaling exponents only
when coordinatewise second moments retain spectral information.  This note gives
a coordinate-free sufficient condition: if the diagonal adaptive preconditioner
is Loewner-comparable to a spectral function of the covariance, then the
preconditioned spectrum and scaling law are determined by that function.

This theorem separates true spectral visibility from accidental variation in the
sorted coordinate variances.

## 1. Setup

Let \(\Sigma\succeq0\) be the covariance in optimizer coordinates, with ordered
eigenvalues

\[
    \lambda_1\ge\lambda_2\ge\cdots\ge0.
\]

Let \(P\succ0\) be the diagonal preconditioner produced by a coordinatewise
second-moment method, for example

\[
    P=\operatorname{diag}\left((\Sigma_{jj}+\rho)^{-1/2}\right).
\]

The transformed covariance is

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}.
\]

Let \(f:[0,\|\Sigma\|]\to(0,\infty)\) be a nonincreasing spectral profile.  The
case relevant for partial visibility is

\[
    f_{\theta,\rho}(t)=(t^\theta+\rho)^{-1/2}.
\]

## 2. Spectral-comparability assumption

Assume there are constants \(0<c\le C<\infty\) such that

\[
\boxed{
    c f(\Sigma)
    \preceq
    P
    \preceq
    C f(\Sigma).
}
\]

This condition says that the coordinatewise preconditioner behaves like a
spectral function of \(\Sigma\), not merely that its diagonal entries have a
nontrivial sorted slope.

## Theorem 1: effective eigenvalues under spectral comparability

Under the spectral-comparability assumption, the eigenvalues \(\widetilde\mu_i\)
of

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}
\]

satisfy

\[
\boxed{
    c\lambda_i f(\lambda_i)
    \le
    \widetilde\mu_i
    \le
    C\lambda_i f(\lambda_i)
    \qquad\text{for every }i.
}
\]

### Proof

The nonzero eigenvalues of

\[
    P^{1/2}\Sigma P^{1/2}
\]

are the same as the nonzero eigenvalues of

\[
    \Sigma^{1/2}P\Sigma^{1/2}.
\]

By Loewner comparability,

\[
    c\Sigma^{1/2}f(\Sigma)\Sigma^{1/2}
    \preceq
    \Sigma^{1/2}P\Sigma^{1/2}
    \preceq
    C\Sigma^{1/2}f(\Sigma)\Sigma^{1/2}.
\]

Since \(f(\Sigma)\) commutes with \(\Sigma\),

\[
    \Sigma^{1/2}f(\Sigma)\Sigma^{1/2}
    =
    \Sigma f(\Sigma).
\]

The eigenvalues of \(\Sigma f(\Sigma)\) are exactly

\[
    \lambda_i f(\lambda_i).
\]

The min-max principle then gives the claimed two-sided eigenvalue bound.

## 3. Partial-visibility corollary

Take

\[
    f_{\theta,\rho}(t)=(t^\theta+\rho)^{-1/2},
    \qquad 0\le\theta\le1.
\]

If

\[
    P\asymp f_{\theta,\rho}(\Sigma)
\]

in Loewner order, then

\[
\boxed{
    \widetilde\mu_i
    \asymp
    \lambda_i(\lambda_i^\theta+\rho)^{-1/2}.
}
\]

If

\[
    \lambda_i\asymp i^{-a},
\]

then before the damping knee,

\[
    \widetilde\mu_i\asymp i^{-a(1-\theta/2)}.
\]

Thus

\[
\boxed{
    q_{\rm eff}=\theta/2.
}
\]

The corresponding learned-mode count is

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

with the natural interpretation \(K(n)\asymp n^{1/a}\) when \(\theta=0\).

## 4. Stability under relative perturbations

Suppose instead that

\[
    P=f(\Sigma)+E
\]

and

\[
    \left\|f(\Sigma)^{-1/2}E f(\Sigma)^{-1/2}\right\|_{\rm op}
    \le \eta<1.
\]

Then

\[
    (1-\eta)f(\Sigma)
    \preceq
    P
    \preceq
    (1+\eta)f(\Sigma).
\]

Therefore Theorem 1 applies with constants \(1-\eta\) and \(1+\eta\).  In
particular, small relative perturbations of a spectral preconditioner do not
change exponents.

## 5. Relation to previous alignment theorems

This theorem subsumes the earlier cases.

### Aligned coordinates

If the optimizer coordinates are eigenvectors, then

\[
    P=(\Sigma+\rho I)^{-1/2}

\]

and the theorem applies with \(\theta=1\).  Hence \(q_{\rm eff}=1/2\).

### Flat coordinates

If \(P\asymp cI\), then the theorem applies with \(f(t)=c\), i.e. \(\theta=0\).
The spectrum is unchanged up to constants and \(q_{\rm eff}=0\).

### Band-limited coordinates

If coordinates mix only within comparable-eigenvalue bands, the previous
band-limited theorem proves

\[
    P\asymp (\Sigma+\rho I)^{-1/2}
\]

up to band constants.  Hence \(q_{\rm eff}=1/2\) again.

### Gaussian sketches

For an isotropic Gaussian sketch with sufficiently flat coordinate variances,
\(P\asymp cI\), so the theorem predicts \(q_{\rm eff}=0\).  If a structured sketch
produces a preconditioner spectrally comparable to
\((\Sigma^\theta+\rho I)^{-1/2}\), then the theorem predicts
\(q_{\rm eff}=\theta/2\).

## 6. Experimental diagnostic

For a feature map or sketch, compute:

1. the covariance \(\Sigma\);
2. the coordinate preconditioner \(P=\operatorname{diag}((\Sigma_{jj}+\rho)^{-1/2})\);
3. the effective covariance \(P^{1/2}\Sigma P^{1/2}\);
4. the fitted exponent \(q_{\rm eff}=1-\alpha_{\rm eff}/\alpha_{\rm orig}\);
5. a commutator diagnostic

\[
    \frac{\|P\Sigma-\Sigma P\|_{\rm op}}{\|P\Sigma\|_{\rm op}}.
\]

A small commutator is not sufficient by itself, but it is a useful warning
signal.  If the commutator is large, sorted diagonal-profile slopes alone should
not be trusted as spectral visibility estimates.

## 7. Consequence for the paper

The publishable message becomes:

\[
\boxed{
\text{Adaptive optimizers change scaling exponents only when their diagonal
second moments are spectrally comparable to a nontrivial function of the
covariance.}
}
\]

The diagonal eigenbasis theorem is the clean \(\theta=1\) case.  Randomly mixed
or isotropic sketched coordinates are often \(\theta=0\).  Structured
random-feature maps may produce intermediate \(\theta\), giving a continuum of
optimizer-dependent scaling laws.
