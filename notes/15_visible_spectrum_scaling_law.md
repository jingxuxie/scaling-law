# 15. Visible-spectrum scaling law in arbitrary optimizer coordinates

The previous notes identify the mechanism behind optimizer-dependent exponents:
coordinatewise second moments only help when they are spectrally informative.
This note packages that mechanism into a general scaling theorem for arbitrary
optimizer coordinates.

The theorem is deliberately stated with a separate source-condition assumption in
the transformed basis.  This is necessary: changing coordinates can preserve the
covariance spectrum while rotating the target.  A covariance-only theorem cannot
control the bias term unless the target/source condition is also stable.

## 1. Setup

Let

\[
    z\sim \mathcal N(0,\Sigma),
    \qquad
    y=\langle z,w^\star\rangle+\xi,
    \qquad
    \mathbb E\xi^2=\sigma^2.
\]

Let the eigenvalues of \(\Sigma\) satisfy

\[
    \lambda_i(\Sigma)\asymp i^{-a},
    \qquad a>1.
\]

Let \(P\succ0\) be the effective diagonal preconditioner produced by RMSProp,
Adam, AdamW without the decoupled shrinkage term, or a frozen proxy.  The
transformed covariance is

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}.
\]

The core spectral-comparability condition is: for some
\(\theta\in[0,1]\) and \(\rho>0\),

\[
\boxed{
    c(\Sigma^\theta+\rho I)^{-1/2}
    \preceq
    P
    \preceq
    C(\Sigma^\theta+\rho I)^{-1/2}.
}
\]

When \(\theta=1\), coordinatewise adaptivity is fully spectral and we recover the
aligned/band-limited Adam/RMSProp law.  When \(\theta=0\), the preconditioner is
scalar at exponent level and we recover SGD-like scaling.

## 2. Effective spectrum

By the spectral-comparability criterion,

\[
\boxed{
    \lambda_i(\widetilde\Sigma)
    \asymp
    \lambda_i(\Sigma)(\lambda_i(\Sigma)^\theta+\rho)^{-1/2}.
}
\]

If \(\lambda_i(\Sigma)\asymp i^{-a}\), then the transformed spectrum has the
pre-damping exponent

\[
\boxed{
    \alpha_{\rm eff}=a(1-\theta/2).
}
\]

Equivalently,

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
    & n\gtrsim \rho^{-(1/\theta-1/2)},
\end{cases}
}
\]

with the convention that \(\theta=0\) gives \(K(n)\asymp n^{1/a}\).

## 3. Source condition in the transformed problem

Let \((\widetilde\mu_i,\widetilde v_i)\) be the eigenpairs of
\(\widetilde\Sigma\), and let the transformed parameter be
\(u=P^{-1/2}w\).  The transformed target is

\[
    u^\star=P^{-1/2}w^\star.
\]

Assume the transformed source condition

\[
\boxed{
    \widetilde s_i
    :=
    \widetilde\mu_i\langle \widetilde v_i,u^\star\rangle^2
    \asymp i^{-b},
    \qquad b>1.
}
\]

When \(P\) is exactly a spectral function of \(\Sigma\), this source condition is
identical to the original source condition in the eigenbasis of \(\Sigma\).  In
noncommuting coordinate systems it is an additional target-alignment assumption,
and experiments should check its stability.

## Theorem 1: visible-spectrum scaling law

Assume the spectral-comparability condition, the transformed source condition,
and the standard one-pass SGD stability assumptions for the transformed Gaussian
problem.  Let

\[
    n=N_{\rm eff}\gamma,
    \qquad
    N_{\rm eff}=N/\log N.
\]

Then fixed-preconditioner SGD, frozen RMSProp, tracked RMSProp, and Adam up to
\(\beta_1\)-dependent constants satisfy

\[
\boxed{
    \mathbb E R_M(w_N)-\sigma^2
    \asymp
    \sum_{i>M}\widetilde s_i
    +
    \sum_{i\le M}\frac{\widetilde s_i}{1+n\widetilde\mu_i}
    +
    \frac1{N_{\rm eff}}
    \sum_{i\le M}\min\{1,n^2\widetilde\mu_i^2\}.
}
\]

In the clean source range

\[
    1<b<a(1-\theta/2)+1,
\]

and when \(\alpha_{\rm eff}=a(1-\theta/2)>1/2\), this simplifies to

\[
\boxed{
    
    \mathbb E R_M(w_N)-\sigma^2
    \asymp
    M^{1-b}
    +K_{\rho,\theta}(n)^{1-b}
    +\frac{\min\{M,K_{\rho,\theta}(n)\}}{N_{\rm eff}}.
}
\]
### Proof

Apply the change of variables

\[
    w=P^{1/2}u,
    \qquad
    \widetilde z=P^{1/2}z.
\]

Preconditioned SGD in \(w\) is ordinary SGD in \(u\), with covariance
\(\widetilde\Sigma=P^{1/2}\Sigma P^{1/2}\).  The spectral-comparability theorem
gives the effective eigenvalue sequence \(\widetilde\mu_i\).  The transformed
source condition gives the source energies \(\widetilde s_i\).  The standard
one-pass Gaussian SGD scaling theorem applied to the transformed problem gives
the three-filter expression.

The simplified form follows from the two-slope spectral-sum lower and upper
bounds in `09_matching_filter_lower_bounds.md`, with
\(\alpha=a(1-\theta/2)\).

## 4. AdamW visible-spectrum law

For AdamW, let \(\delta\) denote the decoupled weight-decay threshold in the same
units as the effective eigenvalues.  The active horizon becomes

\[
    T_{\rm eff}=\min\{n,\delta^{-1}\}.
\]

The learned-mode count is

\[
\boxed{
    K_{\rho,\theta,{\rm wd}}(n,\delta)
    \asymp
    K_{\rho,\theta}(T_{\rm eff}).
}
\]

The full AdamW filters are

\[
    B_{\rm wd}
    =
    \sum_{i\le M}\widetilde s_i
    \left[
        \left(\frac{\delta}{\widetilde\mu_i+\delta}\right)^2
        +
        \left(\frac{\widetilde\mu_i}{\widetilde\mu_i+\delta}\right)^2
        \frac1{1+n(\widetilde\mu_i+\delta)}
    \right],
\]

and

\[
    V_{\rm wd}
    =
    \frac1{N_{\rm eff}}
    \sum_{i\le M}
    \left(\frac{\widetilde\mu_i}{\widetilde\mu_i+\delta}\right)^2
    \min\{1,n^2(\widetilde\mu_i+\delta)^2\}.
\]

In the clean count regime, the simplified AdamW law is

\[
\boxed{
    R_M-\sigma^2
    \asymp
    M^{1-b}
    +K_{\rho,\theta}(\min\{n,\delta^{-1}\})^{1-b}
    +\frac{K_{\rho,\theta}(\min\{n,\delta^{-1}\})}{N_{\rm eff}}.
}
\]

## 5. Consequences

The theorem gives the paper's general mechanism:

\[
\boxed{
\text{visible spectral profile }\theta
\Rightarrow
q_{\rm eff}=\theta/2
\Rightarrow
K_{\rho,\theta}(n)
\Rightarrow
\text{optimizer-dependent scaling law}.
}
\]

The diagonal eigenbasis and band-limited cases have \(\theta=1\).  Flat,
Hadamard, Haar, and isotropic Gaussian sketch coordinates are typically
\(\theta\approx0\).  Structured random features may yield intermediate
\(\theta\), which is the most interesting next empirical target.

## 6. What remains

To turn this theorem into the main theorem of a high-impact paper, two pieces
remain:

1. an empirical estimator of \(\theta\) that is robust when \(P\) and \(\Sigma\)
   do not commute;
2. random-feature/sketched-model assumptions that imply spectral comparability
   or a controlled approximation to it.

The stochastic alignment experiment tests exactly whether the risk curves follow
this visible-spectrum prediction when coordinates are aligned, band-limited, or
globally mixed.
