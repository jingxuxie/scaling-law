# 21. Random-feature visibility prediction

The current theory shows that coordinatewise adaptive optimizers change scaling
exponents only through spectral information visible to their coordinatewise
second moments.  This note states the random-feature version of that principle.

## Setup

Let

\[
    x\sim\mathcal N(0,H),
    \qquad
    H=\operatorname{diag}(\lambda_i),
    \qquad
    \lambda_i\asymp i^{-a}.
\]

Let a feature map or sketch be represented linearly by

\[
    z=Sx,
    \qquad
    \Sigma=S H S^\top.
\]

For coordinatewise RMSProp/Adam, the gradient-second-moment bridge gives

\[
    v_j\asymp d\,\Sigma_{jj},
\]

up to a scalar residual-energy factor.  Therefore the diagonal preconditioner is

\[
    P_j\asymp (\Sigma_{jj}+\rho)^{-1/2}.
\]

The effective covariance is

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}.
\]

## Visible-profile hypothesis

Suppose the feature coordinates have a sorted coordinate-variance profile obeying

\[
    d_{(j)}:=\operatorname{sort}(\Sigma_{jj})_j
    \asymp j^{-a\theta},
    \qquad 0\le \theta\le1,
\]

and suppose, in addition, that the preconditioner is spectrally comparable to a
visible spectral function:

\[
\boxed{
    c(\Sigma^\theta+\rho I)^{-1/2}
    \preceq
    P
    \preceq
    C(\Sigma^\theta+\rho I)^{-1/2}.
}
\]

The sorted diagonal condition alone is not enough in noncommuting coordinates;
the spectral-comparability condition is the needed alignment condition.

## Theorem: predicted random-feature exponent

Under the spectral-comparability hypothesis, the transformed covariance obeys

\[
\boxed{
    \lambda_i(\widetilde\Sigma)
    \asymp
    \lambda_i(\Sigma)(\lambda_i(\Sigma)^\theta+ho)^{-1/2}.
}
\]

Thus, before the damping knee,

\[
    \lambda_i(\widetilde\Sigma)\asymp i^{-a(1-	heta/2)},
\]

and

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
    n^{1/[a(1-	heta/2)]},
    & n\lesssim \rho^{-(1/\theta-1/2)},\\[4pt]
    \rho^{-1/(2a)}n^{1/a},
    & n\gtrsim \rho^{-(1/\theta-1/2)}.
\end{cases}
}
\]

with the natural interpretation \(K(n)\asymp n^{1/a}\) when \(\theta=0\).

## Consequences

- Isotropic/global Gaussian sketches typically flatten \(\Sigma_{jj}\), so
  \(\theta\approx0\) and diagonal Adam/RMSProp is scalar at exponent level.
- Aligned or band-limited feature maps have \(\theta\approx1\) and realize
  \(q_{\rm eff}\approx1/2\).
- Structured random features may have intermediate \(\theta\), giving a continuum
  of optimizer-dependent scaling laws.

## Experimental diagnostic

For any proposed random-feature map, report:

1. the covariance spectrum slope \(a_{\rm orig}\);
2. the coordinate-variance profile slope \(a\theta\);
3. the commutator diagnostic

\[
    \frac{\|P\Sigma-\Sigma P\|_{\rm op}}{\|P\Sigma\|_{\rm op}};
\]

4. the fitted effective exponent

\[
    q_{\rm eff}=1-\alpha_{\rm eff}/\alpha_{\rm orig};
\]

5. stochastic-training risk curves for SGD, coordinatewise Adam/RMSProp, and a
   spectral oracle.

A successful random-feature extension should show that the measured
\(q_{\rm eff}\) agrees with \(\theta/2\) when the commutator/alignment diagnostic
is favorable, and collapses toward zero for globally mixed sketches.
