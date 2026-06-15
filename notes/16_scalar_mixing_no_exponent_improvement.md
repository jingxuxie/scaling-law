# 16. Scalar-mixing no-exponent-improvement theorem

The stochastic alignment experiments show a useful subtlety.  In flat or Haar-mixed coordinates, Adam/RMSProp diagnostics give

\[
    q_{\rm eff}\approx 0,
\]

yet adaptive methods can still have better finite-time constants than vanilla SGD.  This note explains the distinction: scalar or nearly scalar adaptivity can improve normalization, stability, and constants, but it cannot improve the **spectral scaling exponent**.

## 1. Fixed preconditioner statement

Let

\[
    z\sim \mathcal N(0,\Sigma),
    \qquad
    \lambda_i(\Sigma)\asymp i^{-a},
    \quad a>1.
\]

Consider a positive definite preconditioner \(P\) satisfying the scalar comparability condition

\[
\boxed{
    p_- I\preceq P\preceq p_+ I,
    \qquad
    0<p_-\le p_+<\infty.
}
\]

The transformed covariance is

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}.
\]

By Loewner order,

\[
\boxed{
    p_-\Sigma\preceq \widetilde\Sigma\preceq p_+\Sigma.
}
\]

Therefore, by the min-max principle,

\[
\boxed{
    p_-\lambda_i(\Sigma)
    \le
    \lambda_i(\widetilde\Sigma)
    \le
    p_+\lambda_i(\Sigma).
}
\]

Thus

\[
    \lambda_i(\widetilde\Sigma)\asymp i^{-a}.
\]

The learned-mode count remains

\[
\boxed{
    K(n)=\#\{i:\lambda_i(\widetilde\Sigma)\gtrsim n^{-1}\}
    \asymp n^{1/a}.
}
\]

So a scalar-comparable preconditioner has

\[
\boxed{q_{\rm eff}=0.}
\]

## 2. Risk-filter consequence

Under the usual source condition

\[
    s_i\asymp i^{-b},
    \qquad b>1,
\]

the spectral filters are the same as SGD up to constants:

\[
\boxed{
    B(n,M)
    =
    \sum_{i\le M}\frac{s_i}{1+n\lambda_i(\widetilde\Sigma)}
    \asymp
    \sum_{i\le M}\frac{s_i}{1+n\lambda_i(\Sigma)}.
}
\]

When \(1<b<a+1\) and \(K(n)<M\),

\[
\boxed{
    B(n,M)\asymp n^{-(b-1)/a}.
}
\]

Similarly,

\[
\boxed{
    V(n,M)
    \asymp
    \frac{\min\{M,n^{1/a}\}}{N_{\rm eff}}.
}
\]

Hence scalar-comparable adaptivity cannot improve the asymptotic bias or variance exponents over SGD.

## 3. Time-varying scalar-comparable preconditioners

Now suppose the preconditioner varies with time and satisfies

\[
\boxed{
    p_{t,-}I\preceq P_t\preceq p_{t,+}I,
    \qquad
    \frac{p_{t,+}}{p_{t,-}}\le \kappa
}
\]

uniformly over the training horizon.  In the population dynamics, coordinate \(i\) contracts by a product controlled by

\[
    \sum_{t<N}\gamma_t p_{t,\pm}\lambda_i.
\]

Define the scalar effective time

\[
    n_-=\sum_{t<N}\gamma_t p_{t,-},
    \qquad
    n_+=\sum_{t<N}\gamma_t p_{t,+}.
\]

Then

\[
    n_+\le \kappa n_-.
\]

The learned set is sandwiched as

\[
\boxed{
    \{i:n_-\lambda_i\gtrsim 1\}
    \subseteq
    \text{learned modes}
    \subseteq
    \{i:n_+\lambda_i\gtrsim 1\}.
}
\]

Therefore

\[
\boxed{
    K_{\rm learned}\asymp n_-^{1/a}
}
\]

up to \(\kappa\)-dependent constants.  Again, the exponent is SGD-like.

## 4. Application to flat, Haar, and isotropic-sketch coordinates

If coordinate variances are flat or bounded,

\[
    d_{\min}\le \Sigma_{jj}\le d_{\max},
    \qquad d_{\max}/d_{\min}=O(1),
\]

then RMSProp/Adam second moments satisfy

\[
    v_{t,j}\asymp d_t\Sigma_{jj}\asymp d_t
\]

uniformly over coordinates.  Hence

\[
    P_{t,j}=(v_{t,j}+\epsilon)^{-1/2}
\]

is scalar-comparable.  The theorem above applies and predicts

\[
\boxed{q_{\rm eff}=0.}
\]

This matches the flat/Haar stochastic alignment results: the direct \(q_{\rm eff}\) diagnostic is close to zero, even when Adam/RMSProp has better finite-time constants than SGD.

## 5. Interpretation for experiments

This theorem clarifies how to read the stochastic training plots.

- If Adam/RMSProp has \(q_{\rm eff}\approx1/2\), then it is changing the spectral scaling exponent.
- If Adam/RMSProp has \(q_{\rm eff}\approx0\), then any advantage is from scalar learning-rate normalization, stability, momentum, or constants, not from a spectral scaling-law improvement.

Therefore the next experimental step should tune learning rates separately by case and optimizer.  After tuning, compare both final risk and \(q_{\rm eff}\).  The high-impact claim should focus on the robust diagnostic:

\[
\boxed{
    \text{spectral exponent improvement occurs only when }q_{\rm eff}>0.
}
\]

The risk curves are important, but they must be interpreted together with the spectral diagnostic.
