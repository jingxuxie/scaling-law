# 17. Tuned risk comparisons versus spectral exponent improvements

The stochastic alignment experiments show an important distinction: Adam/RMSProp can improve finite-time risk even when the measured spectral exponent is unchanged.  This note formalizes why this does not contradict the visible-spectrum theory.

## 1. Scalar tuning can change constants but not power laws

Suppose an optimizer has effective spectrum

\[
    \widetilde\mu_i\asymp \lambda_i^{1-q},
    \qquad
    \lambda_i\asymp i^{-a}.
\]

Let

\[
    \alpha(q)=a(1-q).
\]

In the clean source range, the risk proxy is

\[
    R(M,n)-\sigma^2
    \asymp
    M^{1-b}+n^{-(b-1)/\alpha(q)}+\frac{n^{1/\alpha(q)}}{N_{\rm eff}}.
\]

Changing the scalar learning rate rescales the effective horizon:

\[
    n\mapsto cn.
\]

This changes constants:

\[
    (cn)^{-(b-1)/\alpha(q)}=c^{-(b-1)/\alpha(q)}n^{-(b-1)/\alpha(q)},
\]

but it does not change the exponent in \(n\).  Therefore learning-rate tuning can improve finite-time risk without changing the spectral exponent class.

## 2. Visible spectral information changes the exponent class

If the visible preconditioning exponent changes from \(q=0\) to \(q>0\), then

\[
    \alpha(q)=a(1-q)<a.
\]

In the hard-source/time-limited phase, the compute-optimal exponent is

\[
    \beta(q)=\frac{b-1}{a(1-q)+1}.
\]

Thus

\[
\boxed{
    q_2>q_1
    \quad\Longrightarrow\quad
    \beta(q_2)>\beta(q_1)
}
\]

as long as both are in the hard-source phase.

For Adam/RMSProp in aligned or band-limited coordinates, \(q=1/2\).  For flat/Haar coordinates, \(q=0\).  The compute-exponent gap is

\[
\boxed{
    \beta(1/2)-\beta(0)
    =
    (b-1)\left(\frac{1}{a/2+1}-\frac{1}{a+1}\right)>0.
}
\]

## 3. Interpretation of the LR sweep

The LR sweep shows that flat/Haar Adam can beat SGD in finite-time risk while retaining

\[
    q_{\rm eff}\approx0.
\]

This means that those improvements are due to scalar normalization, momentum, transient effects, or constants.  They are not the \(a\mapsto a/2\) spectral flattening mechanism.

The spectral oracle in flat/Haar coordinates retains

\[
    q_{\rm eff}=1/2,
\]

and therefore isolates what coordinatewise Adam is missing: spectral information in its coordinatewise second moments.

## 4. Paper message

A tuned optimizer comparison should report both:

1. best final risk after learning-rate tuning;
2. measured \(q_{\rm eff}\) or effective spectral exponent.

The first measures finite-time performance.  The second measures the scaling-law mechanism.  A high-impact paper should distinguish these quantities explicitly.
