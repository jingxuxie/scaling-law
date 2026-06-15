# 12. Smooth-source bias saturation

The exponent-level filter sweep revealed an important point: the simplified bias law

\[
    B(n)\asymp K(n)^{1-b}
\]

is sharp only in the rough/intermediate source regime

\[
    1<b<\alpha+1,
\]

where \(\alpha\) is the active effective spectral exponent.  When the target is smoother than the effective covariance, \(b\ge \alpha+1\), the bias no longer improves with exponent \((b-1)/\alpha\).  It saturates at the optimization filter exponent \(-1\) in effective time \(n\), up to a logarithm at the boundary.

This note proves the missing spectral-sum theorem.  It explains why some Adam/RMSProp bias slopes in the sweeps are less steep than the naive clean-regime prediction when \(b\) is too large relative to \(a/2\).

## 1. One-slope bias filter

Assume

\[
    \mu_i\asymp i^{-\alpha},
    \qquad
    s_i\asymp i^{-b},
    \qquad
    \alpha>0,
    \quad b>1.
\]

The bias filter is

\[
    B(n,M)=\sum_{i\le M}\frac{s_i}{1+n\mu_i}.
\]

Let

\[
    K(n)=\#\{i:\mu_i\ge n^{-1}\}\asymp n^{1/\alpha}.
\]

Assume \(K(n)<M\) so the model-size cutoff is not active.

## Theorem 1: complete bias phase diagram

For the one-slope spectrum above,

\[
\boxed{
B(n,M)\asymp
\begin{cases}
    n^{-(b-1)/\alpha}, & 1<b<\alpha+1,\\[4pt]
    n^{-1}\log n, & b=\alpha+1,\\[4pt]
    n^{-1}, & b>\alpha+1,
\end{cases}
}
\]

up to constants depending on \(a,b\) and the spectral comparability constants.

Equivalently, ignoring the logarithmic boundary,

\[
\boxed{
    B(n,M)\asymp n^{-\min\{(b-1)/\alpha,\,1\}}.
}
\]

### Proof

Use

\[
    \frac1{1+n\mu_i}\asymp \min\{1,(n\mu_i)^{-1}\}.
\]

Then

\[
    B(n,M)\asymp
    \frac1n\sum_{i\le K}\frac{s_i}{\mu_i}
    +
    \sum_{K<i\le M}s_i.
\]

Since \(s_i/\mu_i\asymp i^{\alpha-b}\), the head term is

\[
    H(n):=\frac1n\sum_{i\le K}i^{\alpha-b}.
\]

The tail term is

\[
    T(n):=\sum_{i>K}i^{-b}\asymp K^{1-b}\asymp n^{-(b-1)/\alpha}.
\]

There are three cases.

If \(b<\alpha+1\), then \(\alpha-b>-1\), and

\[
    H(n)\asymp n^{-1}K^{\alpha-b+1}
    \asymp n^{-1}n^{(\alpha-b+1)/\alpha}
    =n^{-(b-1)/\alpha}.
\]

Thus \(H\) and \(T\) have the same order.

If \(b=\alpha+1\), then

\[
    H(n)\asymp n^{-1}\log K\asymp n^{-1}\log n,
\]

while \(T(n)\asymp K^{-\alpha}\asymp n^{-1}\).  The logarithmic head dominates.

If \(b>\alpha+1\), then \(\alpha-b<-1\), so

\[
    \sum_{i\le K}i^{\alpha-b}\asymp 1,
\]

and

\[
    H(n)\asymp n^{-1}.
\]

The tail term is

\[
    T(n)\asymp n^{-(b-1)/\alpha}=o(n^{-1}),
\]

because \(b-1>\alpha\).  Thus \(B(n,M)\asymp n^{-1}\).

This proves the theorem.

## 2. Two-slope extension

For damped Adam/RMSProp, the effective spectrum is two-slope:

\[
    \mu_i^\rho=\lambda_i(\lambda_i+\rho)^{-1/2}
    \asymp
    \begin{cases}
        i^{-a/2}, & i\lesssim \rho^{-1/a},\\[3pt]
        \rho^{-1/2}i^{-a}, & i\gtrsim \rho^{-1/a}.
    \end{cases}
\]

In the pre-damping active regime \(n\lesssim\rho^{-1/2}\), the active exponent is

\[
    \alpha=a/2.
\]

Therefore the Adam/RMSProp bias slope in \(n\) is

\[
\boxed{
    -\min\left\{\frac{b-1}{a/2},\,1\right\}
    =
    -\min\left\{\frac{2(b-1)}{a},\,1\right\},
}
\]

up to the logarithmic boundary \(b=a/2+1\).

In the post-damping tail regime \(n\gtrsim\rho^{-1/2}\), the active exponent reverts to

\[
    \alpha=a,
\]

so the bias slope becomes

\[
\boxed{
    -\min\left\{\frac{b-1}{a},\,1\right\}.
}
\]

Damping therefore creates both a spectrum-knee transition and a possible smooth-source saturation transition.

## 3. AdamW version

For AdamW, define

\[
    T_{\rm eff}=\min\{n,\delta^{-1}\}.
\]

In the active pre-damping phase, the bias behaves as

\[
\boxed{
    B_{\rm AdamW}(n)
    \asymp
    T_{\rm eff}^{-\min\{2(b-1)/a,\,1\}}
}
\]

again up to the logarithmic boundary and shrinkage-floor constants.  Once \(n\gtrsim\delta^{-1}\), the effective horizon stops increasing, so the bias saturates even if the no-weight-decay optimizer would keep improving.

## 4. Consequence for interpreting the sweeps

The clean learned-count predictions are robust:

\[
    K_{\rm SGD}(n)\asymp n^{1/a},
    \qquad
    K_{\rm Adam}(n)\asymp n^{2/a}
\]

before damping.  But the simplified bias prediction

\[
    B_{\rm Adam}(n)\asymp K_{\rm Adam}(n)^{1-b}
    \asymp n^{-2(b-1)/a}
\]

requires

\[
    b<a/2+1.
\]

When \(b>a/2+1\), the correct slope is \(-1\), not \(-2(b-1)/a\).  This explains why experiments with \(a=1.25,b=1.8\) or \(a=1.5,b=1.8\) should not be judged against the clean-regime slope.  They are in or near the smooth-source saturation regime.

## 5. Next implementation update

Future versions of `experiments/run_sweeps.py` should report both:

1. the clean-regime predicted bias slope, useful when \(1<b<\alpha+1\);
2. the saturated predicted bias slope

\[
    -\min\{(b-1)/\alpha,1\},
\]

which is valid across smoothness regimes.

The current sweep results already support the main count predictions.  The bias-slope discrepancies mainly identify where the clean-regime theorem's assumptions are being violated or where finite-size effects are still visible.
