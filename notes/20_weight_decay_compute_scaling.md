# 20. Compute-optimal scaling with compute-dependent AdamW weight decay

The compute-optimal visible-spectrum theorem assumes no weight-decay cap.  AdamW adds an effective horizon cap

\[
    n\le \delta^{-1},
\]

where \(\delta\) is the weight-decay threshold in the same units as the visible effective spectrum.  This note derives the compute law when \(\delta\) is fixed or scaled with compute.

## 1. Setup

Let

\[
    \alpha=a(1-\theta/2),
\]

and use the simplified risk proxy

\[
    R(M,N,K)-\sigma^2
    \asymp
    M^{1-b}+K^{1-b}+\frac{K}{N}.
\]

The constraints are

\[
    C\asymp MN,
    \qquad
    K\le N^{1/\alpha},
    \qquad
    K\le K_\delta:=\delta^{-1/\alpha}.
\]

The no-weight-decay compute law has

\[
    m=\max\{\alpha,b\},
    \qquad
    K_\star^{(0)}(C)\asymp C^{1/(m+1)}.
\]

## 2. Fixed weight decay produces an asymptotic floor

If \(\delta>0\) is fixed, then \(K_\delta\) is fixed.  Hence the bias term

\[
    K^{1-b}
\]

cannot decay past

\[
    K_\delta^{1-b}=\delta^{(b-1)/\alpha}.
\]

Thus

\[
\boxed{
    \liminf_{C\to\infty}\left(R_\star(C)-\sigma^2\right)
    \gtrsim
    \delta^{(b-1)/\alpha}.
}
\]

So fixed AdamW weight decay ultimately destroys asymptotic power-law decay, even though over finite compute ranges it can still improve constants and regularize variance.

## 3. Decay schedule \(\delta(C)=C^{-s}\)

Now suppose

\[
    \delta(C)=C^{-s},
    \qquad s\ge0.
\]

Then

\[
    K_\delta(C)=\delta(C)^{-1/\alpha}=C^{s/\alpha}.
\]

The no-decay optimum is feasible precisely when

\[
    K_\delta(C)\gtrsim K_\star^{(0)}(C),
\]

that is,

\[
    C^{s/\alpha}\gtrsim C^{1/(m+1)}.
\]

Therefore the critical scaling is

\[
\boxed{
    s_c=\frac{\alpha}{m+1}.
}
\]

If

\[
    s\ge s_c,
\]

then weight decay is asymptotically inactive and the no-decay compute law is preserved:

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-(b-1)/(m+1)}.
}
\]

## 4. Cap-active phase

If

\[
    0\le s<s_c,
\]

then the weight-decay cap is active:

\[
    K_\star(C)\asymp K_\delta(C)=C^{s/\alpha}.
\]

The cap-induced bias is

\[
    K_\delta^{1-b}\asymp C^{-s(b-1)/\alpha}.
\]

In the relevant cap-active range, this term dominates the optimized approximation/variance terms, so the risk exponent becomes

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-s(b-1)/\alpha}
    \qquad (0\le s<s_c).
}
\]

The fixed-weight-decay case is \(s=0\), giving exponent zero and a nonzero floor.

## 5. Unified exponent

The compute-risk exponent under \(\delta(C)=C^{-s}\) is

\[
\boxed{
    \beta_{\rm wd}(s)
    =
    \min\left\{
        \frac{b-1}{m+1},
        \frac{s(b-1)}{\alpha}
    \right\}.
}
\]

Equivalently, AdamW preserves the no-decay scaling law if and only if

\[
\boxed{
    \delta(C)\lesssim C^{-\alpha/(m+1)}.
}
\]

## 6. Consequence for experiments

The compute-optimal phase-grid runs with fixed \(\delta=10^{-3}\) and \(10^{-4}\) should not match the no-weight-decay prediction columns.  They are intentionally cap-active over much of the tested compute range.  The expected signatures are:

1. learned-mode exponent \(K_\star(C)\) collapses relative to the no-decay exponent;
2. risk exponent becomes much smaller in magnitude;
3. optimal \(M\) grows faster because additional compute is forced into approximation error once the learned-mode cap is active;
4. decreasing \(\delta\) from \(10^{-3}\) to \(10^{-4}\) partially restores the no-decay exponents.

This is exactly the qualitative pattern observed in the current phase-grid results.

## 7. Next diagnostic experiment

To directly verify the theorem, run a compute-dependent weight-decay schedule experiment with

\[
    \delta(C)=C^{-s}
\]

for several \(s\) values.  The script `experiments/compute_optimal_weight_decay_schedule.py` implements this test and compares observed exponents to

\[
    \beta_{\rm wd}(s)
    =
    \min\left\{\frac{b-1}{m+1},\frac{s(b-1)}{\alpha}\right\}.
\]
