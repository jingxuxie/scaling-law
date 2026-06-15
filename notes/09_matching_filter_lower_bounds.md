# 09. Matching filter lower bounds and sharp learned-mode counts

The previous notes derive upper/sandwich filter laws for damped spectral preconditioning, RMSProp, Adam, and AdamW.  For a scaling-law paper, we also need matching lower bounds: the filters must not merely upper-bound risk; they must identify the correct order.

This note proves the deterministic spectral-sum lower bounds that make the learned-mode counts sharp.  The result applies to the diagonal Gaussian model once the algorithm has been reduced to an effective spectrum

\[
    \mu_i.
\]

For Adam/RMSProp,

\[
    \mu_i^\rho=\lambda_i(\lambda_i+\rho)^{-1/2}.
\]

For AdamW, the same \(\mu_i^\rho\) appears, but weight decay changes the active threshold from \(n^{-1}\) to \(\max\{n^{-1},\delta\}\).

## 1. General monotone filter calculus

Let \((\mu_i)_{i\ge1}\) be nonincreasing and positive.  Let

\[
    s_i\asymp i^{-b},
    \qquad b>1,
\]

where \(s_i=\lambda_i(w_i^\star)^2\) is the source energy.  For a horizon \(n\ge1\), define

\[
    K(n)=\#\{i:\mu_i\ge n^{-1}\}.
\]

The no-weight-decay bias and variance filters are

\[
    B(n,M)=\sum_{i\le M}\frac{s_i}{1+n\mu_i},
\]

and

\[
    V(n,M)=\frac1{N_{\rm eff}}\sum_{i\le M}\min\{1,n^2\mu_i^2\}.
\]

The elementary equivalences are

\[
\boxed{
    B(n,M)\asymp
    \sum_{i\le M}s_i\min\{1,(n\mu_i)^{-1}\},
}
\]

and

\[
\boxed{
    V(n,M)=
    \frac1{N_{\rm eff}}\sum_{i\le M}\min\{1,n^2\mu_i^2\}.
}
\]

The first identity uses

\[
    \frac1{1+x}\asymp \min\{1,x^{-1}\},
    \qquad x>0.
\]

Thus all matching bounds reduce to estimating spectral sums.

## 2. Two-slope spectra

Assume the effective spectrum has a two-slope form

\[
    \mu_i\asymp
    \begin{cases}
        i^{-\alpha}, & i\lesssim m,\\[3pt]
        m^{a-\alpha}i^{-a}, & i\gtrsim m,
    \end{cases}
\]

where

\[
    a>1,
    \qquad
    \alpha>0,
\]

and the normalization makes the two pieces match at \(i\asymp m\).  For damped \(q\)-preconditioning,

\[
    \alpha=a(1-q),
    \qquad
    m\asymp \rho^{-1/a}.
\]

For Adam/RMSProp, \(q=1/2\), so

\[
    \alpha=a/2.
\]

## Theorem 1: sharp bias count for two-slope spectra

Assume

\[
    1<b<\alpha+1.
\]

Let

\[
    K=K(n)=\#\{i:\mu_i\ge n^{-1}\}.
\]

If \(K<M\), then

\[
\boxed{
    B(n,M)\asymp K^{1-b}.
}
\]

If \(K\ge M\), then

\[
\boxed{
    B(n,M)\asymp n^{-1}\sum_{i\le M}\frac{s_i}{\mu_i}.
}
\]

In particular, in the non-saturated regime, the lower and upper bias bounds match the learned-mode prediction.

### Proof

Using the monotone filter equivalence,

\[
    B(n,M)\asymp
    \frac1n\sum_{i\le \min\{K,M\}}\frac{s_i}{\mu_i}
    +
    \sum_{K<i\le M}s_i.
\]

If \(K<M\), the tail term gives

\[
    \sum_{i>K}i^{-b}\asymp K^{1-b},
\]

so \(B(n,M)\gtrsim K^{1-b}\).

For the upper bound, split into cases.

If \(K\lesssim m\), then \(\mu_i\asymp i^{-\alpha}\) for \(i\le K\), so

\[
    \frac1n\sum_{i\le K}\frac{s_i}{\mu_i}
    \asymp
    \frac1n\sum_{i\le K}i^{\alpha-b}.
\]

The condition \(b<\alpha+1\) gives

\[
    \sum_{i\le K}i^{\alpha-b}\asymp K^{\alpha-b+1}.
\]

Since \(n\mu_K\asymp1\), we have \(n\asymp K^\alpha\), and therefore

\[
    \frac1n K^{\alpha-b+1}\asymp K^{1-b}.
\]

If \(K\gtrsim m\), split the head at \(m\).  For \(i>m\),

\[
    \mu_i\asymp m^{a-\alpha}i^{-a}.
\]

Using \(n\mu_K\asymp1\),

\[
    n\asymp m^{\alpha-a}K^a.
\]

Thus the post-knee part is

\[
    \frac1n\sum_{m<i\le K}\frac{i^{-b}}{m^{a-\alpha}i^{-a}}
    \asymp
    \frac{m^{\alpha-a}}n K^{a-b+1}
    \asymp
    K^{1-b}.
\]

The pre-knee part is lower order under \(b<\alpha+1\).  Indeed,

\[
    \frac1n\sum_{i\le m}i^{\alpha-b}
    \lesssim
    \frac{m^{\alpha-b+1}}{m^{\alpha-a}K^a}
    =
    m^{a-b+1}K^{-a}
    \le
    K^{1-b},
\]

because \(K\ge m\) and \(b>1\).  The tail term is again \(K^{1-b}\).  This proves the claim.

## Theorem 2: sharp variance count for two-slope spectra

Assume

\[
    \alpha>1/2
    \qquad\text{and}\qquad
    a>1/2.
\]

Then

\[
\boxed{
    V(n,M)\asymp \frac{\min\{M,K(n)\}}{N_{\rm eff}}.
}
\]

For Adam/RMSProp, \(\alpha=a/2\), so the condition \(\alpha>1/2\) follows from the original trace condition \(a>1\).

### Proof

If \(M\le K\), every term with \(i\le M\) satisfies \(n\mu_i\gtrsim1\), hence

\[
    V(n,M)\asymp M/N_{\rm eff}.
\]

Now suppose \(K<M\).  The first \(K\) terms contribute \(K/N_{\rm eff}\), so this is a lower bound.  It remains to upper-bound the tail:

\[
    n^2\sum_{i>K}\mu_i^2.
\]

If \(K\lesssim m\), then

\[
    n^2\sum_{K<i\le m}i^{-2\alpha}
    \lesssim
    n^2K^{1-2\alpha}
    \asymp K,
\]

because \(n\asymp K^\alpha\) and \(\alpha>1/2\).  The post-knee tail is no larger, because the spectrum is continuous at the knee and decays faster with exponent \(a>1/2\).

If \(K\gtrsim m\), then

\[
    n^2\sum_{i>K}m^{2(a-\alpha)}i^{-2a}
    \lesssim
    n^2m^{2(a-\alpha)}K^{1-2a}.
\]

Using \(n\asymp m^{\alpha-a}K^a\), this is \(O(K)\).  Therefore the total variance effective dimension is \(\asymp K\), proving the theorem.

## 3. Adam/RMSProp corollary

For Adam/RMSProp second-moment preconditioning,

\[
    \mu_i^\rho=\lambda_i(\lambda_i+\rho)^{-1/2}.
\]

If \(\lambda_i\asymp i^{-a}\), then

\[
    \mu_i^\rho
    \asymp
    \begin{cases}
        i^{-a/2}, & i\lesssim \rho^{-1/a},\\[3pt]
        \rho^{-1/2}i^{-a}, & i\gtrsim \rho^{-1/a}.
    \end{cases}
\]

Thus

\[
\boxed{
K_{\rho,1/2}(n)
\asymp
\begin{cases}
    n^{2/a}, & n\lesssim \rho^{-1/2},\\[3pt]
    \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2}.
\end{cases}
}
\]

If

\[
    1<b<a/2+1,
\]

then

\[
\boxed{
    B_{\rho,1/2}(n,M)\asymp K_{\rho,1/2}(n)^{1-b}
}
\]

whenever \(K_{\rho,1/2}(n)<M\).  If \(a>1\), then

\[
\boxed{
    V_{\rho,1/2}(n,M)\asymp \frac{\min\{M,K_{\rho,1/2}(n)\}}{N_{\rm eff}}.
}
\]

Therefore the simplified Adam/RMSProp scaling law is not only an upper bound but sharp at the level of spectral sums:

\[
\boxed{
    R_M-\sigma^2
    \asymp
    M^{1-b}
    +K_{\rho,1/2}(n)^{1-b}
    +\frac{\min\{M,K_{\rho,1/2}(n)\}}{N_{\rm eff}}.
}
\]

## 4. AdamW lower bounds

For AdamW, define

\[
    \theta=\max\{n^{-1},\delta\},
    \qquad
    T_{\rm eff}=\theta^{-1}=\min\{n,\delta^{-1}\},
\]

and

\[
    K_{\rm wd}(n,
    \delta)=\#\{i:\mu_i\ge\theta\}.
\]

The AdamW bias filter is

\[
    B_{\rm wd}(n,\delta,M)
    =
    \sum_{i\le M}s_i
    \left[
        \left(\frac{\delta}{\mu_i+\delta}\right)^2
        +
        \left(\frac{\mu_i}{\mu_i+
        \delta}\right)^2
        \frac{1}{1+n(\mu_i+
        \delta)}
    \right].
\]

The AdamW variance filter is

\[
    V_{\rm wd}(n,
    \delta,M)
    =
    \frac1{N_{\rm eff}}
    \sum_{i\le M}
    \left(\frac{\mu_i}{\mu_i+
    \delta}\right)^2
    \min\{1,n^2(\mu_i+
    \delta)^2\}.
\]

## Theorem 3: sharp AdamW learned-mode count

Under the two-slope assumptions above, if \(1<b<\alpha+1\), then whenever \(K_{\rm wd}<M\),

\[
\boxed{
    B_{\rm wd}(n,
    \delta,M)\asymp K_{\rm wd}(n,
    \delta)^{1-b}.
}
\]

If \(\alpha>1/2\), then

\[
\boxed{
    V_{\rm wd}(n,
    \delta,M)\asymp
    \frac{\min\{M,K_{\rm wd}(n,
    \delta)\}}{N_{\rm eff}}.
}
\]

Consequently, for AdamW with \(\mu_i^\rho=\lambda_i(\lambda_i+\rho)^{-1/2}\),

\[
\boxed{
K_{\rho,{\rm wd}}(n,
\delta)
\asymp
K_{\rho,1/2}(T_{\rm eff})
\asymp
\begin{cases}
    T_{\rm eff}^{2/a}, & T_{\rm eff}\lesssim \rho^{-1/2},\\[3pt]
    \rho^{-1/(2a)}T_{\rm eff}^{1/a}, & T_{\rm eff}\gtrsim \rho^{-1/2}.
\end{cases}
}
\]

### Proof

The upper bounds follow by the same decomposition as Theorems 1 and 2 with the threshold \(\theta\) replacing \(n^{-1}\).

For the bias lower bound, take indices with \(i>K_{\rm wd}\), so \(\mu_i<\theta=\max\{n^{-1},\delta\}\).  If \(\theta=\delta\), then \(\mu_i<\delta\), hence

\[
    \left(\frac{\delta}{\mu_i+\delta}\right)^2\ge 1/4.
\]

If \(\theta=n^{-1}\), then \(\delta\le n^{-1}\) and \(\mu_i<n^{-1}\), hence

\[
    n(\mu_i+\delta)\le2.
\]

For all \(x=\mu_i/\delta\in[0,\infty]\), with the convention for \(\delta=0\),

\[
    \left(\frac{\delta}{\mu_i+\delta}\right)^2
    +
    \left(\frac{\mu_i}{\mu_i+\delta}\right)^2
    \frac1{1+n(\mu_i+\delta)}
    \ge c
\]

for a universal constant \(c>0\).  Therefore

\[
    B_{\rm wd}(n,
    \delta,M)\gtrsim \sum_{i>K_{\rm wd}}s_i
    \asymp K_{\rm wd}^{1-b}.
\]

The upper bound is obtained by splitting \(i\le K_{\rm wd}\) and \(i>K_{\rm wd}\).  In the learned region \(\mu_i\ge\theta\), the finite-time term is bounded by a constant multiple of \((n\mu_i)^{-1}\) when \(\theta=n^{-1}\), and the shrinkage term is bounded by \((\delta/\mu_i)^2\) when \(\theta=\delta\).  The two-slope integral estimates from Theorem 1 control both contributions by \(K_{\rm wd}^{1-b}\).  The unlearned tail is \(\sum_{i>K_{\rm wd}}s_i\asymp K_{\rm wd}^{1-b}\).

For variance, indices \(i\le K_{\rm wd}\) contribute a constant per active mode unless the weight-decay shrinkage factor is at the transition; this still gives a constant-order lower bound on a fixed fraction of the first \(K_{\rm wd}\) modes.  The tail is controlled by the same squared-spectrum integral as in Theorem 2, with threshold \(\theta\).  Hence

\[
    V_{\rm wd}(n,
    \delta,M)\asymp \min\{M,K_{\rm wd}\}/N_{\rm eff}.
\]

This proves the AdamW count.

## 5. What this theorem contributes

The previous notes identified the right filters.  This note shows that, for the power-law regimes used in the scaling law, those filters are sharp.  In particular:

\[
\text{RMSProp/Adam:}\qquad
K(n)=K_{\rho,1/2}(n),
\]

and

\[
\text{AdamW:}\qquad
K(n)=K_{\rho,1/2}(\min\{n,
\delta^{-1}\}).
\]

Thus the project now has matching spectral upper and lower bounds for the diagonal Gaussian optimizer-theory chain.  What remains for a paper is less about inventing the diagonal theorem and more about packaging, strengthening assumptions, experiments, and extension beyond diagonal coordinates.
