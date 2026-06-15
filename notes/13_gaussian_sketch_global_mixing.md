# 13. Global Gaussian sketches flatten coordinate variances

The band-limited theorem says that diagonal Adam/RMSProp preserves spectral flattening when optimizer coordinates mix only comparable-eigenvalue directions.  This note proves the complementary statement: a global isotropic Gaussian sketch tends to flatten coordinate variances, so coordinatewise Adam/RMSProp becomes approximately scalar and does not change the spectral exponent.

This is important for connecting to Gaussian-sketched linear regression.  If the trainable sketch coordinates are global random mixtures of all population eigenmodes, then the diagonal second moment

\[
    v_j\asymp d\,(S H S^\top)_{jj}
\]

may contain little ordered spectral information.  In the high-effective-rank regime, all diagonal entries concentrate near the same value.

## 1. Setup

Let

\[
    H\succeq0
\]

be a covariance matrix in population eigenbasis, and let

\[
    S\in\mathbb R^{M\times D}
\]

have independent isotropic Gaussian rows

\[
    s_j\sim\mathcal N(0,I_D/D).
\]

The sketched feature covariance is

\[
    \Sigma_S=S H S^\top.
\]

The coordinate variance seen by diagonal RMSProp/Adam in sketch coordinate \(j\) is

\[
    d_j=(\Sigma_S)_{jj}=s_j^\top Hs_j.
\]

Let

\[
    \tau=\mathbb E[d_j]=\frac{\operatorname{tr}(H)}{D}.
\]

Define the effective rank

\[
\boxed{
    r_{\rm eff}(H)=\frac{\operatorname{tr}(H)^2}{\operatorname{tr}(H^2)}.
}
\]

## 2. Concentration of sketch-coordinate variances

For a Gaussian quadratic form,

\[
    d_j-\tau=s_j^\top Hs_j-\frac{\operatorname{tr}(H)}D.
\]

A standard Hanson-Wright or chi-square weighted-sum bound gives, for \(0<\varepsilon<1\),

\[
\boxed{
    \Pr\left(|d_j-\tau|>\varepsilon\tau\right)
    \le
    2\exp\left(-c\varepsilon^2 r_{\rm eff}(H)\right)
}
\]

for a universal constant \(c>0\).

Union bounding over \(M\) sketch coordinates gives the following theorem.

## Theorem 1: global Gaussian sketches have flat coordinate variances

If

\[
\boxed{
    r_{\rm eff}(H)
    \ge
    C\varepsilon^{-2}\log(2M/\delta),
}
\]

then with probability at least \(1-\delta\), simultaneously for all \(j\le M\),

\[
\boxed{
    (1-\varepsilon)\tau
    \le
    d_j
    \le
    (1+\varepsilon)\tau.
}
\]

Thus the coordinate variance profile of the global Gaussian sketch is flat up to a factor

\[
    \frac{1+\varepsilon}{1-\varepsilon}.
\]

## 3. Consequence for RMSProp/Adam preconditioning

By the arbitrary-coordinate second-moment identity,

\[
    v_j\asymp d_{\rm res}\,d_j,
\]

where \(d_{\rm res}\) is the residual-energy scalar/EMA.  Ignoring this scalar, RMSProp/Adam applies a diagonal preconditioner

\[
    D_\rho=\operatorname{diag}\left((d_j+\rho)^{-1/2}\right).
\]

On the event of Theorem 1,

\[
    ((1+\varepsilon)\tau+\rho)^{-1/2}I
    \preceq
    D_\rho
    \preceq
    ((1-\varepsilon)\tau+\rho)^{-1/2}I.
\]

The transformed sketched covariance is

\[
    \widetilde\Sigma_S=D_\rho^{1/2}\Sigma_S D_\rho^{1/2}.
\]

Therefore, in Loewner order,

\[
\boxed{
    ((1+\varepsilon)\tau+\rho)^{-1/2}\Sigma_S
    \preceq
    \widetilde\Sigma_S
    \preceq
    ((1-\varepsilon)\tau+\rho)^{-1/2}\Sigma_S.
}
\]

By the min-max principle, every eigenvalue is changed only by a common constant factor:

\[
\boxed{
    ((1+\varepsilon)\tau+\rho)^{-1/2}\lambda_k(\Sigma_S)
    \le
    \lambda_k(\widetilde\Sigma_S)
    \le
    ((1-\varepsilon)\tau+\rho)^{-1/2}\lambda_k(\Sigma_S).
}
\]

Thus global Gaussian-sketch coordinates make diagonal RMSProp/Adam approximately scalar at exponent level.

## 4. Learned-mode count consequence

If

\[
    \lambda_k(\Sigma_S)\asymp k^{-a_S},
\]

then, under the effective-rank condition above,

\[
    \lambda_k(\widetilde\Sigma_S)\asymp k^{-a_S}.
\]

Hence the learned-mode count remains

\[
\boxed{
    K_{\rm sketch,Adam}(n)\asymp n^{1/a_S}
}
\]

up to constants, not

\[
    n^{2/a_S}.
\]

So in a high-effective-rank global Gaussian sketch, coordinatewise Adam/RMSProp should not exhibit the aligned-coordinate \(a\mapsto a/2\) spectral flattening.  Any exponent improvement would require nontrivial alignment between sketch coordinates and spectral bands.

## 5. Low-effective-rank caveat

The condition

\[
    r_{\rm eff}(H)\gg \log M
\]

is sufficient, not necessary.  For trace-class power laws with small effective rank, the diagonal entries \(d_j\) may fluctuate.  However, in an isotropic global sketch they are exchangeable random quadratic forms, not ordered eigenvalues.  Their sorted order statistics can create some coordinate heterogeneity, but this heterogeneity is not aligned with eigenvectors or source coefficients in the way required for the oracle \(q=1/2\) spectrum.

This motivates an empirical alignment statistic for arbitrary sketches:

\[
    d_j=(S H S^\top)_{jj},
\]

plus an off-band leakage norm.  If the sorted \(d_j\) profile is flat, expect \(q_{\rm eff}\approx0\).  If the feature map is band-limited so that \(d_j\) tracks spectral band scales, expect \(q_{\rm eff}\approx1/2\).

## 6. Implication for the project

The original Gaussian-sketch model is a potentially hostile setting for diagonal Adam/RMSProp spectral gains, because global random sketch coordinates mix many population eigenmodes.  This makes the coordinate-alignment experiments central rather than auxiliary.

The next model-level theorem should consider structured or band-limited random features.  The expected phase diagram is:

\[
\boxed{
\begin{array}{lll}
\text{spectral/eigenbasis features} & \Rightarrow & q_{\rm eff}=1/2,\\[3pt]
\text{band-limited random features} & \Rightarrow & q_{\rm eff}=1/2 \text{ up to constants},\\[3pt]
\text{global Gaussian sketches} & \Rightarrow & q_{\rm eff}=0 \text{ under high effective rank.}
\end{array}
}
\]

This sharpens the paper's claim: Adam/RMSProp changes scaling exponents when the optimizer coordinate system exposes spectral structure; otherwise, its diagonal adaptivity mainly changes constants.
