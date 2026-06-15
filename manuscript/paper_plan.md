# Paper plan: Visible spectrum, adaptive optimizers, and scaling laws

## Candidate title

**Visible Spectrum, Adaptive Optimizers, and Scaling Laws in Linear Regression**

## Main claim

Coordinatewise adaptive optimizers change scaling exponents only through spectral information visible to their coordinatewise second moments.

Equivalently:

\[
\text{visible spectral exponent }\theta
\quad\Longrightarrow\quad
q_{\rm eff}=\theta/2
\quad\Longrightarrow\quad
K_{\rho,\theta}(n)
\quad\Longrightarrow\quad
R(M,N), R_\star(C).
\]

## Core theorem stack

1. **Damped spectral preconditioning.**
   If \(P_{\rho,q}=(H+\rho I)^{-q}\), then the effective covariance eigenvalues are
   \[
   \mu_i=\lambda_i(\lambda_i+\rho)^{-q}.
   \]

2. **RMSProp/Adam second moments.**
   In Gaussian linear regression,
   \[
   \mathbb E[g_i^2\mid e]=\lambda_i(\sigma^2+\|e\|_H^2)+2\lambda_i^2e_i^2,
   \]
   so RMSProp/Adam learns \(P_i\asymp(\lambda_i+\rho)^{-1/2}\) in aligned coordinates.

3. **Visible-spectrum theorem.**
   If
   \[
   P\asymp(\Sigma^\theta+\rho I)^{-1/2},
   \]
   then
   \[
   \lambda_i(P^{1/2}\Sigma P^{1/2})
   \asymp
   \lambda_i(\Sigma)(\lambda_i(\Sigma)^\theta+\rho)^{-1/2},
   \]
   and \(q_{\rm eff}=\theta/2\).

4. **Risk law.**
   In the clean source range,
   \[
   R_M-\sigma^2
   \asymp
   M^{1-b}+K_{\rho,\theta}(n)^{1-b}
   +\frac{\min\{M,K_{\rho,\theta}(n)\}}{N_{\rm eff}}.
   \]

5. **Compute-optimal law.**
   With \(\alpha=a(1-\theta/2)\) and \(m=\max\{\alpha,b\}\),
   \[
   R_\star(C)-\sigma^2
   \asymp
   C^{-(b-1)/(m+1)}.
   \]

6. **AdamW schedule law.**
   If \(\delta(C)=C^{-s}\), then
   \[
   \beta_{\rm wd}(\theta,s)
   =
   \min\left\{
      \frac{(b-1)s}{a(1-\theta/2)},
      \frac{b-1}{\max\{a(1-\theta/2),b\}+1}
   \right\}.
   \]

## Main figures

1. **Mechanism schematic.**
   Coordinatewise second moments reveal a visible spectral profile; this determines \(q_{\rm eff}\), learned modes, and scaling.

2. **Deterministic filter exponents.**
   Observed versus predicted slopes for SGD, Adam/RMSProp, and AdamW.

3. **Visibility phase diagram.**
   Aligned and band-limited coordinates give \(q_{\rm eff}\approx1/2\); flat, Haar, and global Gaussian sketches give \(q_{\rm eff}\approx0\); synthetic profiles give \(q_{\rm eff}=\theta/2\).

4. **Stochastic alignment training.**
   Learning-rate calibrated training curves and measured \(q_{\rm eff}\) across aligned, band, flat, and Haar coordinates.

5. **Theta-profile stochastic interpolation.**
   Best final risk and risk slope improve monotonically as \(\theta\) increases.

6. **Compute-optimal phase transition.**
   Risk exponent versus \(\theta\), showing improvement until the variance-limited saturation phase.

7. **AdamW weight-decay schedule.**
   Fixed decay creates a floor; \(\delta(C)=C^{-s}\) recovers the no-decay law once \(s\) exceeds the predicted threshold.

8. **Random-feature wide-LR experiment.**
   Globally mixed features have \(q_{\rm eff}\approx0\), while spectral oracle has \(q_{\rm eff}=1/2\) and better final risk under wide learning-rate tuning.

## Key positioning

The paper should not claim that Adam only helps when \(q_{\rm eff}>0\).  The correct claim is more precise:

```text
Adam/RMSProp can improve finite-time constants even when q_eff≈0,
but they improve the spectral scaling exponent only when second moments are
spectrally informative.
```

## Remaining work before submission

1. Convert notes into a coherent theorem/proof manuscript.
2. Produce polished figures from existing CSV files.
3. Decide whether to keep all experiments in the main paper or put some in appendix.
4. Add related work and comparison to optimizer-dependent scaling, Chinchilla-style allocation, AdamW scaling, and random-feature scaling laws.
5. Write limitations: Gaussian linear model, coordinatewise adaptivity, and finite-time risk constants versus asymptotic exponents.

No additional toy theorem is essential before drafting.
