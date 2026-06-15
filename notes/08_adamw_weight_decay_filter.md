# 08. AdamW decoupled weight decay is a separate shrinkage filter

The previous note proves that Adam first-moment momentum does not change the spectral exponent created by Adam/RMSProp's second-moment preconditioner.  Adam without weight decay has the same exponent-level learned-mode count as RMSProp, with emergent

\[
    q_{\rm eff}=1/2.
\]

This note adds the last optimizer ingredient in AdamW: **decoupled weight decay**.  The conclusion is that AdamW is Adam's damped \(q=1/2\) preconditioner plus a separate multiplicative shrinkage filter.  Weight decay creates a second cutoff

\[
    \mu_i \gtrsim \lambda_{\rm wd},
\]

in addition to the finite-time cutoff

\[
    N_{\rm eff}\gamma\mu_i\gtrsim 1.
\]

Thus the effective learned-mode count is controlled by

\[
\boxed{
    \mu_i\gtrsim \max\{n^{-1},\lambda_{\rm wd}\}.
}
\]

For Adam's damped \(q=1/2\) spectrum, this means weight decay replaces the effective training horizon \(n\) by

\[
\boxed{
    T_{\rm eff}=\min\{n,\lambda_{\rm wd}^{-1}\},
}
\]

up to the scalar normalization described below.

## 1. Fixed-preconditioner AdamW without momentum

Freeze a diagonal preconditioner

\[
    A=\operatorname{diag}(a_i),
\]

and define the effective eigenvalues

\[
    \mu_i=a_i\lambda_i.
\]

The decoupled-weight-decay update is

\[
    w_{t+1}=(1-\gamma\delta)w_t-\gamma A g_t,
\]

where \(\delta=\lambda_{\rm wd}\).  In population dynamics,

\[
    g_t=H(w_t-w^\star).
\]

Let \(e_t=w_t-w^\star\).  Then coordinate \(i\) satisfies

\[
\begin{aligned}
    e_{t+1,i}
    &= (1-\gamma\delta)(e_{t,i}+w_i^\star)
       -\gamma a_i\lambda_i e_{t,i}-w_i^\star \\
    &= \left(1-\gamma(\mu_i+\delta)\right)e_{t,i}
       -\gamma\delta w_i^\star .
\end{aligned}
\]

This is the key difference from coupled \(\ell_2\) regularization: the shrinkage term is isotropic in the original parameter coordinates, while the gradient term is preconditioned by \(A\).

## 2. Exact deterministic bias filter

Set

\[
    r_i=1-\gamma(\mu_i+\delta),
\]

and assume scalar stability, \(|r_i|<1\).  With initialization \(w_0=0\), so \(e_{0,i}=-w_i^\star\), the recursion solves to

\[
\boxed{
    e_{N,i}
    =
    -\left[
        \frac{\delta}{\mu_i+\delta}
        +
        \frac{\mu_i}{\mu_i+\delta}r_i^N
    \right]w_i^\star .
}
\]

Therefore the coordinate bias contribution to excess risk is

\[
\boxed{
    \lambda_i e_{N,i}^2
    =
    s_i
    \left[
        \frac{\delta}{\mu_i+\delta}
        +
        \frac{\mu_i}{\mu_i+\delta}r_i^N
    \right]^2,
}
\]

where

\[
    s_i=\lambda_i(w_i^\star)^2.
\]

The finite-time residual is

\[
    \frac{\mu_i}{\mu_i+\delta}r_i^N,
\]

and the irreducible shrinkage bias caused by decoupled weight decay is

\[
    \frac{\delta}{\mu_i+\delta}.
\]

## 3. AdamW bias and variance filters

Use the effective one-pass time

\[
    n=N_{\rm eff}\gamma.
\]

The finite-time residual is represented by a spectral filter comparable to

\[
    \frac{1}{1+n(\mu_i+\delta)}.
\]

Since the two bias pieces are nonnegative, their squared sum is equivalent up to constants to the sum of squares.  Thus the AdamW bias filter is

\[
\boxed{
    B_{\rm wd}(n,\delta,M)
    =
    \sum_{i\le M}s_i
    \left[
        \left(\frac{\delta}{\mu_i+\delta}\right)^2
        +
        \left(\frac{\mu_i}{\mu_i+\delta}\right)^2
        \frac{1}{1+n(\mu_i+\delta)}
    \right].
}
\]

When \(\delta=0\), this reduces to

\[
    \sum_{i\le M}\frac{s_i}{1+n\mu_i}.
\]

When \(n\to\infty\), it converges to the shrinkage floor

\[
    \sum_{i\le M}s_i\left(\frac{\delta}{\mu_i+\delta}\right)^2.
\]

So fixed positive weight decay creates a nonzero optimization floor unless the spectral threshold \(\delta\) moves to zero with scale.

For variance, noise is injected through the same preconditioned gradient channel, so its amplitude is controlled by \(\mu_i\), while the closed-loop contraction is controlled by \(\mu_i+\delta\).  The finite-time variance filter is

\[
\boxed{
    V_{\rm wd}(n,\delta,M)
    =
    \frac1{N_{\rm eff}}
    \sum_{i\le M}
    \left(\frac{\mu_i}{\mu_i+\delta}\right)^2
    \min\{1,n^2(\mu_i+\delta)^2\}.
}
\]

This has the correct limits.  If \(\delta=0\), it becomes the no-decay variance filter.  If \(n(\mu_i+\delta)\ll1\), the contribution is \(N_{\rm eff}^{-1}n^2\mu_i^2\).  If \(n(\mu_i+\delta)\gg1\), the contribution is \(N_{\rm eff}^{-1}(\mu_i/(\mu_i+\delta))^2\).

## 4. Fixed-preconditioner AdamW scaling theorem

Under the same stability and one-pass assumptions as the previous fixed-preconditioner theorem, preconditioned SGD with decoupled weight decay satisfies

\[
\boxed{
    \mathbb E R_M(w_N)-\sigma^2
    \asymp
    \sum_{i>M}s_i
    +B_{\rm wd}(n,\delta,M)
    +V_{\rm wd}(n,\delta,M).
}
\]

With first-moment momentum \(\beta_1<1\), the same law holds up to constants depending on \(\beta_1\) and the stability margin, by the momentum-filter theorem in `07_adam_momentum_filter.md`.

The effective learned-mode cutoff is

\[
\boxed{
    \mu_i\gtrsim \max\{n^{-1},\delta\}.
}
\]

Equivalently,

\[
\boxed{
    K_{\rm wd}(n,\delta)=\#\{i:\mu_i\gtrsim \max\{n^{-1},\delta\}\}.
}
\]

So decoupled weight decay imposes a regularization ceiling:

\[
    K_{\rm wd}(n,\delta)\le K(\delta^{-1}).
\]

## 5. AdamW with tracked second moments

Now insert Adam's RMSProp-like second moment.  By the raw-EMA theorem,

\[
    v_{t,i}\asymp d_t\lambda_i.
\]

Assume \(d_t\) stays in a constant band and choose a reference \(d_\circ\).  Set

\[
    \rho_\circ=\epsilon/d_\circ.
\]

Then

\[
    (v_{t,i}+\epsilon)^{-1/2}
    \asymp
    d_\circ^{-1/2}(\lambda_i+\rho_\circ)^{-1/2}.
\]

The population gradient channel is

\[
    d_\circ^{-1/2}\lambda_i(\lambda_i+\rho_\circ)^{-1/2}.
\]

Absorb the scalar \(d_\circ^{-1/2}\) into the effective time.  Define

\[
    \mu_i^\circ=\lambda_i(\lambda_i+\rho_\circ)^{-1/2},
\]

\[
    n=N_{\rm eff}\gamma d_\circ^{-1/2},
\]

and the weight-decay threshold in the same units as \(\mu_i^\circ\):

\[
\boxed{
    \delta_\circ=\lambda_{\rm wd}d_\circ^{1/2}.
}
\]

Then AdamW is governed by the fixed-preconditioner AdamW filters with \(\mu_i=\mu_i^\circ\), \(n=N_{\rm eff}\gamma d_\circ^{-1/2}\), and \(\delta=\delta_\circ\):

\[
\boxed{
    \mathbb E R_M(w_N)-\sigma^2
    \asymp_{\beta_1}
    \sum_{i>M}s_i
    +B_{\rho_\circ,{\rm wd}}(n,\delta_\circ,M)
    +V_{\rho_\circ,{\rm wd}}(n,\delta_\circ,M).
}
\]

Here

\[
\boxed{
    B_{\rho,{\rm wd}}(n,\delta,M)
    =
    \sum_{i\le M}s_i
    \left[
        \left(\frac{\delta}{\mu_i^\rho+\delta}\right)^2
        +
        \left(\frac{\mu_i^\rho}{\mu_i^\rho+\delta}\right)^2
        \frac{1}{1+n(\mu_i^\rho+\delta)}
    \right],
}
\]

\[
\boxed{
    V_{\rho,{\rm wd}}(n,\delta,M)
    =
    \frac1{N_{\rm eff}}
    \sum_{i\le M}
    \left(\frac{\mu_i^\rho}{\mu_i^\rho+\delta}\right)^2
    \min\{1,n^2(\mu_i^\rho+\delta)^2\},
}
\]

and

\[
    \mu_i^\rho=\lambda_i(\lambda_i+\rho)^{-1/2}.
\]

## 6. Power-law learned-mode count

Assume

\[
    \lambda_i\asymp i^{-a},
    \qquad a>1.
\]

For the Adam/RMSProp second-moment spectrum,

\[
    \mu_i^\rho=\lambda_i(\lambda_i+\rho)^{-1/2}.
\]

The damping knee occurs at

\[
    i_\rho\asymp \rho^{-1/a},
\]

and

\[
    \mu_i^\rho
    \asymp
    \begin{cases}
        i^{-a/2}, & i\lesssim \rho^{-1/a},\\[3pt]
        \rho^{-1/2}i^{-a}, & i\gtrsim \rho^{-1/a}.
    \end{cases}
\]

AdamW's active threshold is

\[
    \theta=\max\{n^{-1},\delta\}.
\]

Equivalently define

\[
\boxed{
    T_{\rm eff}=\theta^{-1}=\min\{n,\delta^{-1}\}.
}
\]

Then the learned-mode count is

\[
\boxed{
K_{\rho,{\rm wd}}(n,\delta)
\asymp
K_{\rho,1/2}(T_{\rm eff})
\asymp
\begin{cases}
    T_{\rm eff}^{2/a}, & T_{\rm eff}\lesssim \rho^{-1/2},\\[3pt]
    \rho^{-1/(2a)}T_{\rm eff}^{1/a}, & T_{\rm eff}\gtrsim \rho^{-1/2}.
\end{cases}
}
\]

Thus weight decay does not change the local spectral exponent while \(n\lesssim\delta^{-1}\).  Instead it saturates the effective horizon once \(n\gtrsim\delta^{-1}\).  More data or more steps cannot increase the learned-mode count beyond

\[
    K_{\rho,1/2}(\delta^{-1})
\]

unless weight decay decreases with scale.

## 7. Simplified source-condition scaling

Assume the source condition

\[
    s_i\asymp i^{-b},
    \qquad b>1.
\]

In the clean range where the variance effective dimension is comparable to the learned-mode count, the filter law simplifies to

\[
\boxed{
    \mathbb E R_M(w_N)-\sigma^2
    \asymp
    M^{1-b}
    +K_{\rho,{\rm wd}}(n,\delta)^{1-b}
    +\frac{K_{\rho,{\rm wd}}(n,\delta)}{N_{\rm eff}}.
}
\]

Without weight decay, the active Adam-like phase has

\[
    K_{\rho,1/2}(n)\asymp n^{2/a}
\]

before the damping knee.  With fixed positive weight decay, this becomes

\[
    K_{\rho,{\rm wd}}(n,\delta)
    \asymp
    K_{\rho,1/2}(\min\{n,\delta^{-1}\}).
\]

So fixed \(\lambda_{\rm wd}\) creates a scale-dependent floor.  To preserve the Adam/RMSProp compute-optimal exponent at growing scale, the effective decay must satisfy

\[
\boxed{
    \delta_\circ\lesssim n_\star^{-1},
}
\]

where \(n_\star\) is the compute-optimal training horizon.  In the active pre-damping phase with effective spectral exponent \(\alpha=a/2\), the no-decay allocation gives

\[
    n_\star\asymp C^{\alpha/(\alpha+1)}.
\]

Therefore AdamW must scale roughly as

\[
\boxed{
    \delta_\circ
    \lesssim
    C^{-\alpha/(\alpha+1)},
    \qquad \alpha=a/2,
}
\]

if one wants to avoid weight-decay-induced saturation of the scaling law.

## 8. Interpretation and remaining work

The optimizer decomposition is now

\[
\text{AdamW}
=
\underbrace{\text{second-moment spectral preconditioner}}_{q_{\rm eff}=1/2}
+
\underbrace{\text{first-moment temporal filter}}_{\text{constants/stability}}
+
\underbrace{\text{decoupled shrinkage filter}}_{\mu_i\gtrsim\lambda_{\rm wd}}.
\]

The second-moment term is responsible for the optimizer-dependent spectral exponent.  The first-moment term changes constants and stability.  Decoupled weight decay imposes a regularization ceiling that must scale down with compute if the asymptotic Adam/RMSProp exponent is to persist.

The main optimizer-theory chain is now complete for diagonal Gaussian linear regression:

\[
\text{RMSProp/Adam/AdamW}
\Rightarrow
q_{\rm eff}=1/2
\Rightarrow
\text{damped learned-mode count}
\Rightarrow
\text{optimizer-dependent scaling law}.
\]

The next project steps should be to consolidate the notes into a manuscript-style theorem stack, add matching lower-bound statements for every filter law, run exponent-level experiments over \((a,b,\rho,\lambda_{\rm wd},M,N)\), add coordinate-alignment/random-rotation experiments, and move from diagonal Gaussian regression to Gaussian-sketched/random-feature regression.
