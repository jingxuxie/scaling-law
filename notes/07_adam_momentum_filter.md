# 07. Adam first-moment momentum is a temporal filter, not a new spectral exponent

The previous notes prove that ordinary RMSProp's second-moment EMA tracks

\[
    v_{t,i}\asymp d_t\lambda_i,
\]

and therefore its coordinate preconditioner has the damped spectral shape

\[
    (v_{t,i}+\epsilon)^{-1/2}
    \asymp
    d_t^{-1/2}\left(\lambda_i+\frac{\epsilon}{d_t}\right)^{-1/2}.
\]

Adam adds a first-moment EMA

\[
    m_{t+1}=\beta_1m_t+(1-\beta_1)g_t.
\]

This note proves the next step: first-moment momentum changes constants and the
stability window, but it does **not** change the exponent-level spectral shape.
Under the RMSProp tracking event, Adam has the same learned-mode count as
RMSProp:

\[
K_{\rho,1/2}(n)
\asymp
\begin{cases}
    n^{2/a}, & n\lesssim \rho^{-1/2},\\[3pt]
    \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2}.
\end{cases}
\]

Thus Adam's first moment is a temporal/momentum filter on top of the same
\(q_{\rm eff}=1/2\) second-moment preconditioner.

## 1. Fixed-preconditioner momentum model

First freeze the diagonal preconditioner

\[
    A=\operatorname{diag}(a_i).
\]

The effective covariance eigenvalues are

\[
    \mu_i=a_i\lambda_i.
\]

For the Adam-like first-moment update, ignoring bias correction for now,

\[
    m_{t+1}=\beta m_t+(1-\beta)g_t,
    \qquad
    w_{t+1}=w_t-\gamma A m_{t+1}.
\]

Bias correction multiplies \(m_t\) by the scalar \((1-\beta^t)^{-1}\).  After
\(O((1-\beta)^{-1}\log N)\) warmup steps this scalar is a constant, and before
that it can be absorbed into constants.  It does not change any spectral
exponent.

In the population/noiseless coordinate dynamics,

\[
    g_{t,i}=\lambda_i e_{t,i},
    \qquad e_t=w_t-w^\star.
\]

The coordinate update is

\[
    m_{t+1,i}=\beta m_{t,i}+(1-\beta)\lambda_i e_{t,i},
\]

\[
    e_{t+1,i}=e_{t,i}-\gamma a_i m_{t+1,i}.
\]

Let

\[
    x_i=\gamma\mu_i=\gamma a_i\lambda_i.
\]

## 2. Momentum bias polynomial

Define the momentum transfer polynomial \(p_t^{(\beta)}(x)\) by

\[
    e_{t,i}=p_t^{(\beta)}(x_i)e_{0,i}
\]

for the population dynamics initialized with \(m_{0,i}=0\).  Then

\[
    p_{-1}^{(\beta)}(x)=p_0^{(\beta)}(x)=1,
\]

and

\[
\boxed{
    p_{t+1}^{(\beta)}(x)
    =
    \left(1+\beta-(1-\beta)x\right)p_t^{(\beta)}(x)
    -
    \beta p_{t-1}^{(\beta)}(x).
}
\]

### Proof

Since

\[
    e_t-e_{t-1}=-\gamma a_i m_{t,i},
\]

we have

\[
\begin{aligned}
    e_{t+1,i}
    &=e_{t,i}-\gamma a_i\left[\beta m_{t,i}+(1-\beta)\lambda_i e_{t,i}\right] \\
    &=e_{t,i}+\beta(e_{t,i}-e_{t-1,i})-(1-\beta)x_i e_{t,i} \\
    &=\left(1+\beta-(1-\beta)x_i\right)e_{t,i}-\beta e_{t-1,i}.
\end{aligned}
\]

This proves the recursion.

## 3. The low-eigenvalue learning time is unchanged

The characteristic polynomial is

\[
    r^2-\left(1+\beta-(1-\beta)x\right)r+\beta=0.
\]

For small \(x\), one root is near \(1\), and the other is near \(\beta\).  A
Taylor expansion of the root near one gives

\[
    r_+(x)=1-x+O_\beta(x^2),
\]

while

\[
    r_-(x)=\beta+O_\beta(x).
\]

Therefore the slow mode decays on the time scale

\[
    t\asymp x^{-1}=\frac{1}{\gamma\mu_i},
\]

exactly as in non-momentum preconditioned SGD.  Momentum does not replace
\(\mu_i\) by \(\mu_i^\theta\) or by any other spectral power.

A convenient quantitative form is the following.

## Lemma 1: bias cutoff with first-moment momentum

Fix \(\beta\in[0,\beta_{\max}]\) with \(\beta_{\max}<1\).  Assume the scalar step
is in a stable compact set

\[
    0\le x\le x_{\max}<\frac{2(1+\beta)}{1-\beta}.
\]

Then there are constants \(c_\beta,C_\beta>0\) such that, for all \(N\ge1\),

\[
\boxed{
    |p_N^{(\beta)}(x)|^2
    \le
    \frac{C_\beta}{1+Nx}
    +C_\beta\beta^N .
}
\]

Moreover, for \(Nx\le c_\beta\),

\[
\boxed{
    |p_N^{(\beta)}(x)|^2\ge c_\beta .
}
\]

Consequently, the population bias cutoff is

\[
    N\gamma\mu_i\asymp 1.
\]

### Proof sketch

The exact solution is a linear combination of the two characteristic roots.
The root near one satisfies

\[
    1-C_\beta x\le r_+(x)\le 1-c_\beta x
\]

on any stable compact interval away from the stability boundary, while the other
root has modulus at most a constant strictly smaller than one unless the roots
are a complex pair, in which case both have modulus \(\sqrt\beta\).  Hence

\[
    |p_N^{(\beta)}(x)|
    \le
    C_\beta\left(e^{-c_\beta Nx}+\beta^{N/2}\right).
\]

Using \(e^{-u}\le C/(1+u)\) gives the upper bound.  If \(Nx\le c_\beta\), then
the recursion is a perturbation of \(p_t\equiv1\) over \(N\) steps, so
\(|p_N^{(\beta)}(x)|\ge c_\beta\).

## 4. Bias filter for fixed preconditioner Adam

Let

\[
    s_i=\lambda_i(w_i^\star)^2.
\]

The population bias contribution satisfies, up to \(\beta\)-dependent constants,

\[
\boxed{
    B_\beta(N,\gamma,M)
    \asymp_\beta
    \sum_{i\le M}\frac{s_i}{1+N\gamma\mu_i}.
}
\]

The lower bound follows from the unlearned coordinates
\(N\gamma\mu_i\lesssim1\).  The upper bound follows from Lemma 1 and the same
spectral summation used for non-momentum preconditioned SGD.  The extra
\(\beta^N\sum_i s_i\) startup term is negligible after the usual momentum warmup
or is absorbed into constants for finite \(\beta<1\).

Thus first-moment momentum leaves the bias filter's spectral cutoff unchanged.

## 5. Variance filter is changed only by constants

Now add martingale gradient noise:

\[
    g_{t,i}=\lambda_i e_{t,i}+\zeta_{t,i},
    \qquad
    \mathbb E[\zeta_{t,i}\mid\mathcal F_t]=0.
\]

For a fixed coordinate, the noise-to-error transfer function of the first-moment
EMA is the no-momentum transfer function multiplied by the stable scalar filter

\[
    H_\beta(z)=\frac{1-\beta}{1-\beta z^{-1}}.
\]

This filter has DC gain one and satisfies

\[
    \sup_{|z|=1}|H_\beta(z)|\le1.
\]

On time scales longer than the first-moment window

\[
    W_1=(1-\beta)^{-1},
\]

it therefore preserves the low-frequency response that determines the learned
mode count.  For finite \(\beta<1\), the full closed-loop impulse response has
\(\ell_2\) norm comparable to the no-momentum response on every stable compact
set of stepsizes.  Hence

\[
\boxed{
    V_\beta(N,\gamma,M)
    \asymp_\beta
    \frac1{N_{\rm eff}}
    \sum_{i\le M}\min\{1,(N_{\rm eff}\gamma\mu_i)^2\}.
}
\]

Momentum may improve or worsen the constant depending on \(\beta\) and the
stepsize margin, but it does not change the dependence on \(\mu_i\).

## 6. Fixed-preconditioner Adam-momentum scaling theorem

For a fixed diagonal preconditioner \(A\), define

\[
    \mu_i=a_i\lambda_i,
    \qquad
    n=N_{\rm eff}\gamma.
\]

Under the stable-stepsize and finite-momentum assumptions above, one-pass
preconditioned SGD with Adam first-moment momentum satisfies

\[
\boxed{
    \mathbb E R_M(w_N)-\sigma^2
    \asymp_\beta
    \sum_{i>M}s_i
    +
    \sum_{i\le M}\frac{s_i}{1+n\mu_i}
    +
    \frac1{N_{\rm eff}}
    \sum_{i\le M}\min\{1,n^2\mu_i^2\}.
}
\]

This is the same filter law as fixed-preconditioner SGD, up to constants that
depend on \(\beta_1\) and the stability margin.  Therefore the learned-mode count
is still

\[
    K(n)=\#\{i:\mu_i\gtrsim n^{-1}\}.
\]

## 7. Adam inherits the RMSProp damped spectral shape

Now return to Adam.  The update is

\[
    m_{t+1}=\beta_1m_t+(1-\beta_1)g_t,
\]

\[
    v_{t+1}=\beta_2v_t+(1-\beta_2)g_t^2,
\]

\[
    w_{t+1}=w_t-\gamma (v_{t+1}+\epsilon)^{-1/2}m_{t+1}.
\]

Assume the raw-EMA tracking theorem from `06_raw_ema_tracking.md` holds:

\[
    v_{t,i}\asymp d_t\lambda_i.
\]

Assume also that \(d_t\) stays in a constant band and is slowly varying over the
first-moment window \(W_1=(1-\beta_1)^{-1}\).  This is the natural condition
ensuring that the first-moment EMA is averaging gradients under essentially the
same preconditioner.  A sufficient practical condition is

\[
    W_1\lesssim W_2=(1-\beta_2)^{-1}
\]

plus the slow-drift condition used in the raw-EMA theorem.

Choose a reference scale \(d_\circ\) in the band and set

\[
    \rho_\circ=\epsilon/d_\circ.
\]

Then

\[
    (v_{t,i}+\epsilon)^{-1/2}
    \asymp
    d_\circ^{-1/2}(\lambda_i+\rho_\circ)^{-1/2}.
\]

Therefore Adam is sandwiched between constant multiples of fixed
preconditioned momentum with effective eigenvalues

\[
    \mu_i^\circ
    =
    d_\circ^{-1/2}\lambda_i(\lambda_i+\rho_\circ)^{-1/2}.
\]

The scalar \(d_\circ^{-1/2}\) only rescales the effective learning rate.

## Theorem 2: Adam has the same scaling law as RMSProp

Let

\[
    \lambda_i\asymp i^{-a},
    \qquad
    s_i=\lambda_i(w_i^\star)^2\asymp i^{-b}.
\]

Under raw second-moment tracking, slow residual-scale drift, stable stepsizes,
and finite first-moment window, Adam satisfies

\[
\boxed{
    \mathbb E R_M(w_N)-\sigma^2
    \asymp_{\beta_1}
    \sum_{i>M}s_i
    +B_{\rho_\circ,1/2}(n,M)
    +V_{\rho_\circ,1/2}(n,M),
}
\]

where

\[
    n=N_{\rm eff}\gamma d_\circ^{-1/2},
\]

\[
    B_{\rho_\circ,1/2}(n,M)
    =
    \sum_{i\le M}\frac{s_i}{1+n\lambda_i(\lambda_i+\rho_\circ)^{-1/2}},
\]

and

\[
    V_{\rho_\circ,1/2}(n,M)
    =
    \frac1{N_{\rm eff}}
    \sum_{i\le M}
    \min\left\{1,n^2\lambda_i^2(\lambda_i+\rho_\circ)^{-1}\right\}.
\]

Thus Adam and RMSProp share the same exponent-level learned-mode count:

\[
\boxed{
K_{\rho_\circ,1/2}(n)
\asymp
\begin{cases}
    n^{2/a}, & n\lesssim \rho_\circ^{-1/2},\\[3pt]
    \rho_\circ^{-1/(2a)}n^{1/a}, & n\gtrsim \rho_\circ^{-1/2}.
\end{cases}
}
\]

The first-moment parameter \(\beta_1\) changes constants, transient behavior, and
stability margins, but not the spectral power \(q_{\rm eff}=1/2\).

## 8. Interpretation

The optimizer now decomposes into two mechanisms:

\[
\text{Adam}
=
\underbrace{\text{second-moment spectral preconditioner}}_{q_{\rm eff}=1/2}
+
\underbrace{\text{first-moment temporal filter}}_{\text{constants/stability}}.
\]

The second-moment EMA is responsible for the optimizer-dependent scaling
exponent.  The first-moment EMA changes how the algorithm moves through time but
keeps the same effective spectrum.

This gives the clean chain

\[
\text{raw Adam }v_t
\Rightarrow
v_{t,i}\asymp d_t\lambda_i
\Rightarrow
(v_{t,i}+\epsilon)^{-1/2}\asymp d_t^{-1/2}(\lambda_i+\epsilon/d_t)^{-1/2}
\Rightarrow
q_{\rm eff}=1/2,
\]

while

\[
\text{raw Adam }m_t
\Rightarrow
\text{temporal averaging with cutoff }N\gamma\mu_i\asymp1.
\]

## 9. What remains

This theorem brings the theory from RMSProp to Adam without weight decay.  The
main missing piece for AdamW is decoupled weight decay:

\[
    w_{t+1}
    =
    (1-\gamma\lambda_{\rm wd})w_t
    -
    \gamma(v_t+\epsilon)^{-1/2}m_t.
\]

Unlike coupled \(\ell_2\) regularization, decoupled AdamW shrinkage is not just a
change in the loss geometry.  The next theorem should treat it as a separate
spectral shrinkage filter, derive the modified bias term, and then optimize the
joint scaling law over \((M,N,\gamma,\epsilon,\lambda_{\rm wd})\).
