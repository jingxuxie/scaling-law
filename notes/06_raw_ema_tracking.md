# 06. Raw EMA tracking for ordinary online RMSProp

This note removes the main conditional assumption from `05_online_rmsprop_sandwich.md`.
The sandwich theorem assumed that online second moments track their conditional means.
Here we prove that the ordinary raw RMSProp exponential moving average has this form,
up to the natural EMA lag, under an effective-window condition.

The main conclusion is

\[
    v_{t,i} \asymp d_t\lambda_i,
\]

where \(d_t\) is the EMA of the scalar residual scale

\[
    c_t=\sigma^2+\|w_t-w^\star\|_H^2.
\]

Therefore

\[
    (v_{t,i}+\epsilon)^{-1/2}
    \asymp
    d_t^{-1/2}\left(\lambda_i+\frac{\epsilon}{d_t}\right)^{-1/2},
\]

so ordinary online RMSProp tracks a time-lagged damped spectral preconditioner
with emergent exponent \(q_{\rm eff}=1/2\).

## Setup

Work in diagonal Gaussian linear regression:

\[
    x_t\sim \mathcal N(0,H),
    \qquad H=\operatorname{diag}(\lambda_1,\ldots,\lambda_M),
\]

\[
    y_t=\langle x_t,w^\star\rangle+\xi_t,
    \qquad \xi_t\sim\mathcal N(0,\sigma^2).
\]

Let \(\mathcal F_t\) be the history before drawing \((x_t,y_t)\), and assume
\(w_t\) is \(\mathcal F_t\)-measurable.  Define

\[
    e_t=w_t-w^\star,
    \qquad
    c_t=\sigma^2+\|e_t\|_H^2 .
\]

The stochastic gradient is

\[
    g_t=(\langle x_t,e_t\rangle-\xi_t)x_t .
\]

Let

\[
    Z_{t,i}=g_{t,i}^2,
    \qquad
    \nu_{t,i}=\mathbb E[Z_{t,i}\mid\mathcal F_t].
\]

Raw RMSProp uses

\[
    v_{t+1,i}=\beta v_{t,i}+(1-\beta)Z_{t,i},
    \qquad 0<\beta<1.
\]

Set

\[
    W_\beta=(1-\beta)^{-1}.
\]

## Lemma 1: conditional gradient second moments are spectral

For every predictable iterate,

\[
\boxed{
    \nu_{t,i}
    =
    \lambda_i c_t+2\lambda_i^2 e_{t,i}^2 .
}
\]

Consequently,

\[
\boxed{
    \lambda_i c_t
    \le
    \nu_{t,i}
    \le
    3\lambda_i c_t .
}
\]

### Proof

Conditionally on \(\mathcal F_t\), define

\[
    U_t=\langle x_t,e_t\rangle-\xi_t,
    \qquad X_{t,i}=x_{t,i}.
\]

The pair \((U_t,X_{t,i})\) is centered Gaussian with

\[
    \mathbb E[U_t^2\mid\mathcal F_t]=c_t,
    \qquad
    \mathbb E[X_{t,i}^2]=\lambda_i,
    \qquad
    \mathbb E[U_tX_{t,i}\mid\mathcal F_t]=\lambda_i e_{t,i}.
\]

Since \(Z_{t,i}=U_t^2X_{t,i}^2\), Isserlis' identity gives

\[
    \mathbb E[U_t^2X_{t,i}^2\mid\mathcal F_t]
    =
    \mathbb E[U_t^2\mid\mathcal F_t]\mathbb E[X_{t,i}^2]
    +2\mathbb E[U_tX_{t,i}\mid\mathcal F_t]^2 .
\]

This proves the identity.  The lower bound is immediate.  For the upper bound,
use

\[
    \lambda_i e_{t,i}^2\le \|e_t\|_H^2\le c_t.
\]

Then \(2\lambda_i^2e_{t,i}^2\le 2\lambda_i c_t\), and hence
\(\nu_{t,i}\le 3\lambda_i c_t\).

## Lemma 2: squared gradients have product-Gaussian tails

There are universal constants \(C,c>0\) such that, conditionally on
\(\mathcal F_t\),

\[
\boxed{
    \left\|\frac{Z_{t,i}}{\nu_{t,i}}\right\|_{\psi_{1/2}}
    \le C,
}
\]

or equivalently

\[
    \mathbb P\left(
        Z_{t,i}\ge u\nu_{t,i}\mid\mathcal F_t
    \right)
    \le
    2\exp(-c\sqrt u),
    \qquad u\ge 1.
\]

Also,

\[
    \mathbb E[(Z_{t,i}-\nu_{t,i})^2\mid\mathcal F_t]
    \le C\nu_{t,i}^2 .
\]

### Proof sketch

Conditionally on \(\mathcal F_t\), \(g_{t,i}=U_tX_{t,i}\) is a degree-two
Gaussian chaos.  Degree-two Gaussian chaos is sub-exponential after
normalization by its standard deviation; squaring it gives a sub-Weibull-\(1/2\)
random variable.  The fourth-moment bound follows from hypercontractivity, or
from Isserlis' formula for \(\mathbb E[U_t^4X_{t,i}^4]\).  The ratio

\[
    \frac{\mathbb E[U_t^4X_{t,i}^4]}{\mathbb E[U_t^2X_{t,i}^2]^2}
\]

is uniformly bounded over all correlations between \(U_t\) and \(X_{t,i}\).

## The population EMA target

With \(v_{0,i}=0\), unrolling the EMA gives

\[
    v_{t,i}=\sum_{s=0}^{t-1}\alpha_{t,s}Z_{s,i},
    \qquad
    \alpha_{t,s}=(1-\beta)\beta^{t-1-s}.
\]

Define the predictable target

\[
    m_{t,i}=\sum_{s=0}^{t-1}\alpha_{t,s}\nu_{s,i}
\]

and the scalar residual-scale EMA

\[
    d_t=\sum_{s=0}^{t-1}\alpha_{t,s}c_s.
\]

By Lemma 1,

\[
\boxed{
    \lambda_i d_t
    \le
    m_{t,i}
    \le
    3\lambda_i d_t .
}
\]

This is the key point: raw RMSProp tracks \(d_t\lambda_i\), not exactly
\(c_t\lambda_i\).  Under slow drift, \(d_t\asymp c_t\).

## Effective-window leverage

Define

\[
    \Lambda_{t,i}
    :=
    \max_{0\le s<t}
    \frac{\alpha_{t,s}\nu_{s,i}}{m_{t,i}}.
\]

This is the maximum fractional contribution of one sample to the EMA target.  If
\(\nu_{s,i}\) is nearly constant over the active EMA window, then
\(\Lambda_{t,i}\asymp 1-\beta=W_\beta^{-1}\).  If the residual scale drifts too
quickly, the leverage becomes larger and raw EMA concentration weakens.

## Theorem 1: raw EMA tracks its predictable target

Fix \(M,N\ge2\), \(\delta\in(0,1)\), and \(\tau\in(0,1/2)\).  Suppose that for
all \(i\le M\) and \(1\le t\le N\),

\[
    \Lambda_{t,i}\le \Lambda.
\]

There is a universal constant \(c>0\) such that if

\[
\boxed{
    \Lambda
    \le
    \frac{c\tau^2}{\log^2(2MN/\delta)},
}
\]

then, with probability at least \(1-\delta\), simultaneously for all
\(i\le M\) and all \(1\le t\le N\),

\[
\boxed{
    |v_{t,i}-m_{t,i}|\le \tau m_{t,i}.
}
\]

Consequently,

\[
\boxed{
    (1-\tau)\lambda_i d_t
    \le
    v_{t,i}
    \le
    3(1+\tau)\lambda_i d_t .
}
\]

### Proof sketch

Write

\[
    v_{t,i}-m_{t,i}
    =
    \sum_{s=0}^{t-1}\alpha_{t,s}(Z_{s,i}-\nu_{s,i}).
\]

The summands are martingale differences.  By Lemma 2,

\[
    \left\|\alpha_{t,s}(Z_{s,i}-\nu_{s,i})\right\|_{\psi_{1/2}\mid\mathcal F_s}
    \le
    C\alpha_{t,s}\nu_{s,i}
    \le
    C\Lambda m_{t,i}.
\]

Moreover,

\[
    \sum_{s=0}^{t-1}
    \mathbb E[\alpha_{t,s}^2(Z_{s,i}-\nu_{s,i})^2\mid\mathcal F_s]
    \le
    C\sum_{s=0}^{t-1}\alpha_{t,s}^2\nu_{s,i}^2
    \le
    C\Lambda m_{t,i}^2.
\]

Truncate the martingale differences at level

\[
    b=c_0\tau m_{t,i}/\log(2MN/\delta).
\]

The sub-Weibull-\(1/2\) tail bound makes the probability of a truncation error
at most \(\delta/(2MN)\) under the stated leverage condition.  Conditional
Bernstein/Freedman applied to the truncated martingale differences gives

\[
    \mathbb P\left(|v_{t,i}-m_{t,i}|>\tau m_{t,i}\right)
    \le
    \delta/(MN).
\]

Union bounding over all \((t,i)\) proves the claim.  The extra
\(\log^2(2MN/\delta)\) factor is the price of using raw EMA on product-Gaussian
squared-gradient noise; the median-of-means estimator in the frozen bridge note
needs only a logarithmic number of blocks.

## Corollary 1: slow drift implies small leverage

Suppose that for every \(t\), the residual scale is comparable over the active
EMA window:

\[
    \chi^{-1}c_t\le c_s\le \chi c_t
\]

for all \(s<t\) with

\[
    t-s\le L_\eta:=\lceil W_\beta\log(1/\eta)\rceil,
\]

and the older EMA tail contributes at most an \(\eta\)-fraction of \(d_t\).  Then

\[
    \Lambda_{t,i}\le C\chi(1-\beta)+C\eta.
\]

Thus raw EMA tracking holds when

\[
\boxed{
    W_\beta
    \gtrsim
    \chi\tau^{-2}\log^2(2MN/\delta)
}
\]

up to constants and the negligible tail parameter \(\eta\).

## Theorem 2: ordinary online RMSProp has damped `q=1/2` shape

On the event of Theorem 1,

\[
    v_{t,i}\asymp \lambda_i d_t.
\]

Therefore the RMSProp preconditioner

\[
    A_{t,i}=(v_{t,i}+\epsilon)^{-1/2}
\]

satisfies

\[
\boxed{
    A_{t,i}
    \asymp
    d_t^{-1/2}\left(\lambda_i+\frac{\epsilon}{d_t}\right)^{-1/2}.
}
\]

With

\[
    \rho_t^{\rm ema}=\epsilon/d_t,
\]

ordinary online RMSProp tracks

\[
\boxed{
    A_t\asymp d_t^{-1/2}(H+\rho_t^{\rm ema}I)^{-1/2}.
}
\]

The scalar \(d_t^{-1/2}\) rescales the learning rate; the spectral shape has
emergent exponent

\[
\boxed{q_{\rm eff}=1/2.}
\]

## Corollary 2: risk law for ordinary raw-EMA RMSProp

Assume Theorem 1, and suppose the EMA residual scale stays in a constant band:

\[
    0<d_{\min}\le d_t\le d_{\max}<\infty.
\]

Choose \(d_\circ\in[d_{\min},d_{\max}]\), set

\[
    \rho_\circ=\epsilon/d_\circ,
    \qquad
    \mu_i^\circ=\lambda_i(\lambda_i+\rho_\circ)^{-1/2},
\]

and

\[
    n=N_{\rm eff}\gamma d_\circ^{-1/2}.
\]

Then ordinary online RMSProp is sandwiched between constant multiples of frozen
damped \(q=1/2\) spectral preconditioning.  Hence the same filters govern the
risk:

\[
\boxed{
    \mathbb E R_M(w_N)-\sigma^2
    \asymp
    \sum_{i>M}s_i
    +B_{\rho_\circ,1/2}(n,M)
    +V_{\rho_\circ,1/2}(n,M),
}
\]

where

\[
    B_{\rho_\circ,1/2}(n,M)
    =
    \sum_{i\le M}\frac{s_i}{1+n\mu_i^\circ},
\]

and

\[
    V_{\rho_\circ,1/2}(n,M)
    =
    \frac1{N_{\rm eff}}
    \sum_{i\le M}\min\{1,n^2(\mu_i^\circ)^2\}.
\]

For \(\lambda_i\asymp i^{-a}\), the learned-mode count is

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

Thus raw-EMA RMSProp has the same optimizer-dependent scaling law as frozen and
population RMSProp, provided its effective window is large enough to average the
product-Gaussian squared-gradient noise and short enough that the residual scale
does not drift too far.

## What this closes

The previous online sandwich theorem was conditional on a tracking event.  This
note proves that ordinary raw RMSProp satisfies the event in the form

\[
    v_{t,i}\asymp d_t\lambda_i.
\]

Under slow drift, \(d_t\asymp c_t\), so the chain is now

\[
\text{raw RMSProp EMA}
\Rightarrow
v_{t,i}\asymp d_t\lambda_i
\Rightarrow
A_{t,i}\asymp d_t^{-1/2}(\lambda_i+\epsilon/d_t)^{-1/2}
\Rightarrow
q_{\rm eff}=1/2
\Rightarrow
\text{damped spectral scaling law}.
\]

## Next theorem target

The next theorem should add Adam first-moment momentum.  The target is to show
that \(\beta_1>0\) leaves the exponent-level spectral shape unchanged when the
first-moment window is shorter than or comparable to the second-moment window,
while changing constants and the stability region.  AdamW decoupled weight decay
should then be added as a separate spectral shrinkage filter.
