# Manuscript theorem stack: optimizer-dependent scaling laws

This file consolidates the proof notes into the theorem stack for a paper draft.
The current scope is diagonal Gaussian linear regression.  The intended paper
claim is not that diagonal regression alone is the final destination, but that it
is the rigorous core mechanism explaining why Adam/RMSProp/AdamW can have
optimizer-dependent scaling exponents.

## Model and source condition

Let

\[
    x\sim\mathcal N(0,H),
    \qquad
    H=\operatorname{diag}(\lambda_i),
    \qquad
    \lambda_i\asymp i^{-a},
    \quad a>1.
\]

Labels are

\[
    y=\langle x,w^\star\rangle+\xi,
    \qquad
    \mathbb E\xi^2=\sigma^2.
\]

The target/source condition is

\[
    s_i:=\lambda_i(w_i^\star)^2\asymp i^{-b},
    \qquad b>1.
\]

Let

\[
    N_{\rm eff}=N/\log N,
    \qquad
    n=N_{\rm eff}\gamma
\]

up to optimizer-dependent scalar normalizations.

## Theorem A: fixed spectral preconditioning

For

\[
    P_q=H^{-q},
    \qquad 0\le q<1,
\]

preconditioned SGD is ordinary SGD after the change of variables

\[
    \widetilde x=P_q^{1/2}x.
\]

The transformed covariance is

\[
    \widetilde H=H^{1-q},
\]

so the spectral exponent changes from \(a\) to

\[
    \alpha=a(1-q).
\]

The source energies are invariant:

\[
    \widetilde\lambda_i(\widetilde w_i^\star)^2
    =
    \lambda_i(w_i^\star)^2=s_i.
\]

Thus the risk filters are the SGD filters with \(a\) replaced by \(a(1-q)\).

## Theorem B: damped spectral preconditioning

For

\[
    P_{\rho,q}=(H+\rho I)^{-q},
\]

preconditioned SGD has effective eigenvalues

\[
    \mu_i^{\rho,q}=\lambda_i(\lambda_i+\rho)^{-q}.
\]

If \(\lambda_i\asymp i^{-a}\), then

\[
    \mu_i^{\rho,q}
    \asymp
    \begin{cases}
        i^{-a(1-q)}, & i\lesssim \rho^{-1/a},\\[3pt]
        \rho^{-q}i^{-a}, & i\gtrsim \rho^{-1/a}.
    \end{cases}
\]

The learned-mode count is

\[
    K_{\rho,q}(n)=\#\{i:\mu_i^{\rho,q}\gtrsim n^{-1}\}.
\]

Hence

\[
    K_{\rho,q}(n)
    \asymp
    \begin{cases}
        n^{1/[a(1-q)]}, & n\lesssim \rho^{-(1-q)},\\[3pt]
        \rho^{-q/a}n^{1/a}, & n\gtrsim \rho^{-(1-q)}.
    \end{cases}
\]

## Theorem C: RMSProp learns the damped \(q=1/2\) spectrum

For squared loss at iterate \(w_t\), let

\[
    e_t=w_t-w^\star,
    \qquad
    c_t=\sigma^2+\|e_t\|_H^2.
\]

The coordinate gradient second moment satisfies

\[
\boxed{
    \mathbb E[g_{t,i}^2\mid\mathcal F_t]
    =
    \lambda_i c_t+2\lambda_i^2e_{t,i}^2.
}
\]

Therefore

\[
    \lambda_i c_t
    \le
    \mathbb E[g_{t,i}^2\mid\mathcal F_t]
    \le
    3\lambda_i c_t.
\]

So a frozen or tracked RMSProp second-moment estimate satisfies

\[
    v_{t,i}\asymp d_t\lambda_i,
\]

where \(d_t\) is the EMA of \(c_t\).  Consequently,

\[
    (v_{t,i}+\epsilon)^{-1/2}
    \asymp
    d_t^{-1/2}\left(\lambda_i+\frac{\epsilon}{d_t}\right)^{-1/2}.
\]

Thus RMSProp behaves like damped spectral preconditioning with

\[
    q_{\rm eff}=1/2,
    \qquad
    \rho_t=\epsilon/d_t.
\]

## Theorem D: raw EMA concentration

Ordinary RMSProp uses

\[
    v_{t+1,i}=\beta_2v_{t,i}+(1-
    \beta_2)g_{t,i}^2.
\]

After unrolling the EMA, define the predictable target

\[
    m_{t,i}=\sum_{s<t}(1-\beta_2)\beta_2^{t-1-s}
    \mathbb E[g_{s,i}^2\mid\mathcal F_s],
\]

and the scalar target

\[
    d_t=\sum_{s<t}(1-\beta_2)\beta_2^{t-1-s}c_s.
\]

Then

\[
    \lambda_i d_t\le m_{t,i}\le3\lambda_i d_t.
\]

Under an effective-window/leverage condition, raw EMA concentration gives

\[
    v_{t,i}\asymp m_{t,i}\asymp d_t\lambda_i
\]

simultaneously over the relevant coordinates and times.  This removes the main conditional tracking assumption from the online RMSProp theorem.

## Theorem E: Adam first-moment momentum is a temporal filter

Adam adds

\[
    m_{t+1}=\beta_1m_t+(1-
    \beta_1)g_t.
\]

For a fixed preconditioner with effective eigenvalues \(\mu_i\), the population error polynomial obeys

\[
    p_{t+1}^{(\beta_1)}(x)
    =
    (1+\beta_1-(1-
    \beta_1)x)p_t^{(\beta_1)}(x)
    -\beta_1p_{t-1}^{(\beta_1)}(x),
    \qquad x=\gamma\mu_i.
\]

The root near one satisfies

\[
    r_+(x)=1-x+O_{\beta_1}(x^2).
\]

Hence the learning cutoff is still

\[
    N\gamma\mu_i\asymp1.
\]

Adam therefore has the same exponent-level learned-mode count as RMSProp; \(\beta_1\) changes constants, transients, and stability margins.

## Theorem F: AdamW adds a shrinkage cutoff

AdamW uses decoupled weight decay:

\[
    w_{t+1}=(1-
    \gamma\lambda_{\rm wd})w_t-
    \gamma(v_t+\epsilon)^{-1/2}m_t.
\]

For fixed effective eigenvalues \(\mu_i\), the population coordinate error solves exactly as

\[
    e_{N,i}
    =
    -\left[
        \frac{\delta}{\mu_i+
        \delta}
        +
        \frac{\mu_i}{\mu_i+
        \delta}r_i^N
    \right]w_i^\star,
    \qquad
    r_i=1-
    \gamma(\mu_i+
    \delta).
\]

Thus weight decay adds the shrinkage floor

\[
    \frac{\delta}{\mu_i+\delta}.
\]

The active cutoff becomes

\[
    \mu_i\gtrsim \max\{n^{-1},\delta\}.
\]

For Adam/RMSProp's damped \(q=1/2\) spectrum, define

\[
    T_{\rm eff}=\min\{n,
    \delta^{-1}\}.
\]

Then

\[
K_{\rho,{\rm wd}}(n,
\delta)
\asymp
K_{\rho,1/2}(T_{\rm eff}).
\]

## Theorem G: matching spectral lower bounds

For the two-slope effective spectrum

\[
    \mu_i\asymp
    \begin{cases}
        i^{-\alpha}, & i\lesssim m,\\[3pt]
        m^{a-
        \alpha}i^{-a}, & i\gtrsim m,
    \end{cases}
\]

with \(1<b<\alpha+1\) and \(\alpha>1/2\), the filter laws are sharp:

\[
    \sum_{i\le M}\frac{s_i}{1+n\mu_i}
    \asymp K(n)^{1-b},
\]

and

\[
    \frac1{N_{\rm eff}}
    \sum_{i\le M}\min\{1,n^2\mu_i^2\}
    \asymp
    \frac{\min\{M,K(n)\}}{N_{\rm eff}}.
\]

For AdamW, the same statements hold with

\[
    K(n)
    \quad\text{replaced by}\quad
    K_{\rm wd}(n,
    \delta)=\#\{i:\mu_i\gtrsim \max\{n^{-1},
    \delta\}\}.
\]

## Main conclusion so far

For diagonal Gaussian linear regression, RMSProp/Adam/AdamW have a common second-moment mechanism:

\[
    v_{t,i}\asymp d_t\lambda_i
    \quad\Longrightarrow\quad
    (v_{t,i}+\epsilon)^{-1/2}
    \asymp
    d_t^{-1/2}(\lambda_i+\epsilon/d_t)^{-1/2}.
\]

Therefore their adaptive part behaves like damped spectral preconditioning with

\[
    q_{\rm eff}=1/2.
\]

The learned-mode count is

\[
K_{\rho,1/2}(n)
\asymp
\begin{cases}
    n^{2/a}, & n\lesssim \rho^{-1/2},\\[3pt]
    \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2}.
\end{cases}
\]

Adam first-moment momentum does not change this exponent.  AdamW weight decay changes the effective horizon:

\[
    n\mapsto T_{\rm eff}=\min\{n,
    \lambda_{\rm wd}^{-1}\}
\]

after scalar normalization.

Thus, in the clean source range, the simplified diagonal-theory scaling law is

\[
\boxed{
    R_M-\sigma^2
    \asymp
    M^{1-b}
    +K^{1-b}
    +\frac{K}{N_{\rm eff}},
}
\]

with

\[
    K=K_{\rho,1/2}(n)
\]

for RMSProp/Adam, and

\[
    K=K_{\rho,1/2}(\min\{n,
    \lambda_{\rm wd}^{-1}\})
\]

for AdamW.

## What remains for a paper

The diagonal theorem stack is now coherent.  The project is not finished, because a strong paper still needs:

1. careful theorem assumptions and constants in a single manuscript;
2. direct exponent-level experiments across \((a,b,\rho,
   \lambda_{\rm wd},M,N)\);
3. coordinate-alignment and random-rotation experiments;
4. extension from diagonal Gaussian coordinates to Gaussian-sketched or random-feature regression;
5. comparison to SGD, signSGD, Muon/Shampoo-like spectral reshaping, and empirical AdamW scaling work.

The next theoretical theorem should target coordinate alignment: diagonal adaptive methods can act like spectral preconditioners only when optimizer coordinates are aligned with covariance eigendirections or when gradient second moments retain spectral information in the chosen coordinate system.
