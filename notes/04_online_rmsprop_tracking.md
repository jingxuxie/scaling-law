# Online RMSProp spectral tracking theorem

This note proves the next bridge after the frozen-RMSProp theorem.  The frozen result showed that a second-moment estimate at a fixed anchor satisfies

\[
\widehat v_i\asymp c_e\lambda_i,
\qquad
c_e=\sigma^2+\|\bar w-w^\star\|_H^2,
\]

so the frozen RMSProp preconditioner is spectrally equivalent to

\[
(\widehat v_i+\epsilon)^{-1/2}
\asymp
c_e^{-1/2}(\lambda_i+\epsilon/c_e)^{-1/2}.
\]

The online question is what happens when both \(w_t\) and the exponential moving average \(v_t\) evolve.  The theorem below shows that the online second-moment state remains spectrally equivalent to

\[
v_{t,i}\asymp \bar c_t\lambda_i,
\]

where \(\bar c_t\) is the same exponential moving average applied to the residual energy

\[
c_t:=\sigma^2+\|w_t-w^\star\|_H^2.
\]

Consequently, online RMSProp tracks the time-varying damped spectral preconditioner

\[
(v_{t,i}+\epsilon)^{-1/2}
\asymp
\bar c_t^{-1/2}(\lambda_i+\epsilon/\bar c_t)^{-1/2}.
\]

This gives the online version of the Adam/RMSProp bridge:

\[
\boxed{q_{\rm eff}=1/2,
\qquad
\rho_t=\epsilon/\bar c_t.}
\]

The clean theorem is first stated for the population EMA.  A robust online estimator gives a high-probability version with the same constants.  Raw RMSProp is then checked empirically in `experiments/online_rmsprop_tracking.py`.

---

## 1. Setup

Work in the population covariance eigenbasis:

\[
x_t\sim \mathcal N(0,H),
\qquad
H=\operatorname{diag}(\lambda_i),
\]

and

\[
y_t=\langle x_t,w^\star\rangle+\xi_t,
\qquad
\mathbb E\xi_t=0,
\qquad
\mathbb E\xi_t^2=\sigma^2,
\]

with \(\xi_t\) independent of \(x_t\).  Let \(\mathcal F_t\) be the sigma-field before drawing \((x_t,y_t)\).  Define

\[
e_t:=w_t-w^\star,
\qquad
c_t:=\sigma^2+\|e_t\|_H^2.
\]

The single-sample squared-loss gradient is

\[
g_t=(\langle x_t,e_t\rangle-\xi_t)x_t.
\]

The coordinatewise online RMSProp recursion is

\[
v_{t+1,i}=\beta v_{t,i}+(1-\beta)g_{t,i}^2,
\qquad
A_{t+1,i}=(v_{t+1,i}+\epsilon)^{-1/2},
\qquad
0\le \beta<1.
\]

For the population theorem, define

\[
\nu_{t,i}:=\mathbb E[g_{t,i}^2\mid\mathcal F_t],
\]

and the predictable EMA

\[
m_{t+1,i}=\beta m_{t,i}+(1-\beta)\nu_{t,i},
\qquad
m_{0,i}=0.
\]

Also define the scalar residual-energy EMA

\[
d_{t+1}=\beta d_t+(1-\beta)c_t,
\qquad
d_0=0.
\]

Bias correction divides both \(m_t\) and \(d_t\) by the same scalar \(1-\beta^t\), so it does not change the spectral equivalences below.

---

## 2. Conditional gradient second moment

### Lemma 1

For every time \(t\) and coordinate \(i\),

\[
\boxed{
\nu_{t,i}
=
\mathbb E[g_{t,i}^2\mid\mathcal F_t]
=
\lambda_i(\sigma^2+\|e_t\|_H^2)+2\lambda_i^2e_{t,i}^2.
}
\]

Equivalently, with

\[
r_{t,i}:=\lambda_i e_{t,i}^2,
\]

we have

\[
\nu_{t,i}=\lambda_i(c_t+2r_{t,i}).
\]

Since

\[
0\le r_{t,i}\le \sum_j\lambda_j e_{t,j}^2\le c_t,
\]

we get the uniform bound

\[
\boxed{
\lambda_i c_t
\le
\nu_{t,i}
\le
3\lambda_i c_t.
}
\]

### Proof

Condition on \(\mathcal F_t\), so \(e_t\) is fixed.  Let

\[
z_t=\langle x_t,e_t\rangle.
\]

Then \((z_t,x_{t,i})\) is centered Gaussian with

\[
\mathbb E[z_t^2\mid\mathcal F_t]=\|e_t\|_H^2,
\qquad
\mathbb E[x_{t,i}^2]=\lambda_i,
\qquad
\mathbb E[z_tx_{t,i}\mid\mathcal F_t]=\lambda_i e_{t,i}.
\]

By independence of \(\xi_t\) and Wick's formula,

\[
\begin{aligned}
\mathbb E[g_{t,i}^2\mid\mathcal F_t]
&=\mathbb E[(z_t-\xi_t)^2x_{t,i}^2\mid\mathcal F_t]\\
&=\mathbb E[z_t^2x_{t,i}^2\mid\mathcal F_t]+\sigma^2\mathbb E[x_{t,i}^2]\\
&=\lambda_i\|e_t\|_H^2+2\lambda_i^2e_{t,i}^2+\sigma^2\lambda_i.
\end{aligned}
\]

This proves the identity and the bound.

---

## 3. Population online RMSProp tracks the spectrum

### Theorem 1: population spectral tracking

For every \(t\ge0\) and coordinate \(i\),

\[
\boxed{
\lambda_i d_t
\le
m_{t,i}
\le
3\lambda_i d_t.
}
\]

Consequently, for the population preconditioner

\[
A^{\rm pop}_{t,i}:=(m_{t,i}+\epsilon)^{-1/2},
\]

we have, whenever \(d_t>0\),

\[
\boxed{
\frac1{\sqrt3}\,d_t^{-1/2}(\lambda_i+\epsilon/d_t)^{-1/2}
\le
A^{\rm pop}_{t,i}
\le
d_t^{-1/2}(\lambda_i+\epsilon/d_t)^{-1/2}.
}
\]

Thus population online RMSProp is spectrally equivalent to a time-varying damped \(q=1/2\) preconditioner:

\[
\boxed{
A^{\rm pop}_{t,i}\asymp d_t^{-1/2}(\lambda_i+\rho_t)^{-1/2},
\qquad
\rho_t:=\epsilon/d_t.
}
\]

### Proof

The proof is an induction.  At \(t=0\), \(m_{0,i}=d_0=0\), so the claim holds.  Suppose

\[
\lambda_i d_t\le m_{t,i}\le 3\lambda_i d_t.
\]

Using Lemma 1,

\[
\lambda_i c_t\le \nu_{t,i}\le3\lambda_i c_t.
\]

Therefore

\[
\begin{aligned}
m_{t+1,i}
&=\beta m_{t,i}+(1-\beta)\nu_{t,i}\\
&\ge \beta\lambda_i d_t+(1-\beta)\lambda_i c_t\\
&=\lambda_i d_{t+1},
\end{aligned}
\]

and similarly

\[
\begin{aligned}
m_{t+1,i}
&\le 3\beta\lambda_i d_t+3(1-\beta)\lambda_i c_t\\
&=3\lambda_i d_{t+1}.
\end{aligned}
\]

This proves the first display.  Adding \(\epsilon\) gives

\[
d_t\lambda_i+\epsilon
\le
m_{t,i}+\epsilon
\le
3d_t\lambda_i+\epsilon
\le
3(d_t\lambda_i+\epsilon).
\]

Taking inverse square roots and using

\[
(d_t\lambda_i+\epsilon)^{-1/2}=d_t^{-1/2}(\lambda_i+\epsilon/d_t)^{-1/2}
\]

proves the preconditioner bound.

---

## 4. High-probability robust online version

The raw EMA uses \(g_{t,i}^2\), a fourth-order Gaussian-chaos variable.  To avoid heavy-tail distractions, a clean theorem uses a robust per-step estimate \(\widehat\nu_{t,i}\) satisfying

\[
(1-\zeta)\nu_{t,i}
\le
\widehat\nu_{t,i}
\le
(1+\zeta)\nu_{t,i}
\]

simultaneously over \(t<T\) and \(i\le M\).  This event follows from coordinatewise median-of-means: at each time, split \(B=LR\) fresh samples into \(R\) blocks of size \(L\), average \(g_{t,i}^2\) in each block, and take the median across blocks.  Since \(g_{t,i}\) is a degree-two Gaussian polynomial, Gaussian hypercontractivity gives

\[
\mathbb E[g_{t,i}^4\mid\mathcal F_t]
\le C\nu_{t,i}^2.
\]

Thus choosing

\[
L\gtrsim \zeta^{-2},
\qquad
R\gtrsim \log(MT/\delta)
\]

gives the simultaneous event with probability at least \(1-\delta\).

Run the robust online EMA

\[
\widehat m_{t+1,i}=\beta\widehat m_{t,i}+(1-\beta)\widehat\nu_{t,i}.
\]

The same induction as above gives

\[
\boxed{
(1-\zeta)\lambda_i d_t
\le
\widehat m_{t,i}
\le
3(1+\zeta)\lambda_i d_t.
}
\]

Therefore

\[
\boxed{
\widehat A_{t,i}:=(\widehat m_{t,i}+\epsilon)^{-1/2}
\asymp_\zeta
d_t^{-1/2}(\lambda_i+\epsilon/d_t)^{-1/2}.
}
\]

So robust online RMSProp has the same time-varying damped spectral form as the population EMA.

---

## 5. Instantaneous transformed spectrum and damping knee

The instantaneous transformed covariance eigenvalue is

\[
\mu_{t,i}^{\rm rms}:=\lambda_i A_{t,i}.
\]

The tracking theorem gives

\[
\boxed{
\mu_{t,i}^{\rm rms}
\asymp
d_t^{-1/2}\lambda_i(\lambda_i+\rho_t)^{-1/2},
\qquad
\rho_t=\epsilon/d_t.
}
\]

The scalar \(d_t^{-1/2}\) only rescales the effective learning rate.  The spectral shape is

\[
\lambda_i(\lambda_i+ho_t)^{-1/2}.
\]

If

\[
\lambda_i\asymp i^{-a},
\]

then the instantaneous damping knee is

\[
m_t\asymp \rho_t^{-1/a},
\]

and

\[
\lambda_i(\lambda_i+ho_t)^{-1/2}
\asymp
\begin{cases}
i^{-a/2}, & i\lesssim m_t,\\[3pt]
\rho_t^{-1/2}i^{-a}, & i\gtrsim m_t.
\end{cases}
\]

Thus online RMSProp has the same two-slope spectrum as frozen RMSProp with \(q=1/2\), but with time-varying damping \(\rho_t=\epsilon/d_t\).

---

## 6. Quasi-static scaling corollary

Fix an interval \(I\) on which

\[
d_t\asymp d_I,
\qquad
\rho_t=\epsilon/d_t\asymp \rho_I:=\epsilon/d_I.
\]

Absorb the scalar \(d_I^{-1/2}\) into the learning rate and define

\[
n_I:=\sum_{t\in I}\eta_t d_I^{-1/2}.
\]

On this interval, online RMSProp has the same spectral filters as the frozen damped \(q=1/2\) theorem with

\[
\mu_i^{\rm rms}=\lambda_i(\lambda_i+ho_I)^{-1/2}.
\]

Define

\[
K_{\rho_I,1/2}(n_I):=\#\{i:\mu_i^{\rm rms}\gtrsim n_I^{-1}\}.
\]

For \(\lambda_i\asymp i^{-a}\),

\[
\boxed{
K_{\rho_I,1/2}(n_I)
\asymp
\begin{cases}
n_I^{2/a}, & n_I\lesssim \rho_I^{-1/2},\\[3pt]
\rho_I^{-1/(2a)}n_I^{1/a}, & n_I\gtrsim \rho_I^{-1/2}.
\end{cases}
}
\]

In the clean source range \(1<b<a/2+1\), away from the model-size bottleneck, the local risk law has the frozen-RMSProp form

\[
\boxed{
R-\sigma^2
\approx
M^{1-b}
+
K_{\rho_I,1/2}(n_I)^{1-b}
+
\frac{K_{\rho_I,1/2}(n_I)}{N_{\rm eff}}.
}
\]

This is a quasi-static corollary, not yet the final online risk theorem.  It is the precise spectral statement needed for a full sandwich proof.

---

## 7. Conditional risk sandwich

Assume that on the training horizon the raw or robust online preconditioner satisfies

\[
A_{t,i}\asymp d_I^{-1/2}(\lambda_i+\rho_I)^{-1/2}
\]

uniformly over the interval \(I\).  Because all preconditioners are diagonal in the covariance eigenbasis, the coordinate bias filters involve scalar products

\[
\prod_{t\in I}(1-\eta_t\lambda_i A_{t,i}).
\]

If \(\lambda_i A_{t,i}\) is within constant factors of

\[
d_I^{-1/2}\lambda_i(\lambda_i+\rho_I)^{-1/2},
\]

then these products are comparable to the frozen damped filter with effective time \(n_I\).  The stochastic variance filter is controlled by the same effective dimension.  Hence, under the same stability/geometric-decay assumptions as the fixed spectral theorem,

\[
\boxed{
R_M(w_{\rm end(I)})-\sigma^2
\lesssim
M^{1-b}
+B_{\rho_I,1/2}(n_I,M)
+V_{\rho_I,1/2}(n_I,M),
}
\]

where

\[
B_{\rho,1/2}(n,M)=\sum_{i\le M}\frac{s_i}{1+n\lambda_i(\lambda_i+\rho)^{-1/2}},
\]

and

\[
V_{\rho,1/2}(n,M)=\frac1{N_{\rm eff}}\sum_{i\le M}\min\{1,n^2\lambda_i^2(\lambda_i+\rho)^{-1}\}.
\]

A matching lower sandwich should follow from the lower envelope plus the usual nondegenerate-noise lower bound.  Proving that lower bound for fully raw RMSProp is the next technical target.

---

## 8. Interpretation and next steps

This theorem proves the online optimizer bridge:

\[
\boxed{
\text{online second-moment EMA}
\Rightarrow
v_{t,i}\asymp d_t\lambda_i
\Rightarrow
A_{t,i}\asymp d_t^{-1/2}(\lambda_i+\epsilon/d_t)^{-1/2}
\Rightarrow
q_{\rm eff}=1/2.
}
\]

The damping parameter is not fixed; it evolves as

\[
\rho_t=\epsilon/d_t,
\qquad
d_t\approx \operatorname{EMA}(\sigma^2+\|w_t-w^\star\|_H^2).
\]

The next proof target is a full online risk theorem with matching lower bounds, first conditional on the raw-EMA tracking event and then with a direct high-probability proof for raw RMSProp.  After that, the natural extensions are \(\beta_1\)-momentum, Adam bias correction, AdamW decay, and coordinate misalignment.