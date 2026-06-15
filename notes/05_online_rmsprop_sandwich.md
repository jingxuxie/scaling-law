# Online RMSProp sandwich theorem

This note turns the online spectral-tracking theorem into a risk/scaling theorem.
The tracking note proved the population identity

\[
m_{t,i}:=\mathbb E[g_{t,i}^2\mid \mathcal F_t]
=\lambda_i(\sigma^2+\|e_t\|_H^2)+2\lambda_i^2e_{t,i}^2,
\]

and therefore

\[
c_t\lambda_i\le m_{t,i}\le 3c_t\lambda_i,
\qquad
c_t:=\sigma^2+\|e_t\|_H^2.
\]

The result below says that, whenever the online second-moment state tracks these
conditional means and the scalar residual energy does not move by more than a
constant factor over the horizon, RMSProp is sandwiched by fixed damped
\(q=1/2\) spectral preconditioners.  Hence it has the same scaling filters as the
frozen-RMSProp theorem.

---

## 1. Setup

Work in diagonal Gaussian linear regression:

\[
x\sim\mathcal N(0,H),\qquad H=\operatorname{diag}(\lambda_i),
\qquad \lambda_i\asymp i^{-a},\quad a>1,
\]

\[
y=\langle x,w^\star\rangle+\xi,
\qquad \mathbb E\xi^2=\sigma^2,
\]

and source energies

\[
s_i:=\lambda_i(w_i^\star)^2\asymp i^{-b},\qquad b>1.
\]

RMSProp without first-moment momentum is

\[
v_{t,i}=\beta_2v_{t-1,i}+(1-\beta_2)g_{t,i}^2,
\qquad
A_{t,i}=(v_{t,i}+\epsilon)^{-1/2},
\]

\[
w_{t+1}=w_t-\gamma_t A_tg_t.
\]

Define

\[
e_t=w_t-w^\star,
\qquad
c_t=\sigma^2+\|e_t\|_H^2.
\]

---

## 2. Tracking and residual-band assumptions

For constants \(0<\kappa_-\le \kappa_+<\infty\), define the online tracking event

\[
\mathcal E_{\rm track}:=
\left\{
\kappa_-c_t\lambda_i\le v_{t,i}\le \kappa_+c_t\lambda_i,
\quad 1\le i\le M,
\quad 0\le t<N
\right\}.
\]

Also assume the scalar residual level stays in a constant band:

\[
0<c_{\min}\le c_t\le c_{\max}<\infty,
\qquad
\chi:=c_{\max}/c_{\min}=O(1).
\]

Choose any reference value \(c_\circ\in[c_{\min},c_{\max}]\) and set

\[
\rho_\circ:=\epsilon/c_\circ.
\]

The constant-band condition is not meant to hold for all of training in every
problem.  It is the quasi-static regime in which one can assign a single damping
level to the online method.  If \(c_t\) drifts across many scales, the theorem
applies piecewise, with a changing \(\rho_\circ\).

---

## 3. Tracking implies a fixed spectral envelope

### Lemma 1: online preconditioner envelope

On \(\mathcal E_{\rm track}\) and under the residual-band condition,

\[
\boxed{
C_-c_\circ^{-1/2}(\lambda_i+\rho_\circ)^{-1/2}
\le
A_{t,i}
\le
C_+c_\circ^{-1/2}(\lambda_i+\rho_\circ)^{-1/2}
}
\]

for every \(i\le M\) and \(0\le t<N\).  The constants \(C_-,C_+\) depend only on
\(\kappa_-,\kappa_+\) and \(\chi\).  Consequently

\[
\mu_{t,i}:=\lambda_iA_{t,i}
\asymp
c_\circ^{-1/2}\lambda_i(\lambda_i+\rho_\circ)^{-1/2}.
\]

### Proof

On the tracking event,

\[
\kappa_-c_t\lambda_i+\epsilon
\le v_{t,i}+\epsilon
\le
\kappa_+c_t\lambda_i+\epsilon.
\]

Since \(c_t/c_\circ\in[\chi^{-1},\chi]\),

\[
\kappa_-c_t\lambda_i+\epsilon
\ge
\min\{\kappa_-/\chi,1\}(c_\circ\lambda_i+\epsilon),
\]

and

\[
\kappa_+c_t\lambda_i+\epsilon
\le
\max\{\kappa_+\chi,1\}(c_\circ\lambda_i+\epsilon).
\]

Using \(c_\circ\lambda_i+\epsilon=c_\circ(\lambda_i+\rho_\circ)\) and taking inverse
square roots gives the claim.

---

## 4. Filter stability under a time-varying spectral envelope

Let

\[
\mu_i^\circ:=\lambda_i(\lambda_i+\rho_\circ)^{-1/2},
\qquad
N_{\rm eff}:=N/\log N,
\qquad
n:=N_{\rm eff}\gamma c_\circ^{-1/2}.
\]

Define the fixed-reference filters

\[
B_{\rho_\circ,1/2}(n,M)
:=\sum_{i\le M}\frac{s_i}{1+n\mu_i^\circ},
\]

\[
V_{\rho_\circ,1/2}(n,M)
:=\frac1{N_{\rm eff}}\sum_{i\le M}\min\{1,n^2(\mu_i^\circ)^2\}.
\]

### Lemma 2: filter comparison

Assume the same geometric step-decay schedule and stability condition as in the
deterministic damped theorem.  If the time-varying eigenvalues satisfy

\[
\mu_{t,i}\asymp c_\circ^{-1/2}\mu_i^\circ
\]

uniformly over \((t,i)\), then the bias and variance filters of the time-varying
preconditioned recursion are comparable to

\[
B_{\rho_\circ,1/2}(n,M),
\qquad
V_{\rho_\circ,1/2}(n,M).
\]

### Proof

For coordinate \(i\), the relevant cumulative learning mass is

\[
\Theta_i:=\sum_{t<N}\gamma_t\mu_{t,i}.
\]

The spectral envelope and the geometric schedule give

\[
\Theta_i\asymp c_\circ^{-1/2}\mu_i^\circ\sum_{t<N}\gamma_t
\asymp N_{\rm eff}\gamma c_\circ^{-1/2}\mu_i^\circ
=n\mu_i^\circ.
\]

The last-iterate one-pass bias filter has a single transition at
\(\Theta_i\asymp1\) and is comparable to \((1+\Theta_i)^{-1}\), so the
source-weighted bias is comparable to

\[
\sum_{i\le M}\frac{s_i}{1+n\mu_i^\circ}.
\]

The variance part is controlled by the effective variance dimension

\[
\mathcal D\asymp\sum_{i\le M}\min\{1,\Theta_i^2\}.
\]

Dividing by \(N_{\rm eff}\) and substituting \(\Theta_i\asymp n\mu_i^\circ\) gives
\(V_{\rho_\circ,1/2}(n,M)\).  Constant-factor changes in \(\Theta_i\) do not change
these filters beyond constants.

---

## 5. Main theorem

### Theorem 1: conditional online RMSProp sandwich scaling law

Assume:

1. \(\lambda_i\asymp i^{-a}\), \(a>1\), and \(s_i\asymp i^{-b}\), \(b>1\);
2. the tracking event \(\mathcal E_{\rm track}\) holds for \(0\le t<N\), \(i\le M\);
3. \(c_t\in[c_{\min},c_{\max}]\) with \(c_{\max}/c_{\min}=O(1)\);
4. the stepsize schedule is stable with respect to the upper spectral envelope.

Then online RMSProp satisfies, up to constants and logarithmic factors,

\[
\boxed{
\mathbb E R_M(w_N)-\sigma^2
\asymp
\underbrace{\sum_{i>M}s_i}_{\text{approximation}}
+
\underbrace{B_{\rho_\circ,1/2}(n,M)}_{\text{bias/optimization}}
+
\underbrace{V_{\rho_\circ,1/2}(n,M)}_{\text{variance/noise}},
}
\]

where

\[
\rho_\circ=\epsilon/c_\circ,
\qquad
n=N_{\rm eff}\gamma c_\circ^{-1/2},
\qquad
\mu_i^\circ=\lambda_i(\lambda_i+\rho_\circ)^{-1/2}.
\]

Thus, conditional on online tracking, RMSProp has the same scaling law as frozen
damped spectral preconditioning with \(q=1/2\).

### Proof

Lemma 1 gives

\[
\mu_{t,i}=\lambda_iA_{t,i}
\asymp c_\circ^{-1/2}\mu_i^\circ.
\]

Lemma 2 then converts the time-varying recursion to the fixed filters
\(B_{\rho_\circ,1/2}\) and \(V_{\rho_\circ,1/2}\).  The approximation term is
unchanged because the model keeps only the first \(M\) source coordinates:

\[
\sum_{i>M}s_i\asymp M^{1-b}.
\]

The source sequence is invariant under every diagonal preconditioner:

\[
\mu_{t,i}(A_{t,i}^{-1/2}w_i^\star)^2
=\lambda_iA_{t,i}\cdot A_{t,i}^{-1}(w_i^\star)^2
=s_i.
\]

Therefore online RMSProp changes only the effective covariance spectrum, not the
source exponent.  Combining approximation, bias, and variance gives the stated
risk law.

---

## 6. Robust decoupled online RMSProp proves the tracking event

The theorem above is conditional for raw RMSProp.  The condition can be proved
for a robust decoupled estimator that uses fresh probe gradients to estimate the
second moment at each iterate.

At time \(t\), draw \(B=LR\) independent probe samples at the current \(w_t\), set
\(Y_{t,\ell,i}=g_{t,\ell,i}(w_t)^2\), partition into \(R\) blocks of size \(L\),
and define

\[
\widehat v_{t,i}
:=\operatorname{median}_{1\le r\le R}
\frac1L\sum_{\ell\in\text{block }r}Y_{t,\ell,i}.
\]

Use \(A_{t,i}=(\widehat v_{t,i}+\epsilon)^{-1/2}\).

### Theorem 2: robust online tracking

For any \(\tau\in(0,1)\) and \(\delta\in(0,1)\), if

\[
L\ge C_1\tau^{-2},
\qquad
R\ge C_2\log(2MN/\delta),
\]

then with probability at least \(1-\delta\), simultaneously for all
\(0\le t<N\) and \(1\le i\le M\),

\[
\boxed{
(1-\tau)m_{t,i}\le \widehat v_{t,i}\le (1+\tau)m_{t,i},
}
\]

where \(m_{t,i}=\mathbb E[g_{t,i}^2\mid w_t]\).  Consequently,

\[
\boxed{
(1-\tau)c_t\lambda_i
\le \widehat v_{t,i}\le
3(1+\tau)c_t\lambda_i.
}
\]

### Proof

Condition on the past up to time \(t\), so \(w_t\) is fixed.  The probe variables
\(Y_{t,\ell,i}\) are independent degree-four Gaussian-chaos variables.  As in the
frozen theorem,

\[
\operatorname{Var}(Y_{t,\ell,i}\mid w_t)\le C m_{t,i}^2.
\]

Chebyshev's inequality makes each block mean \(\tau\)-accurate with probability
at least \(3/4\) when \(L\ge C_1\tau^{-2}\).  A Chernoff bound over the \(R\)
blocks makes the median fail with probability at most \(\exp(-cR)\) for one
coordinate and time.  Taking \(R\ge C_2\log(2MN/\delta)\) and union bounding over
all \((t,i)\) proves the simultaneous estimate.  The final display follows from
the online gradient-second-moment identity.

### Corollary: robust online RMSProp scaling

If the residual scale stays in a constant band, the robust decoupled online
method satisfies Theorem 1 with probability at least \(1-\delta\).  This gives a
fully proved online-adaptive scaling theorem for a robust RMSProp variant.  Raw
RMSProp/Adam remains the next concentration step.

---

## 7. Power-law simplification

The fixed reference profile is

\[
\mu_i^\circ=\lambda_i(\lambda_i+\rho_\circ)^{-1/2}.
\]

For \(\lambda_i\asymp i^{-a}\), there is a spectral knee

\[
m_{\rho_\circ}\asymp \rho_\circ^{-1/a}
\]

and a sample-size knee

\[
n_{\rho_\circ}\asymp \rho_\circ^{-1/2}.
\]

The learned-mode count is

\[
K_{\rho_\circ,1/2}(n)=\#\{i:\mu_i^\circ\gtrsim n^{-1}\},
\]

so

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

In the clean range \(1<b<a/2+1\), when \(M\) is not the bottleneck,

\[
B_{\rho_\circ,1/2}(n,M)\asymp K_{\rho_\circ,1/2}(n)^{1-b},
\qquad
V_{\rho_\circ,1/2}(n,M)\asymp \frac{\min\{M,K_{\rho_\circ,1/2}(n)\}}{N_{\rm eff}}.
\]

Therefore, before the damping knee,

\[
\boxed{
R_M(w_N)-\sigma^2
\asymp
M^{1-b}+n^{-2(b-1)/a}+\frac{n^{2/a}}{N_{\rm eff}}.
}
\]

After the damping knee,

\[
\boxed{
R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+\rho_\circ^{(b-1)/(2a)}n^{-(b-1)/a}
+\frac{\rho_\circ^{-1/(2a)}n^{1/a}}{N_{\rm eff}}.
}
\]

The active online-RMSProp phase therefore replaces the SGD spectral exponent
\(a\) by \(a/2\), until the \(\epsilon\)-controlled damping knee is reached.

---

## 8. Next theorem target

The remaining gap for ordinary Adam/RMSProp is the raw EMA concentration event:

\[
\kappa_-c_t\lambda_i\le v_{t,i}\le \kappa_+c_t\lambda_i.
\]

For the bias-corrected EMA,

\[
\bar v_{t,i}
=\frac{(1-\beta_2)\sum_{s=1}^t\beta_2^{t-s}g_{s,i}^2}{1-\beta_2^t},
\]

one expects concentration when the effective window
\(W=(1-\beta_2)^{-1}\) satisfies

\[
W\gg \log(MN/\delta)
\quad\text{and}\quad
W\ll t_{\rm drift}.
\]

The first inequality averages noise; the second prevents the target mean
\(c_t\lambda_i\) from moving too much during the EMA window.  Proving this for
raw degree-four Gaussian-chaos observations is the next technical step before
adding first-moment momentum and AdamW weight decay.
