# Frozen RMSProp bridge theorem

This note proves the first real bridge from the deterministic spectral
preconditioning theory to Adam/RMSProp-like adaptive preconditioning.

The deterministic theorem says that, if we train with the fixed spectral
preconditioner

\[
P_{\rho,q}=(H+\rho I)^{-q},
\]

then preconditioned SGD is ordinary SGD on transformed features with covariance

\[
\widetilde H=P_{\rho,q}^{1/2}HP_{\rho,q}^{1/2}.
\]

For Adam/RMSProp, the relevant case should be \(q=1/2\).  This note proves why:
in Gaussian linear regression, the coordinate-wise gradient second moment is
proportional to the population eigenvalue \(\lambda_i\), up to universal
constants.  Therefore a frozen RMSProp preconditioner

\[
A_i=(\widehat v_i+\epsilon)^{-1/2}
\]

is spectrally equivalent to

\[
(H+\rho I)^{-1/2},
\qquad \rho\asymp \epsilon /(\sigma^2+\|\bar w-w^\star\|_H^2).
\]

This turns the previous damped spectral theorem into a frozen-RMSProp theorem.

---

## 1. Setup

Work in the population covariance eigenbasis.  Let

\[
x\sim \mathcal N(0,H),
\qquad
H=\operatorname{diag}(\lambda_i),
\qquad
\lambda_i>0,
\]

and

\[
y=\langle x,w^\star\rangle+\xi,
\qquad
\xi\sim \mathcal N(0,\sigma^2),
\]

with \(\xi\) independent of \(x\).  For a fixed anchor parameter \(\bar w\), define

\[
e:=\bar w-w^\star,
\qquad
c_e:=\sigma^2+\|e\|_H^2.
\]

The single-sample squared-loss gradient at \(\bar w\) is

\[
g(\bar w)=(\langle x,e\rangle-\xi)x.
\]

For coordinate \(i\), write

\[
g_i(\bar w)=(\langle x,e\rangle-\xi)x_i.
\]

---

## 2. Key identity: gradient second moments recover the spectrum

### Lemma 1: coordinate gradient second moment

For every coordinate \(i\),

\[
\boxed{
\mathbb E[g_i(\bar w)^2\mid \bar w]
=
\lambda_i\bigl(\sigma^2+\|e\|_H^2\bigr)
+
2\lambda_i^2e_i^2.
}
\]

Equivalently, if

\[
r_i:=\lambda_i e_i^2,
\]

then

\[
\mathbb E[g_i(\bar w)^2\mid \bar w]
=
\lambda_i(c_e+2r_i).
\]

Since

\[
0\le r_i\le \sum_j\lambda_j e_j^2\le c_e,
\]

we have the uniform constant-factor equivalence

\[
\boxed{
 c_e\lambda_i
\le
\mathbb E[g_i(\bar w)^2\mid \bar w]
\le
3c_e\lambda_i.
}
\]

### Proof

Let

\[
z:=\langle x,e\rangle.
\]

Then \(z\) and \(x_i\) are jointly centered Gaussian with

\[
\mathbb E[z^2]=\|e\|_H^2,
\qquad
\mathbb E[x_i^2]=\lambda_i,
\qquad
\mathbb E[z x_i]=\lambda_i e_i.
\]

Because \(\xi\) is independent and mean zero,

\[
\mathbb E[(z-\xi)^2x_i^2]
=
\mathbb E[z^2x_i^2]+\sigma^2\mathbb E[x_i^2].
\]

By Isserlis/Wick's formula,

\[
\mathbb E[z^2x_i^2]
=
\mathbb E[z^2]\mathbb E[x_i^2]
+2\mathbb E[zx_i]^2
=
\lambda_i\|e\|_H^2+2\lambda_i^2e_i^2.
\]

Therefore

\[
\mathbb E[g_i(\bar w)^2\mid \bar w]
=
\lambda_i(\sigma^2+\|e\|_H^2)+2\lambda_i^2e_i^2.
\]

The constant-factor bound follows from \(0\le \lambda_i e_i^2\le \|e\|_H^2\le c_e\).

---

## 3. Robust burn-in estimator

The raw RMSProp estimate is an exponential moving average.  For a clean theorem,
we first analyze a robust frozen estimate.  This avoids heavy-tail technicalities
while preserving the Adam/RMSProp mechanism.

Let

\[
Y_{t,i}:=g_{t,i}(\bar w)^2.
\]

Partition \(B=LR\) independent burn-in samples into \(R\) blocks of size \(L\), and
define

\[
\bar Y_{r,i}:=\frac1L\sum_{t\in \text{block }r}Y_{t,i},
\qquad
\widehat v_i:=\operatorname{median}_{1\le r\le R}\bar Y_{r,i}.
\]

### Lemma 2: uniform constant-factor concentration

There is a universal numerical constant \(C\) such that, for every \(i\),

\[
\operatorname{Var}(Y_{t,i}\mid \bar w)
\le
C\,\mathbb E[Y_{t,i}\mid \bar w]^2.
\]

Consequently, for any \(\tau\in(0,1)\), if

\[
L\ge C_1\tau^{-2},
\qquad
R\ge C_2\log(2M/\delta),
\]

then with probability at least \(1-\delta\), simultaneously for all \(1\le i\le M\),

\[
\boxed{
(1-\tau)v_i^\star
\le
\widehat v_i
\le
(1+\tau)v_i^\star,
\qquad
v_i^\star:=\mathbb E[g_i(\bar w)^2\mid \bar w].
}
\]

Combining with Lemma 1 gives, on the same event,

\[
\boxed{
(1-\tau)c_e\lambda_i
\le
\widehat v_i
\le
3(1+\tau)c_e\lambda_i,
\qquad 1\le i\le M.
}
\]

### Proof

Let

\[
u:=\langle x,e\rangle-\xi.
\]

Under the Gaussian assumptions, \((u,x_i)\) is a centered bivariate Gaussian pair
with variances \(c_e\) and \(\lambda_i\).  The random variable

\[
Y_i=u^2x_i^2
\]

is a degree-four polynomial in a Gaussian vector.  By Wick's formula, or Gaussian
hypercontractivity,

\[
\mathbb E[u^4x_i^4]
\le
C\,\mathbb E[u^2x_i^2]^2
=
C(v_i^\star)^2.
\]

Thus \(\operatorname{Var}(Y_i)\le C(v_i^\star)^2\).  For a block mean,

\[
\operatorname{Var}(\bar Y_{r,i})
\le
\frac{C}{L}(v_i^\star)^2.
\]

Chebyshev's inequality gives

\[
\Pr\left(|\bar Y_{r,i}-v_i^\star|>\tau v_i^\star\right)
\le
\frac{C}{L\tau^2}.
\]

Choosing \(L\ge C_1\tau^{-2}\) makes each block good with probability at least,
say, \(3/4\).  A Chernoff bound over the \(R\) independent blocks shows that the
median is bad with probability at most \(\exp(-cR)\).  A union bound over
\(i\le M\) gives the stated simultaneous event.

---

## 4. Frozen-RMSProp preconditioner is a damped spectral preconditioner

Define the frozen adaptive preconditioner

\[
A:=\operatorname{diag}(A_i),
\qquad
A_i:=(\widehat v_i+\epsilon)^{-1/2}.
\]

Set the effective damping level

\[
\rho:=\epsilon/c_e.
\]

### Lemma 3: spectral equivalence

On the event of Lemma 2,

\[
\boxed{
C_- c_e^{-1/2}(\lambda_i+\rho)^{-1/2}
\le
A_i
\le
C_+ c_e^{-1/2}(\lambda_i+\rho)^{-1/2},
\qquad 1\le i\le M,
}
\]

where \(C_-,C_+\) depend only on \(\tau\).

Consequently, the transformed covariance

\[
\widehat H:=A^{1/2}HA^{1/2}
\]

has eigenvalues

\[
\widehat\mu_i=\lambda_i A_i
\]

satisfying

\[
\boxed{
\widehat\mu_i
\asymp
c_e^{-1/2}\lambda_i(\lambda_i+\rho)^{-1/2}.
}
\]

### Proof

From Lemma 2,

\[
(1-\tau)c_e\lambda_i+\epsilon
\le
\widehat v_i+\epsilon
\le
3(1+\tau)c_e\lambda_i+\epsilon.
\]

Equivalently, since \(\rho=\epsilon/c_e\),

\[
\widehat v_i+\epsilon
\asymp_{\tau}
c_e(\lambda_i+\rho).
\]

Taking inverse square roots yields the claimed bound on \(A_i\).  Multiplying by
\(\lambda_i\) yields the transformed eigenvalue bound.

---

## 5. Main theorem: frozen RMSProp scaling law

Let the source energy be

\[
s_i:=\lambda_i(w_i^\star)^2.
\]

Assume the source condition

\[
s_i\asymp i^{-b},
\qquad b>1,
\]

and covariance eigenvalues

\[
\lambda_i\asymp i^{-a},
\qquad a>1.
\]

After the burn-in estimator is frozen, run one-pass SGD with update

\[
w_{t+1}
=
w_t-\gamma_t A(\langle x_t,w_t\rangle-y_t)x_t.
\]

Let

\[
N_{\rm eff}:=N/\log N,
\qquad
n:=N_{\rm eff}\gamma c_e^{-1/2},
\]

and define the damped Adam/RMSProp spectral profile

\[
\mu_i^{\rm rms}:=\lambda_i(\lambda_i+\rho)^{-1/2},
\qquad
\rho=\epsilon/c_e.
\]

Define the deterministic bias and variance filters

\[
B_{\rho,1/2}(n,M)
:=
\sum_{i\le M}\frac{s_i}{1+n\mu_i^{\rm rms}},
\]

and

\[
V_{\rho,1/2}(n,M)
:=
\frac1{N_{\rm eff}}
\sum_{i\le M}\min\{1,n^2(\mu_i^{\rm rms})^2\}.
\]

### Theorem 1: frozen RMSProp bridge

Assume the stepsize schedule satisfies the same stability and geometric-decay
conditions as the deterministic spectral-preconditioning theorem, with
\(\gamma\) small enough relative to \(\operatorname{tr}(\widehat H)^{-1}\).  On the
burn-in event of Lemma 2, the post-burn-in frozen-RMSProp iterate satisfies,
up to universal constants and logarithmic factors,

\[
\boxed{
\mathbb E\bigl[R_M(w_N)\mid \widehat v\bigr]-\sigma^2
\asymp
\underbrace{\sum_{i>M}s_i}_{\text{approximation}}
+
\underbrace{B_{\rho,1/2}(n,M)}_{\text{optimization/bias}}
+
\underbrace{V_{\rho,1/2}(n,M)}_{\text{variance/noise}}.
}
\]

Therefore, with probability at least \(1-\delta\) over the burn-in samples,

\[
\boxed{
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+
B_{\rho,1/2}(n,M)
+
V_{\rho,1/2}(n,M),
}
\]

where the outer expectation is over the post-burn-in SGD samples and the only
failure probability is the burn-in concentration event.

### Proof

The preconditioned update is ordinary SGD after the change of variables

\[
w=A^{1/2}u,
\qquad
\widetilde x=A^{1/2}x.
\]

Indeed,

\[
\langle x,w\rangle
=
\langle A^{1/2}x,u\rangle
=
\langle \widetilde x,u\rangle,
\]

and the update in \(u\) is

\[
u_{t+1}
=
u_t-\gamma_t(\langle \widetilde x_t,u_t\rangle-y_t)\widetilde x_t.
\]

The transformed covariance is

\[
\widehat H=A^{1/2}HA^{1/2},
\qquad
\widehat\mu_i=\lambda_i A_i.
\]

The transformed target is

\[
u^\star=A^{-1/2}w^\star.
\]

Crucially, the source energy is exactly invariant:

\[
\widehat\mu_i(u_i^\star)^2
=
\lambda_i A_i\cdot A_i^{-1}(w_i^\star)^2
=
\lambda_i(w_i^\star)^2
=s_i.
\]

So frozen RMSProp changes the covariance spectrum but does not change the source
exponent.  Applying the deterministic spectral-preconditioning theorem to the
transformed problem gives

\[
\mathbb E[R_M(w_N)\mid \widehat v]-\sigma^2
\asymp
\sum_{i>M}s_i
+
\sum_{i\le M}\frac{s_i}{1+N_{\rm eff}\gamma\widehat\mu_i}
+
\frac1{N_{\rm eff}}
\sum_{i\le M}\min\{1,(N_{\rm eff}\gamma\widehat\mu_i)^2\}.
\]

By Lemma 3,

\[
\widehat\mu_i
\asymp
c_e^{-1/2}\mu_i^{\rm rms}.
\]

Thus

\[
N_{\rm eff}\gamma\widehat\mu_i
\asymp
n\mu_i^{\rm rms},
\qquad
n=N_{\rm eff}\gamma c_e^{-1/2}.
\]

The filters \(s/(1+x)\) and \(\min\{1,x^2\}\) are stable under constant-factor
changes in \(x\).  Therefore the deterministic risk expression is comparable to

\[
\sum_{i>M}s_i+B_{\rho,1/2}(n,M)+V_{\rho,1/2}(n,M).
\]

Finally,

\[
\sum_{i>M}s_i\asymp \sum_{i>M}i^{-b}\asymp M^{1-b}.
\]

This proves the theorem.

---

## 6. Power-law simplification and damping knee

Since

\[
\lambda_i\asymp i^{-a},
\qquad
\mu_i^{\rm rms}=\lambda_i(\lambda_i+\rho)^{-1/2},
\]

there is a spectral knee at

\[
m_\rho\asymp \rho^{-1/a}.
\]

For \(i\lesssim m_\rho\),

\[
\mu_i^{\rm rms}\asymp \lambda_i^{1/2}\asymp i^{-a/2}.
\]

For \(i\gtrsim m_\rho\),

\[
\mu_i^{\rm rms}\asymp \rho^{-1/2}\lambda_i\asymp \rho^{-1/2}i^{-a}.
\]

The sample-size knee is

\[
n_\rho\asymp \rho^{-1/2}.
\]

Define the learned-mode count

\[
K_{\rho,1/2}(n):=\#\{i:\mu_i^{\rm rms}\gtrsim n^{-1}\}.
\]

Then

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

In the clean range \(1<b<a/2+1\), the bias filter satisfies

\[
B_{\rho,1/2}(n,M)
\asymp
K_{\rho,1/2}(n)^{1-b}
\]

when \(M\) is not the bottleneck.  Since \(a>1\), the pre-knee exponent \(a/2\)
is larger than \(1/2\), so the variance filter obeys

\[
V_{\rho,1/2}(n,M)
\asymp
\frac{\min\{M,K_{\rho,1/2}(n)\}}{N_{\rm eff}}.
\]

Thus, before the damping knee,

\[
\boxed{
R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+n^{-2(b-1)/a}
+\frac{n^{2/a}}{N_{\rm eff}}.
}
\]

After the damping knee,

\[
\boxed{
R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+\rho^{(b-1)/(2a)}n^{-(b-1)/a}
+\frac{\rho^{-1/(2a)}n^{1/a}}{N_{\rm eff}}.
}
\]

The sign of the \(\rho\) factor in the bias term is important: smaller \(\rho\)
improves the tail bias by increasing the effective learned-mode count.

---

## 7. Interpretation

The theorem proves the central bridge:

\[
\boxed{
\text{gradient second moments}
\Rightarrow
\widehat v_i\asymp \lambda_i
\Rightarrow
A_i\asymp(\lambda_i+\rho)^{-1/2}
\Rightarrow
\text{Adam-like }q=1/2\text{ scaling.}
}
\]

This is stronger than the heuristic \(P_i\propto\lambda_i^{-1/2}\): the damping
\(\epsilon\) creates the effective spectral cutoff

\[
\rho=\epsilon/(\sigma^2+\|\bar w-w^\star\|_H^2),
\]

and therefore a sample-size knee

\[
n_\rho\asymp \rho^{-1/2}.
\]

So Adam/RMSProp-style adaptivity changes scaling exponents only across the part
of the spectrum where

\[
\lambda_i\gtrsim \rho.
\]

Beyond that range, the effective spectrum returns to the original \(i^{-a}\)
tail, with a \(\rho^{-1/(2a)}\) constant boost in the learned-mode count.

---

## 8. Next target theorem

The next theorem should replace the frozen preconditioner by the online RMSProp
recursion

\[
v_{t+1}=\beta_2v_t+(1-\beta_2)g_t^2,
\qquad
w_{t+1}=w_t-\eta(v_t+\epsilon)^{-1/2}g_t.
\]

The same identity gives

\[
\mathbb E[g_{t,i}^2\mid e_t]
=
\lambda_i(\sigma^2+\|e_t\|_H^2)+2\lambda_i^2e_{t,i}^2.
\]

Since the coordinate-specific term is always within a constant factor of the
global term, the online preconditioner should track

\[
(v_{t,i}+\epsilon)^{-1/2}
\asymp
\bigl((\sigma^2+\|e_t\|_H^2)\lambda_i+\epsilon\bigr)^{-1/2}
\asymp
(\lambda_i+\rho_t)^{-1/2},
\]

where

\[
\rho_t=\epsilon/(\sigma^2+\|e_t\|_H^2).
\]

The online theorem should show that RMSProp is sandwiched between two frozen
damped spectral preconditioners with \(\rho\in[\rho_{\min},\rho_{\max}]\), yielding
the same exponent whenever \(\rho_t\) varies slowly relative to the EMA window
\((1-\beta_2)^{-1}\).
