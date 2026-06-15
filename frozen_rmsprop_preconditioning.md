# Frozen RMSProp preconditioning theorem

This note proves the next theorem in the roadmap.  The damped spectral theorem treats a fixed oracle preconditioner

\[
P_{\rho,q}=(H+\rho I)^{-q}.
\]

Here we show that a frozen RMSProp-style second-moment estimate recovers the Adam/RMSProp-like case \(q=1/2\), up to constant spectral equivalence.

The main message is

\[
\widehat v_i\approx \mathbb E[g_i^2]\asymp \lambda_i
\quad\Longrightarrow\quad
(\widehat v_i+\epsilon)^{-1/2}\asymp (\lambda_i+\rho)^{-1/2},
\]

so frozen RMSProp behaves like one-pass SGD on a damped, spectrally flattened covariance.

---

## Setup

Work in the covariance eigenbasis.  Let

\[
H e_i=\lambda_i e_i,
\qquad
\lambda_i\asymp i^{-a},
\qquad
a>1.
\]

The data are Gaussian linear-regression samples

\[
x\sim \mathcal N(0,H),
\qquad
y=\langle x,w_\star\rangle+\xi,
\qquad
\xi\sim\mathcal N(0,\sigma^2),
\]

with \(\xi\) independent of \(x\).  Define the risk-source energies

\[
s_i:=\lambda_i w_{\star,i}^2,
\qquad
s_i\asymp i^{-b},
\qquad
b>1.
\]

Let

\[
\mathcal E_\star:=\sigma^2+\|w_\star\|_H^2,
\qquad
S_\star:=\sup_i s_i.
\]

We assume \(\mathcal E_\star>0\).  The case \(\mathcal E_\star=0\) is degenerate because it means both the noise and the target are zero in risk norm.

The model class is the span of the first \(M\) spectral coordinates.  Therefore the approximation error is

\[
A_M=\sum_{i>M}s_i\asymp M^{1-b}.
\]

---

## Frozen RMSProp estimator

Use an independent burn-in sample stream at initialization \(w_0=0\).  The squared-loss stochastic gradient is

\[
g(w)=(\langle x,w\rangle-y)x.
\]

At initialization,

\[
g_i(0)=-y x_i.
\]

We estimate the coordinatewise second moment by a median-of-means estimator.  Split \(B=RL\) independent burn-in samples into \(R\) blocks of length \(L\), and define

\[
\bar v_{r,i}
:=
\frac1L\sum_{\ell=1}^L g_{r,\ell,i}(0)^2,
\qquad
\widehat v_i
:=
\operatorname{median}_{1\le r\le R}\bar v_{r,i}.
\]

The frozen RMSProp preconditioner is

\[
\widehat P_i=(\widehat v_i+\epsilon)^{-1/2}.
\]

Then train on a fresh, independent one-pass sample stream with

\[
w_{t+1}
=
w_t-\gamma_t\widehat P(\langle x_t,w_t\rangle-y_t)x_t.
\]

The independence between burn-in samples and training samples is used only to make the conditional reduction to ordinary SGD immediate.

---

## Lemma 1: coordinate gradient second moment

Let

\[
e=w-w_\star.
\]

For squared loss,

\[
g_i(w)=(\langle x,e\rangle-\xi)x_i.
\]

Then, for Gaussian design,

\[
\boxed{
\mathbb E[g_i(w)^2\mid e]
=
\lambda_i(\sigma^2+\|e\|_H^2)
+
2\lambda_i^2e_i^2.
}
\]

### Proof

Set

\[
A:=\langle x,e\rangle-\xi,
\qquad
B:=x_i.
\]

Then \((A,B)\) is jointly Gaussian with

\[
\operatorname{Var}(A)=\sigma^2+\|e\|_H^2,
\qquad
\operatorname{Var}(B)=\lambda_i,
\qquad
\operatorname{Cov}(A,B)=\lambda_i e_i.
\]

By Isserlis' identity,

\[
\mathbb E[A^2B^2]
=
\operatorname{Var}(A)\operatorname{Var}(B)
+
2\operatorname{Cov}(A,B)^2.
\]

Substituting the three quantities above gives

\[
\mathbb E[g_i(w)^2\mid e]
=
\lambda_i(\sigma^2+\|e\|_H^2)
+
2\lambda_i^2e_i^2.
\]

---

## Lemma 2: population frozen RMSProp is spectrally damped

Define the population burn-in second moment

\[
v_i^\circ:=\mathbb E[g_i(0)^2].
\]

Then

\[
\boxed{
v_i^\circ
=
\lambda_i\mathcal E_\star+2\lambda_i s_i.
}
\]

Consequently

\[
\mathcal E_\star \lambda_i
\le
v_i^\circ
\le
(\mathcal E_\star+2S_\star)\lambda_i.
\]

Let

\[
\rho:=\epsilon/\mathcal E_\star.
\]

Then

\[
\boxed{
\mathcal E_\star(\lambda_i+\rho)
\le
v_i^\circ+\epsilon
\le
(\mathcal E_\star+2S_\star)(\lambda_i+\rho)
}
\]

up to an inessential enlargement of the upper constant.  Hence

\[
(v_i^\circ+\epsilon)^{-1/2}
\asymp
(\lambda_i+\rho)^{-1/2}.
\]

### Proof

Apply Lemma 1 at \(w=0\), so \(e=-w_\star\).  Then

\[
\mathbb E[g_i(0)^2]
=
\lambda_i(\sigma^2+\|w_\star\|_H^2)
+2\lambda_i^2 w_{\star,i}^2.
\]

Since \(s_i=\lambda_i w_{\star,i}^2\), this is

\[
v_i^\circ=\lambda_i\mathcal E_\star+2\lambda_i s_i.
\]

Because \(0\le s_i\le S_\star\),

\[
\mathcal E_\star\lambda_i
\le
v_i^\circ
\le
(\mathcal E_\star+2S_\star)\lambda_i.
\]

Adding \(\epsilon=\mathcal E_\star\rho\) gives the claimed spectral equivalence.

---

## Lemma 3: uniform robust second-moment concentration

There is a universal constant \(C\) such that if

\[
L\ge C,
\qquad
R\ge C\log(2M/\delta),
\]

then, with probability at least \(1-\delta\) over the burn-in sample stream,

\[
\boxed{
\frac12 v_i^\circ
\le
\widehat v_i
\le
\frac32 v_i^\circ
\qquad
\text{for all }1\le i\le M.
}
\]

### Proof

For a fixed coordinate \(i\), define

\[
F_i:=g_i(0),
\qquad
Z_i:=F_i^2.
\]

The random variable \(F_i\) is a degree-two polynomial of jointly Gaussian variables.  Gaussian hypercontractivity gives

\[
\mathbb E[F_i^4]
\le
C_4(\mathbb E[F_i^2])^2
=
C_4(v_i^\circ)^2
\]

for a universal constant \(C_4\).  Equivalently,

\[
\operatorname{Var}(Z_i)
\le
(C_4-1)(v_i^\circ)^2.
\]

For one block mean \(\bar v_{r,i}\), Chebyshev's inequality gives

\[
\mathbb P\left(
|\bar v_{r,i}-v_i^\circ|>\frac12 v_i^\circ
\right)
\le
\frac{4(C_4-1)}{L}.
\]

Taking \(L\) larger than a universal constant makes this probability at most \(1/8\).  Therefore a block is good with probability at least \(7/8\).  A Chernoff bound implies that the median of \(R\) block means is bad with probability at most \(2e^{-cR}\), for a universal \(c>0\).  A union bound over \(i\le M\) yields the result when \(R\ge C\log(2M/\delta)\).

---

## Theorem 3: frozen RMSProp scaling law

Let

\[
\rho:=\epsilon/\mathcal E_\star,
\qquad
n:=N_{\mathrm{eff}}\gamma,
\qquad
N_{\mathrm{eff}}:=N/\log N.
\]

Assume the burn-in estimator satisfies the event in Lemma 3.  Define

\[
\widehat P_i=(\widehat v_i+\epsilon)^{-1/2}.
\]

Train with frozen-preconditioned one-pass SGD

\[
w_{t+1}
=
w_t-\gamma_t\widehat P(\langle x_t,w_t\rangle-y_t)x_t
\]

on fresh training samples.  Suppose the step-size scale is below the usual transformed-covariance stability threshold and suppose, for the simplified power-law display, that

\[
1<b<a/2+1.
\]

Then, with probability at least \(1-\delta\) over the burn-in samples, the conditional training risk satisfies, up to constants and logarithmic factors,

\[
\boxed{
\mathbb E_{\mathrm{train}}[R_M(w_N)\mid \widehat v]-\sigma^2
\asymp
M^{1-b}
+
K_{\rho,1/2}(n)^{1-b}
+
\frac{\min\{M,K_{\rho,1/2}(n)\}}{N_{\mathrm{eff}}}.
}
\]

Here

\[
K_{\rho,1/2}(n)
\asymp
\begin{cases}
 n^{2/a}, & n\lesssim \rho^{-1/2},\\[4pt]
 \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2}.
\end{cases}
\]

Equivalently, before the damping knee,

\[
\mathbb E_{\mathrm{train}}[R_M(w_N)\mid \widehat v]-\sigma^2
\asymp
M^{1-b}
+
n^{-2(b-1)/a}
+
\frac{\min\{M,n^{2/a}\}}{N_{\mathrm{eff}}},
\qquad
n\lesssim \rho^{-1/2},
\]

while after the damping knee,

\[
\mathbb E_{\mathrm{train}}[R_M(w_N)\mid \widehat v]-\sigma^2
\asymp
M^{1-b}
+
\rho^{(b-1)/(2a)}n^{-(b-1)/a}
+
\frac{\min\{M,\rho^{-1/(2a)}n^{1/a}\}}{N_{\mathrm{eff}}},
\qquad
n\gtrsim \rho^{-1/2}.
\]

### More invariant filter form

The same theorem can be stated without immediately collapsing the answer to a power law.  Conditional on \(\widehat v\), define the transformed eigenvalues

\[
\widehat\mu_i
:=
\lambda_i\widehat P_i
=
\lambda_i(\widehat v_i+\epsilon)^{-1/2}.
\]

Then the inherited SGD risk proxy is

\[
\sum_{i>M}s_i
+
\sum_{i\le M}\frac{s_i}{1+n\widehat\mu_i}
+
\frac1{N_{\mathrm{eff}}}
\sum_{i\le M}\min\{1,n^2\widehat\mu_i^2\}.
\]

On the good burn-in event, this filter expression is equivalent, up to constants, to the same expression with

\[
\widehat\mu_i
\quad\text{replaced by}\quad
\lambda_i(\lambda_i+\rho)^{-1/2}.
\]

The displayed power-law theorem follows from evaluating this two-slope spectrum.

---

## Proof of Theorem 3

### Step 1: empirical preconditioner is spectrally equivalent to damped \(q=1/2\)

By Lemma 3, on the good burn-in event,

\[
\frac12v_i^\circ
\le
\widehat v_i
\le
\frac32v_i^\circ
\qquad
\text{for all }i\le M.
\]

Using Lemma 2, there exist constants \(c_-,c_+>0\), depending only on \(\mathcal E_\star\) and \(S_\star\), such that

\[
\boxed{
c_-(\lambda_i+\rho)
\le
\widehat v_i+\epsilon
\le
c_+(\lambda_i+\rho)
\qquad
\text{for all }i\le M.
}
\]

Therefore

\[
\widehat P_i
=(\widehat v_i+\epsilon)^{-1/2}
\asymp
(\lambda_i+\rho)^{-1/2}.
\]

The frozen RMSProp preconditioner is thus spectrally equivalent to

\[
P_{\rho,1/2}=(H+\rho I)^{-1/2}.
\]

### Step 2: reduce frozen RMSProp to ordinary SGD

Condition on the burn-in samples and hence on \(\widehat P\).  Because the training stream is fresh, \(\widehat P\) is deterministic relative to the training data.

Set

\[
w=\widehat P^{1/2}u,
\qquad
\widetilde x=\widehat P^{1/2}x.
\]

Then

\[
\langle x,w\rangle
=
\langle \widetilde x,u\rangle,
\]

and the frozen RMSProp update becomes ordinary SGD:

\[
u_{t+1}
=
u_t-\gamma_t(\langle \widetilde x_t,u_t\rangle-y_t)\widetilde x_t.
\]

The transformed covariance is

\[
\widehat H
=
\mathbb E[\widetilde x\widetilde x^\top\mid \widehat P]
=
\widehat P^{1/2}H\widehat P^{1/2}.
\]

Since \(\widehat P\) is diagonal in the spectral basis, the transformed eigenvalues are

\[
\widehat\mu_i
=
\lambda_i\widehat P_i
=
\lambda_i(\widehat v_i+\epsilon)^{-1/2}.
\]

By Step 1,

\[
\boxed{
\widehat\mu_i
\asymp
\lambda_i(\lambda_i+\rho)^{-1/2}.
}
\]

Thus

\[
\widehat\mu_i
\asymp
\begin{cases}
 i^{-a/2}, & i\lesssim \rho^{-1/a},\\[3pt]
 \rho^{-1/2}i^{-a}, & i\gtrsim \rho^{-1/a}.
\end{cases}
\]

This is exactly the damped two-slope spectrum from the oracle theorem with \(q=1/2\).

### Step 3: source energies are unchanged

The transformed target is

\[
u_\star=\widehat P^{-1/2}w_\star.
\]

Therefore

\[
\widehat\mu_i u_{\star,i}^2
=
\lambda_i\widehat P_i\cdot \widehat P_i^{-1}w_{\star,i}^2
=
\lambda_i w_{\star,i}^2
=s_i.
\]

So the source exponent remains \(b\), exactly as in the oracle preconditioning theorem.

### Step 4: apply the damped spectral theorem

The transformed problem is ordinary one-pass SGD with covariance eigenvalues \(\widehat\mu_i\) and source energies \(s_i\).  Since \(\widehat\mu_i\) is constant-equivalent to the \(q=1/2\) damped oracle spectrum, the learned dimension satisfies

\[
\widehat K(n)
:=
\#\{i:n\widehat\mu_i\gtrsim 1\}
\asymp
K_{\rho,1/2}(n).
\]

Evaluating this count gives

\[
K_{\rho,1/2}(n)
\asymp
\begin{cases}
 n^{2/a}, & n\lesssim \rho^{-1/2},\\[4pt]
 \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2}.
\end{cases}
\]

Applying the damped spectral preconditioning theorem with \(q=1/2\) yields

\[
\mathbb E_{\mathrm{train}}[R_M(w_N)\mid \widehat v]-\sigma^2
\asymp
M^{1-b}
+
K_{\rho,1/2}(n)^{1-b}
+
\frac{\min\{M,K_{\rho,1/2}(n)\}}{N_{\mathrm{eff}}}.
\]

Substituting the two formulas for \(K_{\rho,1/2}(n)\) gives the pre-knee and post-knee displays.  This proves the theorem.

---

## Interpretation

Frozen RMSProp is not merely a heuristic adaptive method in this model.  Its burn-in gradients estimate

\[
\mathbb E[g_i(0)^2]\asymp \lambda_i,
\]

so its preconditioner estimates

\[
(\lambda_i+\rho)^{-1/2}.
\]

Consequently, before the damping knee, the effective spectral exponent changes from \(a\) to \(a/2\).  This changes the optimization/bias exponent from

\[
n^{-(b-1)/a}
\]

to

\[
n^{-2(b-1)/a}.
\]

The price is a larger variance term, because the method learns more modes:

\[
K_{\rho,1/2}(n)\asymp n^{2/a}
\quad\text{instead of}\quad
n^{1/a}.
\]

The damping parameter controls the range over which the exponent improvement is active:

\[
n\lesssim \rho^{-1/2}.
\]

After this knee, the tail spectrum reverts to exponent \(a\), with a \(\rho\)-dependent constant improvement.

---

## What this theorem does not yet prove

This is a frozen-preconditioner theorem.  It still does not prove the fully online RMSProp/Adam recursion

\[
v_{t+1}=\beta_2v_t+(1-\beta_2)g_t^2,
\qquad
w_{t+1}=w_t-\eta(v_t+\epsilon)^{-1/2}g_t.
\]

The next theorem should replace the independent burn-in estimator by a tracking argument showing that, over the relevant time window,

\[
v_{t,i}\approx c_t\lambda_i,
\qquad
c_t:=\sigma^2+\|w_t-w_\star\|_H^2,
\]

so that

\[
(v_{t,i}+\epsilon)^{-1/2}
\approx
c_t^{-1/2}(\lambda_i+\epsilon/c_t)^{-1/2}.
\]

That would turn the fixed damping \(\rho\) in this note into a time-varying damping floor

\[
\rho_t=\epsilon/c_t.
\]
