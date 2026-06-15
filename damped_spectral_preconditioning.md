# Damped spectral preconditioning theorem

This note proves the next result in the roadmap: fixed damped spectral preconditioning

\[
P_{\rho,q}=(H+\rho I)^{-q},\qquad 0\le q<1,\quad \rho>0.
\]

The damping parameter \(\rho\) is the clean spectral analogue of Adam/RMSProp's \(\epsilon\)-floor. The main Adam/RMSProp proxy is \(q=1/2\).

---

## Setup

Work in the population covariance eigenbasis:

\[
H e_i=\lambda_i e_i,\qquad \lambda_i\asymp i^{-a},\qquad a>1.
\]

The data are Gaussian linear-regression samples

\[
y=\langle x,w_\star\rangle+\xi,
\qquad
\mathbb E[xx^\top]=H,
\qquad
\mathbb E[\xi^2]=\sigma^2.
\]

Assume the source condition

\[
s_i:=\lambda_i(w_{\star,i})^2\asymp i^{-b},\qquad b>1.
\]

The model class is the span of the first \(M\) spectral coordinates, so the approximation error is

\[
A_M=\sum_{i>M}s_i\asymp M^{1-b}.
\]

Train with one-pass preconditioned SGD

\[
w_{t+1}
=
w_t-\gamma_t P_{\rho,q}
(\langle x_t,w_t\rangle-y_t)x_t,
\qquad
P_{\rho,q}=(H+\rho I)^{-q}.
\]

Let

\[
n:=N_{\mathrm{eff}}\gamma,
\qquad
N_{\mathrm{eff}}:=N/\log N,
\]

where \(\gamma\) is the scale of the step-size schedule.

---

## Effective-dimension SGD lemma used as a black box

We use the following Lin-style effective-dimension risk form for ordinary one-pass SGD. This is the part inherited from the original SGD scaling-law theorem. The new work in this note is the exact reduction from damped preconditioned SGD to ordinary SGD plus the evaluation of the damped spectrum.

**Lemma 1.** Consider ordinary one-pass SGD on a Gaussian linear-regression problem with covariance eigenvalues \(\mu_i\) and target energies

\[
\tilde s_i:=\mu_i(u_{\star,i})^2\asymp i^{-b}.
\]

Let

\[
K(n):=\#\{i:\ n\mu_i\gtrsim 1\}.
\]

In the non-very-smooth regime where the usual last-iterate SGD bias estimate is sharp, ordinary SGD satisfies, up to constants and logarithmic factors,

\[
\mathbb E R_M(u_N)-\sigma^2
\asymp
\sum_{i>M}\tilde s_i
+
K(n)^{1-b}
+
\frac{\min\{M,K(n)\}}{N_{\mathrm{eff}}}.
\]

For a pure power-law spectrum \(\mu_i\asymp i^{-\alpha}\), this recovers

\[
M^{1-b}
+
n^{-(b-1)/\alpha}
+
\frac{\min\{M,n^{1/\alpha}\}}{N_{\mathrm{eff}}}.
\]

For the theorem below, a clean sufficient condition for staying in this non-very-smooth regime is

\[
1<b<a(1-q)+1.
\]

The reduction and spectral-counting parts remain valid outside this range, but the last-iterate SGD bias term needs a separate refinement in the very-smooth regime.

---

## Theorem

Define

\[
\alpha:=a(1-q),
\qquad
n_\rho:=\rho^{-(1-q)},
\qquad
K_\rho:=\rho^{-1/a}.
\]

Also define the damped learned dimension

\[
K_{\rho,q}(n)
:=
\#\left\{i:\ n\lambda_i(\lambda_i+\rho)^{-q}\gtrsim 1\right\}.
\]

Then

\[
K_{\rho,q}(n)
\asymp
\begin{cases}
 n^{1/[a(1-q)]}, & n\lesssim \rho^{-(1-q)},\\[4pt]
 \rho^{-q/a}n^{1/a}, & n\gtrsim \rho^{-(1-q)}.
\end{cases}
\]

**Theorem 2, damped spectral preconditioning.** Suppose

\[
\lambda_i\asymp i^{-a},
\qquad
s_i=\lambda_i(w_{\star,i})^2\asymp i^{-b},
\qquad
0\le q<1,
\qquad
1<b<a(1-q)+1.
\]

Then one-pass SGD with fixed preconditioner \(P_{\rho,q}=(H+\rho I)^{-q}\) satisfies

\[
\boxed{
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+
K_{\rho,q}(N_{\mathrm{eff}}\gamma)^{1-b}
+
\frac{\min\{M,K_{\rho,q}(N_{\mathrm{eff}}\gamma)\}}{N_{\mathrm{eff}}}.
}
\]

Equivalently, in the preconditioned phase \(n\lesssim n_\rho\),

\[
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+
n^{-(b-1)/[a(1-q)]}
+
\frac{\min\{M,n^{1/[a(1-q)]}\}}{N_{\mathrm{eff}}}.
\]

In the damped-tail phase \(n\gtrsim n_\rho\),

\[
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+
\rho^{q(b-1)/a}n^{-(b-1)/a}
+
\frac{\min\{M,\rho^{-q/a}n^{1/a}\}}{N_{\mathrm{eff}}}.
\]

---

## Proof

### Step 1: Reduce preconditioned SGD to ordinary SGD

Let

\[
w=P_{\rho,q}^{1/2}u,
\qquad
\tilde x=P_{\rho,q}^{1/2}x.
\]

Then

\[
\langle x,w\rangle
=
\langle x,P_{\rho,q}^{1/2}u\rangle
=
\langle P_{\rho,q}^{1/2}x,u\rangle
=
\langle \tilde x,u\rangle.
\]

The transformed covariance is

\[
\tilde H
=
\mathbb E[\tilde x\tilde x^\top]
=
P_{\rho,q}^{1/2}HP_{\rho,q}^{1/2}.
\]

Since \(P_{\rho,q}\) is a spectral function of \(H\), it commutes with \(H\). Hence

\[
\tilde H=H(H+\rho I)^{-q}.
\]

The preconditioned update becomes

\[
u_{t+1}
=
u_t-
\gamma_t(\langle \tilde x_t,u_t\rangle-y_t)\tilde x_t,
\]

which is ordinary SGD on \((\tilde x_t,y_t)\).

The transformed target is

\[
u_\star=P_{\rho,q}^{-1/2}w_\star
=(H+\rho I)^{q/2}w_\star.
\]

The risk is invariant:

\[
R(w)-\sigma^2
=
\|w-w_\star\|_H^2
=
\|u-u_\star\|_{\tilde H}^2
=
\tilde R(u)-\sigma^2.
\]

So the problem is exactly ordinary SGD with covariance \(\tilde H\).

### Step 2: Compute the transformed spectrum

The transformed eigenvalues are

\[
\mu_i=\lambda_i(\lambda_i+\rho)^{-q}.
\]

Let

\[
i_\rho:=\rho^{-1/a}.
\]

For \(i\lesssim i_\rho\), \(\lambda_i\gtrsim \rho\), so

\[
\mu_i
\asymp
\lambda_i^{1-q}
\asymp
i^{-a(1-q)}.
\]

For \(i\gtrsim i_\rho\), \(\lambda_i\lesssim \rho\), so

\[
\mu_i
\asymp
\rho^{-q}\lambda_i
\asymp
\rho^{-q}i^{-a}.
\]

Thus

\[
\boxed{
\mu_i
\asymp
\begin{cases}
 i^{-a(1-q)}, & i\lesssim \rho^{-1/a},\\[3pt]
 \rho^{-q}i^{-a}, & i\gtrsim \rho^{-1/a}.
\end{cases}
}
\]

At the knee,

\[
\mu_{i_\rho}\asymp \rho^{1-q},
\]

so the time scale needed to reach the knee is

\[
n_\rho\asymp \mu_{i_\rho}^{-1}\asymp \rho^{-(1-q)}.
\]

### Step 3: Count the learned modes

By definition,

\[
K_{\rho,q}(n)=\#\{i:\ n\mu_i\gtrsim 1\}.
\]

If \(n\lesssim n_\rho\), the threshold lies before the damping knee. Solving

\[
i^{-a(1-q)}\asymp n^{-1}
\]

gives

\[
K_{\rho,q}(n)\asymp n^{1/[a(1-q)]}.
\]

If \(n\gtrsim n_\rho\), the threshold lies after the damping knee. Solving

\[
\rho^{-q}i^{-a}\asymp n^{-1}
\]

gives

\[
K_{\rho,q}(n)\asymp \rho^{-q/a}n^{1/a}.
\]

The two formulas match at the crossover:

\[
\left(\rho^{-(1-q)}\right)^{1/[a(1-q)]}
=
\rho^{-1/a}
=
\rho^{-q/a}\left(\rho^{-(1-q)}\right)^{1/a}.
\]

### Step 4: Check the source condition

The transformed target coordinate is

\[
u_{\star,i}=(\lambda_i+\rho)^{q/2}w_{\star,i}.
\]

Therefore

\[
\mu_i u_{\star,i}^2
=
\lambda_i(\lambda_i+\rho)^{-q}
\cdot
(\lambda_i+\rho)^q w_{\star,i}^2
=
\lambda_i w_{\star,i}^2
=s_i.
\]

Thus the source exponent is unchanged:

\[
\tilde s_i:=\mu_i u_{\star,i}^2\asymp i^{-b}.
\]

The approximation error is also unchanged:

\[
\sum_{i>M}\tilde s_i
=
\sum_{i>M}s_i
\asymp M^{1-b}.
\]

### Step 5: Apply Lemma 1

Lemma 1 applied to the transformed ordinary-SGD problem gives

\[
\mathbb E R_M(w_N)-\sigma^2
=
\mathbb E\tilde R_M(u_N)-\sigma^2
\asymp
M^{1-b}
+
K_{\rho,q}(n)^{1-b}
+
\frac{\min\{M,K_{\rho,q}(n)\}}{N_{\mathrm{eff}}}.
\]

Substituting the two formulas for \(K_{\rho,q}(n)\) gives the two phase formulas. This proves Theorem 2.

---

## Adam/RMSProp proxy: \(q=1/2\)

For Adam/RMSProp-like second-moment preconditioning, the idealized spectral exponent is

\[
q=\frac12.
\]

Then

\[
K_{\rho,1/2}(n)
\asymp
\begin{cases}
 n^{2/a}, & n\lesssim \rho^{-1/2},\\[4pt]
 \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2}.
\end{cases}
\]

The risk law becomes

\[
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+
K_{\rho,1/2}(n)^{1-b}
+
\frac{\min\{M,K_{\rho,1/2}(n)\}}{N_{\mathrm{eff}}}.
\]

Before the damping knee,

\[
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+
n^{-2(b-1)/a}
+
\frac{\min\{M,n^{2/a}\}}{N_{\mathrm{eff}}}.
\]

After the damping knee,

\[
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+
\rho^{(b-1)/(2a)}n^{-(b-1)/a}
+
\frac{\min\{M,\rho^{-1/(2a)}n^{1/a}\}}{N_{\mathrm{eff}}}.
\]

Thus damping resolves the trace-class pathology of undamped \(q=1/2\). The algorithm initially behaves as if the spectrum exponent were \(a/2\), then reverts after the damping knee to tail exponent \(a\), with \(\rho\)-dependent constants.

---

## Compute-allocation corollary, ignoring variance

Set \(C=MN\) and take \(n\asymp N\).

In the early preconditioned phase, the effective spectral exponent is

\[
\alpha=a(1-q).
\]

Balancing

\[
M^{-(b-1)}
\quad\text{and}\quad
N^{-(b-1)/\alpha}
\]

under \(C=MN\) gives

\[
M_\star\asymp C^{1/(\alpha+1)},
\qquad
N_\star\asymp C^{\alpha/(\alpha+1)},
\]

and

\[
R(C)-\sigma^2\asymp C^{-(b-1)/(\alpha+1)}.
\]

This early-phase allocation is valid only while

\[
N_\star\lesssim \rho^{-(1-q)}.
\]

In the late damped-tail phase, the exponent returns to \(a\). Balancing

\[
M^{-(b-1)}
\quad\text{and}\quad
\rho^{q(b-1)/a}N^{-(b-1)/a}
\]

under \(C=MN\) gives

\[
M_\star\asymp \rho^{-q/(a+1)}C^{1/(a+1)},
\qquad
N_\star\asymp \rho^{q/(a+1)}C^{a/(a+1)},
\]

and

\[
R(C)-\sigma^2
\asymp
\rho^{q(b-1)/(a+1)}C^{-(b-1)/(a+1)}.
\]

Thus a fixed damping floor gives a transient exponent improvement and an asymptotic constant improvement. To get a persistent exponent change as \(C\to\infty\), the effective damping floor must move with scale, which is plausible for online Adam/RMSProp because \(\rho_t\approx \epsilon/c_t\) depends on the evolving gradient second moment.

---

## Next theorem target: frozen RMSProp

For squared loss,

\[
g_i=(\langle x,e\rangle-\xi)x_i,
\qquad e=w-w_\star.
\]

For Gaussian design,

\[
\mathbb E[g_i^2\mid e]
=
\lambda_i(\sigma^2+\|e\|_H^2)
+
2\lambda_i^2e_i^2.
\]

In a noise-dominated or global-residual-dominated regime,

\[
\mathbb E[g_i^2\mid e]\asymp c_t\lambda_i.
\]

Then

\[
(v_i+\epsilon)^{-1/2}
\approx
(c_t\lambda_i+\epsilon)^{-1/2}
=
c_t^{-1/2}(\lambda_i+\epsilon/c_t)^{-1/2}.
\]

So frozen RMSProp should behave like

\[
P_{\rho,1/2}=(H+\rho I)^{-1/2},
\qquad
\rho\approx \epsilon/c_t.
\]

The frozen-RMSProp theorem should prove a uniform concentration statement of the form

\[
c_1(\lambda_i+\rho)
\le
\widehat v_i+\epsilon
\le
c_2(\lambda_i+\rho)
\]

for all coordinates up to the relevant learned dimension. Once that is proven, Theorem 2 applies directly with \(q=1/2\).
