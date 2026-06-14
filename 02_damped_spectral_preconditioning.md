# Damped spectral preconditioning theorem

This note proves the next theorem in the roadmap: the fixed spectral preconditioner

```math
P_q=H^{-q}
```

is replaced by the Adam/RMSProp-like damped spectral preconditioner

```math
P_{\rho,q}=(H+\rho I)^{-q}, \qquad \rho>0.
```

The damping parameter is the spectral analog of Adam/RMSProp's `epsilon`.

---

## 1. Setup

Work in the eigenbasis of the population covariance:

```math
H e_i = \lambda_i e_i, \qquad \lambda_i\asymp i^{-a}, \qquad a>1.
```

Assume the source condition

```math
s_i:=\mathbb E[\lambda_i(w_i^\star)^2]\asymp i^{-b}, \qquad b>1.
```

For fixed `q in [0,1)` and `rho in (0,1]`, consider

```math
w_{t+1}=w_t-\gamma_t P_{\rho,q}(\langle x_t,w_t\rangle-y_t)x_t,
\qquad
P_{\rho,q}=(H+\rho I)^{-q}.
```

Let

```math
n:=N_{\rm eff}\gamma, \qquad N_{\rm eff}:=N/\log N,
```

where `gamma_t` is the geometric step-decay schedule used in the base one-pass SGD theory.

Define the preconditioned exponent

```math
\alpha:=a(1-q).
```

---

## 2. Reparameterization

Set

```math
w=P_{\rho,q}^{1/2}u, \qquad \widetilde x=P_{\rho,q}^{1/2}x.
```

Then

```math
\langle x,w\rangle=\langle \widetilde x,u\rangle,
```

and preconditioned SGD in `w` is exactly ordinary SGD in `u`:

```math
u_{t+1}=u_t-\gamma_t(\langle \widetilde x_t,u_t\rangle-y_t)\widetilde x_t.
```

The transformed covariance is diagonal with eigenvalues

```math
\mu_i
:=\lambda_i(\lambda_i+\rho)^{-q}.
```

The transformed target is

```math
u^\star=P_{\rho,q}^{-1/2}w^\star=(H+\rho I)^{q/2}w^\star.
```

The source energy is unchanged:

```math
\mu_i(u_i^\star)^2
=\lambda_i(\lambda_i+\rho)^{-q}(\lambda_i+\rho)^q(w_i^\star)^2
=\lambda_i(w_i^\star)^2.
```

Therefore the preconditioned problem is ordinary SGD on a transformed covariance spectrum `mu_i`, but with the same source-energy sequence `s_i`.

---

## 3. Effective two-slope spectrum

Let

```math
m_\rho:=\rho^{-1/a}.
```

This is the spectral index at which `lambda_i` crosses `rho`. Since `lambda_i \asymp i^{-a}`,

```math
\mu_i
=\lambda_i(\lambda_i+\rho)^{-q}
\asymp
\begin{cases}
 i^{-a(1-q)}=i^{-\alpha}, & i\lesssim m_\rho,\\[3pt]
 \rho^{-q}i^{-a}, & i\gtrsim m_\rho.
\end{cases}
```

At the knee,

```math
\mu_{m_\rho}\asymp \rho^{1-q}.
```

Therefore define the sample-size knee

```math
n_\rho:=\mu_{m_\rho}^{-1}\asymp \rho^{-(1-q)}.
```

The learned-mode count is

```math
K_{\rho,q}(n):=\#\{i:\mu_i\gtrsim n^{-1}\}.
```

Solving `mu_i \asymp n^{-1}` gives

```math
K_{\rho,q}(n)
\asymp
\begin{cases}
 n^{1/\alpha}, & n\lesssim n_\rho,\\[3pt]
 \rho^{-q/a}n^{1/a}, & n\gtrsim n_\rho.
\end{cases}
```

Thus preconditioning first makes SGD behave as though the covariance exponent were `alpha=a(1-q)`. After the damping knee, the tail reverts to the original exponent `a`, with a multiplicative gain `rho^{-q/a}` in the number of learned modes.

---

## 4. Stability trace

The transformed trace is

```math
\operatorname{tr}(\widetilde H)=\sum_i \mu_i.
```

Using the two-slope form,

```math
\operatorname{tr}(\widetilde H)
\asymp
\begin{cases}
 1, & \alpha>1,\\
 \log(1/\rho), & \alpha=1,\\
 \rho^{-(1-\alpha)/a}, & \alpha<1.
\end{cases}
```

Hence the SGD stability condition should be written as

```math
\gamma \lesssim \frac{1}{\operatorname{tr}(\widetilde H)}.
```

This is an important difference from undamped `H^{-q}`. Damping makes the trace finite for every `q<1`, but if `alpha<1`, the largest stable constant step size shrinks as `rho -> 0`.

---

## 5. Bias theorem

Let

```math
A_M:=\sum_{i>M}s_i\asymp M^{1-b}
```

be the approximation term.

The algorithmic bias in the transformed coordinates has the same form as the base SGD bias, but with covariance eigenvalues `mu_i` and target coordinates `u_i^star`. The low-frequency norm appearing in the usual bias upper bound is

```math
\mathbb E[(u_i^\star)^2]
=\frac{s_i}{\mu_i}.
```

Indeed,

```math
\mu_i\mathbb E[(u_i^\star)^2]=s_i.
```

For a cutoff `k`, the standard step-decay SGD bias argument gives

```math
B_M(n)
\lesssim
\frac{1}{n}\sum_{i\le k}\frac{s_i}{\mu_i}
+\sum_{k<i\le M}s_i,
```

with the matching lower bound

```math
B_M(n)\gtrsim \sum_{i: \mu_i\lesssim n^{-1},\ i\le M}s_i.
```

Take `k = M \wedge K_{rho,q}(n)`. In the clean source range

```math
1<b<\alpha+1,
```

the first term is of the same order as the tail term. Therefore

```math
B_M(n)\asymp \min\{M,K_{\rho,q}(n)\}^{1-b}
```

whenever the cutoff is below the model dimension; together with approximation this is equivalently

```math
A_M+B_M(n)\asymp M^{1-b}+K_{\rho,q}(n)^{1-b}.
```

### Proof of the bias estimate

Let `K=K_{rho,q}(n)`.

**Case 1: `n \lesssim n_rho`, so `K \lesssim m_rho`.**

Here `mu_i \asymp i^{-alpha}` for all `i <= K`, and

```math
\frac{s_i}{\mu_i}\asymp \frac{i^{-b}}{i^{-\alpha}}=i^{\alpha-b}.
```

Since `b<alpha+1`,

```math
\frac{1}{n}\sum_{i\le K}i^{\alpha-b}
\asymp
\frac{K^{\alpha-b+1}}{n}.
```

Because `K\asymp n^{1/alpha}`, equivalently `n\asymp K^alpha`, this equals

```math
K^{1-b}.
```

The tail source energy is

```math
\sum_{i>K}i^{-b}\asymp K^{1-b}.
```

So `B_M(n) \asymp K^{1-b}`.

**Case 2: `n \gtrsim n_rho`, so `K \gtrsim m_rho`.**

Split the low-frequency norm at `m=m_rho`:

```math
\frac{1}{n}\sum_{i\le K}\frac{s_i}{\mu_i}
=
\frac{1}{n}\sum_{i\le m}\frac{s_i}{\mu_i}
+
\frac{1}{n}\sum_{m<i\le K}\frac{s_i}{\mu_i}.
```

For `i<=m`, `s_i/mu_i \asymp i^{alpha-b}`. Since `b<alpha+1`,

```math
\frac{1}{n}\sum_{i\le m}i^{\alpha-b}
\asymp \frac{m^{\alpha-b+1}}{n}.
```

For `i>m`, `mu_i \asymp rho^{-q}i^{-a}`, hence

```math
\frac{s_i}{\mu_i}\asymp \rho^q i^{a-b}.
```

Since `b<a+1` follows from `b<alpha+1` and `alpha<=a`,

```math
\frac{1}{n}\sum_{m<i\le K}\rho^q i^{a-b}
\asymp
\frac{\rho^q K^{a-b+1}}{n}.
```

In the tail phase `K\asymp rho^{-q/a}n^{1/a}`, equivalently `n\asymp rho^q K^a`, so this term is

```math
K^{1-b}.
```

The head contribution is no larger. Indeed, at the knee the two expressions are equal up to constants, and for `n>=n_rho` the tail-phase `K^{1-b}` decays more slowly than the head contribution under `b<alpha+1`. Therefore the full low-frequency term is `O(K^{1-b})`.

The tail source energy is again

```math
\sum_{i>K}i^{-b}\asymp K^{1-b}.
```

The upper and lower bounds match, proving the bias estimate.

### What happens outside the clean source range

If `b >= alpha+1`, the above proof no longer gives a matching upper bound because

```math
\sum_{i\le K} i^{\alpha-b}
```

is logarithmic at equality and bounded for `b>alpha+1`. The safe bracket is

```math
K_{\rho,q}(n)^{1-b}
\lesssim B_M(n)
\lesssim K_{\rho,q}(n)^{1-b}
+\frac{1}{n}\sum_{i\le M\wedge K_{\rho,q}(n)}\frac{s_i}{\mu_i}.
```

At `b=alpha+1`, the extra term is logarithmic. For `b>alpha+1`, it is a qualification/saturation term of order roughly `n^{-1}` during the pre-damping phase. This is the same type of simple-source gap already visible in the base theory when `b>=a+1`.

---

## 6. Variance theorem

The sharp SGD variance term is controlled by the effective dimension

```math
\mathcal D_M(n)
:=
\#\{i\le M:\mu_i\ge n^{-1}\}
+n^2\sum_{i\le M:\mu_i<n^{-1}}\mu_i^2.
```

The variance contribution is

```math
V_M(n)\asymp \frac{\mathcal D_M(n)}{N_{\rm eff}}.
```

For `M` larger than the learned cutoff, write `D(n)=D_infty(n)`. Then

```math
\mathcal D(n)
\asymp
\begin{cases}
 K_{\rho,q}(n), & \alpha>1/2,\ n\lesssim n_\rho,\\[3pt]
 K_{\rho,q}(n)\bigl[1+\log(m_\rho/K_{\rho,q}(n))\bigr], & \alpha=1/2,\ n\lesssim n_\rho,\\[3pt]
 n^2\rho^{-(1-2\alpha)/a}, & \alpha<1/2,\ n\lesssim n_\rho,\\[3pt]
 K_{\rho,q}(n), & n\gtrsim n_\rho.
\end{cases}
```

In particular, for the Adam/RMSProp-like value `q=1/2`,

```math
\alpha=a/2>1/2
```

because `a>1`. Therefore

```math
\mathcal D(n)\asymp K_{\rho,1/2}(n)
```

throughout both phases.

### Proof of the variance estimate

The first term in `D(n)` is the learned-mode count `K`.

If `n <= n_rho`, then `K<=m=m_rho`. The squared-tail contribution is

```math
n^2\sum_{K<i\le m}i^{-2\alpha}
+n^2\rho^{-2q}\sum_{i>m}i^{-2a}.
```

If `alpha>1/2`, the first sum is `asymp K^{1-2alpha}`, and multiplying by `n^2` gives

```math
n^2K^{1-2\alpha}\asymp K,
```

because `n\asymp K^alpha`. The second sum is

```math
n^2\rho^{-2q}m^{1-2a}
\asymp n^2\rho^{(2\alpha-1)/a},
```

which is at most `K` for `n<=n_rho` and equal to `K` at the knee. Hence `D(n)\asymp K`.

If `alpha=1/2`, the first squared-tail sum gives the logarithmic factor

```math
n^2\log(m/K)=K\log(m/K).
```

If `alpha<1/2`, the squared-tail sum is dominated by the upper endpoint `m`, giving

```math
n^2m^{1-2\alpha}=n^2\rho^{-(1-2\alpha)/a}.
```

If `n>=n_rho`, then `K>=m`, and the only tail is the original damped tail

```math
n^2\rho^{-2q}\sum_{i>K}i^{-2a}
\asymp n^2\rho^{-2q}K^{1-2a}.
```

Using `n\asymp rho^q K^a`, this equals `K`. Thus `D(n)\asymp K` after the damping knee.

---

## 7. Main theorem

Assume

```math
1<b<\alpha+1, \qquad \alpha=a(1-q),
```

and the step size satisfies

```math
\gamma\lesssim 1/\operatorname{tr}(\widetilde H).
```

Then, up to the same logarithmic factors hidden in `N_eff=N/log N` and the same constant-factor convention as the base one-pass SGD theorem,

```math
\boxed{
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+K_{\rho,q}(N_{\rm eff}\gamma)^{1-b}
+\frac{\mathcal D_M(N_{\rm eff}\gamma)}{N_{\rm eff}}.
}
```

Here

```math
K_{\rho,q}(n)
\asymp
\begin{cases}
 n^{1/[a(1-q)]}, & n\lesssim \rho^{-(1-q)},\\[3pt]
 \rho^{-q/a}n^{1/a}, & n\gtrsim \rho^{-(1-q)},
\end{cases}
```

and `D_M` is the effective dimension above. If `alpha>1/2` and `M` is not the bottleneck,

```math
\mathcal D_M(n)\asymp \min\{M,K_{\rho,q}(n)\}.
```

Therefore, in the Adam/RMSProp-like case `q=1/2`,

```math
\boxed{
K_{\rho,1/2}(n)
\asymp
\begin{cases}
 n^{2/a}, & n\lesssim \rho^{-1/2},\\[3pt]
 \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2},
\end{cases}
}
```

and

```math
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+K_{\rho,1/2}(n)^{1-b}
+\frac{\min\{M,K_{\rho,1/2}(n)\}}{N_{\rm eff}}
```

in the clean range `1<b<a/2+1`.

---

## 8. Interpretation

Damped preconditioning creates a two-stage scaling law.

1. **Flattened phase:** for `n \lesssim rho^{-(1-q)}`, the optimizer behaves as if the spectrum exponent were `a(1-q)`.
2. **Damping-limited phase:** for `n \gtrsim rho^{-(1-q)}`, the tail exponent reverts to `a`, but with a larger learned-mode count by the factor `rho^{-q/a}`.

For Adam/RMSProp's idealized `q=1/2`, this means early training learns modes according to exponent `a/2`, then crosses back to exponent `a` after the epsilon/damping knee.

This is the first theorem that genuinely needs damping. It also shows why the undamped `q=1/2` proxy is not sufficient: without damping, the effective spectrum `i^{-a/2}` may have an infinite trace when `a<=2`, while the damped theorem remains well-defined.

---

## 9. Next theorem to prove

The next target is a frozen-RMSProp theorem.

Estimate

```math
\widehat v_i=\frac1B\sum_{t=1}^B g_{t,i}^2,
```

freeze

```math
P_i=(\widehat v_i+\epsilon)^{-1/2},
```

and prove that, on the learned coordinate range,

```math
\widehat v_i+\epsilon \asymp c_t\lambda_i+\epsilon.
```

Then frozen RMSProp reduces to this damped theorem with

```math
q=1/2, \qquad \rho\asymp \epsilon/c_t.
```
