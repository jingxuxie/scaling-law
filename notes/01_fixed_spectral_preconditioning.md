# Fixed spectral preconditioning baseline

## Setup

Let `H e_i = lambda_i e_i`, with

```math
\lambda_i \asymp i^{-a}, \qquad a>1,
```

and source condition

```math
s_i := \mathbb E[\lambda_i (w_i^\star)^2] \asymp i^{-b}, \qquad b>1.
```

For a fixed spectral preconditioner

```math
P_q = H^{-q}, \qquad 0 \le q < 1,
```

consider one-pass preconditioned SGD

```math
w_{t+1}=w_t-\gamma_t P_q(\langle x_t,w_t\rangle-y_t)x_t.
```

## Reparameterization

Set `w=P_q^{1/2}u` and `\widetilde x=P_q^{1/2}x`. Then

```math
\langle x,w\rangle=\langle \widetilde x,u\rangle,
```

and preconditioned SGD in `w` is ordinary SGD in `u`:

```math
u_{t+1}=u_t-\gamma_t(\langle \widetilde x_t,u_t\rangle-y_t)\widetilde x_t.
```

The transformed covariance is

```math
\widetilde H=P_q^{1/2}HP_q^{1/2}=H^{1-q},
```

so

```math
\widetilde\lambda_i = \lambda_i^{1-q} \asymp i^{-\alpha},
\qquad \alpha=a(1-q).
```

The target transforms as `u^\star=P_q^{-1/2}w^\star`, and its source energy is unchanged:

```math
\widetilde\lambda_i (u_i^\star)^2
= \lambda_i^{1-q}\lambda_i^q(w_i^\star)^2
= \lambda_i(w_i^\star)^2.
```

Thus fixed spectral preconditioning changes only the covariance exponent

```math
a \mapsto \alpha=a(1-q),
```

while preserving the source exponent `b`.

## Consequence

In the clean source range

```math
1<b<\alpha+1,
```

and under the same one-pass SGD conditions as the base linear-regression scaling-law theorem,

```math
\mathbb E R_M(w_N)-\sigma^2
\asymp
M^{1-b}
+(N_{\rm eff}\gamma)^{(1-b)/\alpha}
+\frac{\min\{M,(N_{\rm eff}\gamma)^{1/\alpha}\}}{N_{\rm eff}}.
```

Equivalently, the learned mode count is

```math
K_q(n)\asymp n^{1/\alpha}, \qquad n=N_{\rm eff}\gamma,
```

and the risk is

```math
M^{1-b}+K_q(n)^{1-b}+\frac{\min\{M,K_q(n)\}}{N_{\rm eff}}.
```

This is the baseline theorem. The damped theorem in `02_damped_spectral_preconditioning.md` is the version needed for Adam/RMSProp.
