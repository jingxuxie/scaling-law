# 18. Compute-optimal visible-spectrum scaling law

The previous notes identify the effective spectrum produced by a visible diagonal adaptive optimizer.  This note turns that spectrum into a compute-optimal scaling law.

The point is to separate two effects:

\[
\text{learning-rate tuning changes constants,}
\]

while

\[
\text{visible spectral preconditioning changes the compute exponent.}
\]

## 1. Visible spectrum

Assume

\[
    \lambda_i\asymp i^{-a},
    \qquad a>1,
\]

and suppose the adaptive optimizer has visible exponent \(\theta\in[0,1]\).  Before the damping knee, the effective covariance eigenvalues are

\[
    \mu_i
    \asymp
    \lambda_i^{1-\theta/2}
    \asymp
    i^{-\alpha(\theta)},
\]

where

\[
\boxed{
    \alpha(\theta)=a(1-\theta/2).
}
\]

Equivalently,

\[
    q_{\rm eff}=\theta/2.
\]

The two endpoints are

\[
    \theta=0 \Rightarrow \alpha=a
\]

for scalar/flat preconditioning, and

\[
    \theta=1 \Rightarrow \alpha=a/2
\]

for aligned or band-limited Adam/RMSProp.

## 2. Risk law in the hard-source phase

Assume the source condition

\[
    s_i\asymp i^{-b},
    \qquad 1<b\le \alpha(\theta),
\]

so the variance term is lower order after the usual balancing.  The risk proxy is

\[
\boxed{
    R(M,N)-\sigma^2
    \asymp
    M^{1-b}
    +
    N^{-(b-1)/\alpha(\theta)}.
}
\]

Here \(M\) is model/feature dimension and \(N\) is the effective sample/step budget, ignoring logarithms.

## 3. Compute-optimal allocation

Let compute be

\[
    C\asymp MN.
\]

Substitute \(N=C/M\):

\[
    R(M,C/M)-\sigma^2
    \asymp
    M^{1-b}
    +
    C^{-(b-1)/\alpha}M^{(b-1)/\alpha},
\]

where \(\alpha=\alpha(\theta)\).  Balance the two terms:

\[
    M^{-(b-1)}
    \asymp
    C^{-(b-1)/\alpha}M^{(b-1)/\alpha}.
\]

Cancel \(b-1>0\):

\[
    M^{-1}\asymp C^{-1/\alpha}M^{1/\alpha}.
\]

Thus

\[
    M^{1+1/\alpha}\asymp C^{1/\alpha},
\]

so

\[
\boxed{
    M_\star(C)\asymp C^{1/(\alpha+1)}.
}
\]

Consequently,

\[
\boxed{
    N_\star(C)\asymp C^{\alpha/(\alpha+1)}.
}
\]

The compute-optimal risk is

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-(b-1)/(\alpha+1)}.
}
\]

Substituting \(\alpha(\theta)=a(1-\theta/2)\),

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-\beta(\theta)},
    \qquad
    \beta(\theta)=\frac{b-1}{a(1-\theta/2)+1}.
}
\]

## 4. Visible adaptivity improves the compute exponent

Differentiate:

\[
    \beta'(\theta)
    =
    \frac{(b-1)(a/2)}{(a(1-\theta/2)+1)^2}>0.
\]

Therefore the compute exponent is strictly increasing in the visible exponent \(\theta\).  In particular,

\[
\boxed{
    \beta(1)-\beta(0)
    =
    (b-1)
    \left(
        \frac1{a/2+1}
        -
        \frac1{a+1}
    \right)>0.
}
\]

This is the formal asymptotic separation between aligned/band-limited Adam and flat/Haar coordinatewise adaptivity.

## 5. Damping and AdamW weight decay

With damping, the visible-spectrum learned-mode count is

\[
K_{\rho,\theta}(n)
\asymp
\begin{cases}
    n^{1/[a(1-\theta/2)]}, & n\lesssim n_\rho(\theta),\\[3pt]
    \rho^{-1/(2a)}n^{1/a}, & n\gtrsim n_\rho(\theta),
\end{cases}
\]

where the precise knee depends on \(\theta\) and reduces to \(\rho^{-1/2}\) for \(\theta=1\).  The compute exponent above applies in the pre-damping active phase.  Past the damping knee, the tail exponent reverts to \(a\), so the compute exponent returns to the scalar-preconditioned value up to constants.

With AdamW weight decay, the effective horizon is capped:

\[
    n\mapsto T_{\rm eff}=\min\{n,\delta^{-1}\}.
\]

Therefore the compute-optimal visible-spectrum exponent persists only while

\[
    n_\star(C)\lesssim \delta^{-1}.
\]

Since

\[
    n_\star(C)\asymp C^{\alpha/(\alpha+1)},
\]

one needs

\[
\boxed{
    \delta\lesssim C^{-\alpha/(\alpha+1)}
}
\]

to avoid weight-decay saturation at growing compute.

## 6. Experiment predicted by this theorem

The deterministic experiment

```bash
python experiments/compute_optimal_visible_spectrum.py --quick
```

and the larger run

```bash
python experiments/compute_optimal_visible_spectrum.py \
  --dimension 400000 \
  --a 1.5 --b 1.4 \
  --theta-values 0,0.25,0.5,0.75,1 \
  --compute-values 1e4,3e4,1e5,3e5,1e6,3e6,1e7 \
  --m-values 32,64,128,256,512,1024,2048,4096,8192,16384,32768
```

should observe compute-risk slopes close to

\[
    -\beta(\theta)
    =
    -\frac{b-1}{a(1-\theta/2)+1}.
\]

The same experiment should show model/data allocation exponents

\[
    M_\star(C)\asymp C^{1/(\alpha+1)},
    \qquad
    N_\star(C)\asymp C^{\alpha/(\alpha+1)}.
\]

## 7. Consequence for the paper

The paper can now make a compute-optimal claim:

\[
\boxed{
\text{Visible spectral adaptivity improves the compute exponent exactly by replacing }a
\text{ with }a(1-\theta/2).
}
\]

This is stronger than saying Adam has better constants.  It predicts a different optimal allocation between model size and data/steps whenever the optimizer coordinates expose nonzero spectral information \(\theta>0\).
