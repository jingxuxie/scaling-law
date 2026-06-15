# 17. Compute-optimal visible-spectrum phase diagram

The previous notes give the risk law for a visible spectral profile.  This note
turns that risk law into a compute-optimal scaling law.

The key parameter is

\[
    \alpha(\theta)=a(1-\theta/2),
\]

where \(a\) is the original covariance exponent and \(\theta\in[0,1]\) measures
how much spectral information is visible to coordinatewise second moments.  The
corresponding optimizer exponent is

\[
    q_{\rm eff}=\theta/2.
\]

## 1. Visible-spectrum risk law

In the active pre-damping regime, the learned-mode count is

\[
    K_{\theta}(N)\asymp N^{1/\alpha(\theta)}.
\]

Ignoring logarithms and constants, the clean source-range risk law is

\[
\boxed{
    R(M,N)-\sigma^2
    \asymp
    M^{-(b-1)}
    +
    N^{-(b-1)/\alpha}
    +
    \frac{N^{1/\alpha}}{N},
}
\]

where

\[
    \alpha=\alpha(\theta)=a(1-\theta/2).
\]

In the hard-source regime

\[
    1<b\le \alpha,
\]

or whenever the variance term is lower order, the leading tradeoff is

\[
    R(M,N)-\sigma^2
    \asymp
    M^{-(b-1)}+N^{-(b-1)/\alpha}.
\]

## 2. Compute-optimal allocation

Let one-pass compute be

\[
    C\asymp MN.
\]

Then

\[
    N=C/M.
\]

The leading risk proxy becomes

\[
    \mathcal R(M;C)
    =
    M^{-(b-1)}+\left(\frac{C}{M}\right)^{-(b-1)/\alpha}.
\]

Balancing the two terms gives

\[
    M^{-(b-1)}
    \asymp
    C^{-(b-1)/\alpha}M^{(b-1)/\alpha}.
\]

Equivalently,

\[
    M^{(b-1)(1+1/\alpha)}
    \asymp
    C^{(b-1)/\alpha}.
\]

Thus

\[
\boxed{
    M_\star(C)
    \asymp
    C^{1/(\alpha+1)}.
}
\]

and

\[
\boxed{
    N_\star(C)
    \asymp
    C^{\alpha/(\alpha+1)}.
}
\]

The compute-optimal risk exponent is

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-(b-1)/(\alpha+1)}.
}
\]

Substituting \(\alpha=a(1-\theta/2)\), we obtain

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-(b-1)/[a(1-\theta/2)+1]}.
}
\]

This is the visible-spectrum compute law.

## 3. Consequences for optimizers

### SGD or scalar-comparable adaptivity

For SGD, flat coordinates, Haar/global mixing, and isotropic Gaussian sketches,

\[
    \theta=0,
    \qquad
    \alpha=a.
\]

Therefore

\[
\boxed{
    R_{\rm SGD}(C)-\sigma^2
    \asymp
    C^{-(b-1)/(a+1)}.
}
\]

### Adam/RMSProp in aligned or band-limited coordinates

For aligned or band-limited coordinates,

\[
    \theta=1,
    \qquad
    \alpha=a/2.
\]

Therefore

\[
\boxed{
    R_{\rm Adam}(C)-\sigma^2
    \asymp
    C^{-(b-1)/(a/2+1)}.
}
\]

This is a strictly better compute exponent than SGD whenever the theorem's clean
source and active-damping conditions hold.

### Partially visible coordinates

For intermediate visibility,

\[
    0<\theta<1,
\]

we get a continuum of compute exponents:

\[
\boxed{
    \zeta(\theta)
    =
    \frac{b-1}{a(1-\theta/2)+1}.
}
\]

The improvement over SGD is monotone in \(\theta\).

## 4. Damping and AdamW weight decay

The active visible-spectrum law only holds until the damping knee.  For
\(\theta>0\), the damping knee occurs at the horizon

\[
    N_\rho
    \asymp
    \rho^{-(1/\theta-1/2)}.
\]

The active compute law is valid when

\[
    N_\star(C)\lesssim N_\rho.
\]

Equivalently,

\[
    C^{\alpha/(\alpha+1)}
    \lesssim
    \rho^{-(1/\theta-1/2)}.
\]

For AdamW, decoupled weight decay creates an additional horizon ceiling

\[
    N_\delta\asymp \delta^{-1}.
\]

Thus AdamW preserves the active visible-spectrum compute exponent only when

\[
\boxed{
    \delta
    \lesssim
    N_\star(C)^{-1}
    \asymp
    C^{-\alpha/(\alpha+1)}.
}
\]

If \(\delta\) is fixed, the learned-mode count saturates once
\(N_\star(C)\gtrsim\delta^{-1}\), and the compute-risk curve eventually bends
away from the no-decay exponent.

## 5. Easy-source / variance-limited phase

When

\[
    \alpha<b<\alpha+1,
\]

the constant-stepsize variance term can no longer be ignored.  Optimizing the
effective horizon, or equivalently shrinking the learning rate, gives the usual
saturation exponent

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-(b-1)/(b+1)}.
}
\]

This phase is independent of \(\alpha\) at exponent level.  Therefore increasing
visible spectral preconditioning helps until \(\alpha\) reaches the source
smoothness scale; after that, additional flattening mostly changes constants and
optimal learning-rate schedules.

## 6. What this theorem predicts experimentally

The deterministic script

```bash
python experiments/compute_optimal_visible_spectrum.py --quick
```

checks the following predictions:

\[
    M_\star(C)\asymp C^{1/(\alpha+1)},
    \qquad
    N_\star(C)\asymp C^{\alpha/(\alpha+1)},
\]

and

\[
    R_\star(C)-\sigma^2\asymp C^{-(b-1)/(\alpha+1)}.
\]

The main run is

```bash
python experiments/compute_optimal_visible_spectrum.py \
  --a 1.5 --b 1.4 --theta-values 0,0.25,0.5,0.75,1 \
  --rho 1e-16 --dimension 500000 \
  --c-min 1e5 --c-max 1e9 --num-c 18 \
  --outdir experiments/results/compute_optimal_visible
```

This experiment should become one of the paper's main scaling-law figures,
because it translates the visible-spectrum mechanism into compute-optimal
scaling exponents.
