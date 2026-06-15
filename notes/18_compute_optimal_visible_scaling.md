# 18. Compute-optimal visible-spectrum scaling law

The previous notes identify the visible-spectrum exponent

\[
    q_{\rm eff}=\theta/2,
\]

and the effective covariance exponent

\[
    \alpha=a(1-\theta/2).
\]

This note derives the compute-optimal allocation over model size \(M\), data/time \(N\), and effective learned-mode count \(K\).  It is the scaling-law layer needed to turn the spectral mechanism into a Chinchilla-style allocation theorem.

## 1. Simplified risk proxy

In the clean pre-damping regime, the visible-spectrum learned-mode count is

\[
    K\asymp n^{1/\alpha},
    \qquad
    \alpha=a(1-\theta/2).
\]

The sharp diagonal/band-limited risk law has the form

\[
\boxed{
    R(M,N,K)-\sigma^2
    \asymp
    M^{1-b}+K^{1-b}+\frac{K}{N}.
}
\]

Here:

- \(M^{1-b}\) is approximation error;
- \(K^{1-b}\) is optimization/source bias after learning \(K\) modes;
- \(K/N\) is the variance/effective-dimension term.

The algorithmic time constraint is

\[
    K\le N^{1/\alpha},
\]

because with a constant stable learning rate the largest learnable count after \(N\) samples is \(N^{1/\alpha}\).  Allowing smaller learning rates lets us choose any smaller \(K\).

The compute budget is

\[
    C\asymp MN.
\]

## 2. Optimize over the learned-mode count

For fixed \(N\), minimize

\[
    K^{1-b}+K/N
\]

subject to

\[
    K\le N^{1/\alpha}.
\]

Without the constraint, the optimum solves

\[
    K^{-b}\asymp N^{-1},
\]

so

\[
    K\asymp N^{1/b}.
\]

Thus there are two regimes.

### Hard-source / time-limited phase: \(b\le \alpha\)

Then

\[
    N^{1/b}\ge N^{1/\alpha},
\]

so the algorithmic constraint is active:

\[
\boxed{
    K_\star(N)\asymp N^{1/\alpha}.
}
\]

The variance term is lower order and the \(N\)-dependent error is

\[
    N^{-(b-1)/\alpha}.
\]

### Easy-source / variance-limited phase: \(b>\alpha\)

Then

\[
    N^{1/b}<N^{1/\alpha},
\]

so the variance-bias tradeoff determines

\[
\boxed{
    K_\star(N)\asymp N^{1/b}.
}
\]

Equivalently, the effective optimization horizon is reduced to

\[
    n_\star=K_\star^\alpha\asymp N^{\alpha/b}<N,
\]

so the optimal learning-rate scale decays as

\[
    \gamma_\star\asymp N^{\alpha/b-1}.
\]

The \(N\)-dependent error is

\[
    N^{-(b-1)/b}.
\]

## 3. Compute-optimal allocation theorem

Let

\[
    m:=\max\{\alpha,b\}.
\]

Then the compute-optimal allocation satisfies

\[
\boxed{
    M_\star(C)\asymp C^{1/(m+1)},
    \qquad
    N_\star(C)\asymp C^{m/(m+1)}.
}
\]

The learned-mode count obeys

\[
\boxed{
    K_\star(C)\asymp C^{1/(m+1)}.
}
\]

The optimized risk is

\[
\boxed{
    R_\star(C)-\sigma^2
    \asymp
    C^{-(b-1)/(m+1)}.
}
\]

Equivalently, the compute-risk exponent is

\[
\boxed{
    \beta(\theta)
    =
    \frac{b-1}{\max\{a(1-\theta/2),b\}+1}.
}
\]

This unifies the two phases:

\[
\boxed{
\begin{array}{lll}
    b\le \alpha &:& \beta=(b-1)/(\alpha+1),\\[3pt]
    b>\alpha &:& \beta=(b-1)/(b+1).
\end{array}
}
\]

## 4. Monotonicity in visible spectral information

Since

\[
    \alpha(\theta)=a(1-\theta/2)
\]

is decreasing in \(\theta\), the compute-risk exponent is nondecreasing in \(\theta\).  It strictly improves with \(\theta\) in the hard-source phase \(b<\alpha(\theta)\), and saturates once \(\alpha(\theta)\le b\).

Thus increasing visible spectral information can improve asymptotic compute scaling, but only until the variance-limited/source-easy phase is reached.

## 5. Optimal effective learning-rate horizon

The effective optimization horizon is

\[
    n_\star=K_\star^\alpha.
\]

Therefore

\[
\boxed{
    n_\star(C)
    \asymp
    C^{\alpha/(m+1)}.
}
\]

Since \(N_\star(C)\asymp C^{m/(m+1)}\), the effective learning-rate scale is

\[
\boxed{
    \gamma_\star(C)
    :=n_\star/N_\star
    \asymp
    C^{(\alpha-m)/(m+1)}.
}
\]

In the hard-source phase, \(m=\alpha\), so \(\gamma_\star\asymp1\).  In the easy-source phase, \(m=b\), so

\[
    \gamma_\star\asymp C^{(\alpha-b)/(b+1)},
\]

which decreases with compute.

This explains why constant learning-rate experiments can show transiently very steep risk slopes in easy-source regimes: they may be over-learning noisy directions relative to the asymptotic variance-optimal schedule.

## 6. AdamW weight decay

AdamW imposes an additional cap on the effective horizon:

\[
    n\le \delta^{-1}.
\]

Equivalently,

\[
    K\le \delta^{-1/\alpha}

\]

in the pre-damping active phase.  To preserve the no-weight-decay compute-optimal law at compute \(C\), we need

\[
    \delta^{-1/\alpha}\gtrsim K_\star(C).
\]

Thus

\[
\boxed{
    \delta\lesssim C^{-\alpha/(m+1)}.
}
\]

If \(\delta>0\) is fixed while \(C\to\infty\), the learned-mode count saturates and the optimization/source bias has a nonzero floor.

## 7. Experiment

The script

```bash
python experiments/compute_optimal_scaling.py --quick
```

numerically optimizes the simplified risk proxy over \((M,N,K)\) and verifies the exponents above.  The main run is

```bash
python experiments/compute_optimal_scaling.py \
  --a 1.5 \
  --b 1.4 \
  --theta-values 0,0.25,0.5,0.75,1 \
  --c-min 1e4 \
  --c-max 1e10 \
  --num-c 40 \
  --outdir experiments/results/compute_optimal_visible
```

This is the remaining deterministic scaling-law experiment needed before writing the main paper draft.
