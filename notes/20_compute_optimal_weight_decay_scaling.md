# 20. Compute-Optimal Scaling With Weight-Decay Schedules

The fixed-weight-decay rows in the compute-optimal phase-grid experiment should
not be compared to the no-weight-decay exponents. A fixed AdamW decay imposes a
fixed cap on the learned-mode count and therefore creates an asymptotic risk
floor. This note derives the correct compute-optimal law when the effective
weight-decay threshold is allowed to scale with compute.

## Setup

Use the visible-spectrum exponent

```math
\alpha=a(1-\theta/2)
```

and the simplified sharp risk proxy

```math
R(M,N,K)-\sigma^2
\asymp
M^{1-b}+K^{1-b}+\frac{K}{N}.
```

The compute budget is

```math
C\asymp MN.
```

Without weight decay, the algorithmic horizon constraint is

```math
K\le N^{1/\alpha}.
```

AdamW-style decoupled weight decay with effective threshold `delta` adds the cap

```math
K\le K_\delta:=\delta^{-1/\alpha}.
```

Thus

```math
K\le \min\{N^{1/\alpha},\delta^{-1/\alpha}\}.
```

## Fixed Weight Decay Produces A Floor

If `delta > 0` is fixed while `C -> infinity`, then

```math
K_\delta=\delta^{-1/\alpha}
```

is fixed. Even if `M,N -> infinity`, the source/optimization term cannot go
below

```math
K_\delta^{1-b}=\delta^{(b-1)/\alpha}.
```

Therefore

```math
\boxed{
\liminf_{C\to\infty} (R_\star(C)-\sigma^2)
\gtrsim
\delta^{(b-1)/\alpha}.
}
```

So fixed weight decay does not have an asymptotic power-law decay to zero. Over
finite compute ranges one may observe an apparent negative slope, but this is a
pre-asymptotic approach to the weight-decay floor.

This explains the phase-grid rows with fixed `delta=10^{-3}` and `delta=10^{-4}`:
their observed slopes are not supposed to match the no-weight-decay prediction.

## Compute-Dependent Weight Decay

Now suppose the effective weight-decay threshold scales as

```math
\delta(C)=C^{-s},
\qquad s\ge0.
```

Then

```math
K_\delta(C)=C^{s/\alpha}.
```

Let

```math
m=\max\{\alpha,b\}.
```

The no-weight-decay optimum has

```math
K_0(C)\asymp C^{1/(m+1)}
```

and risk exponent

```math
\beta_0=\frac{b-1}{m+1}.
```

The weight-decay cap is inactive exactly when

```math
K_\delta(C)\gtrsim K_0(C),
```

or equivalently

```math
\boxed{
s\ge \frac{\alpha}{m+1}.
}
```

This recovers the condition from the previous compute-optimal theorem.

## Theorem 1: Capped Compute-Optimal Exponent

Let `delta(C)=C^{-s}`. In the clean source range and pre-damping regime, the
compute-optimal risk exponent is

```math
\boxed{
\beta_{\rm wd}(\theta,s)
=
\min\left\{
\frac{(b-1)s}{\alpha},
\frac{b-1}{m+1}
\right\},
\qquad
\alpha=a(1-\theta/2),
\quad m=\max\{\alpha,b\}.
}
```

Thus

```math
R_\star(C)-\sigma^2
\asymp
C^{-\beta_{\rm wd}(\theta,s)}
```

up to constants, until the fixed-`delta` floor is reached when `s=0`.

### Proof

If

```math
s\ge \alpha/(m+1),
```

then `K_delta(C) >= K_0(C)` up to constants, so the no-weight-decay optimizer is
feasible. The exponent is `beta_0=(b-1)/(m+1)`.

Now assume

```math
s<\alpha/(m+1).
```

Then the cap is active and

```math
K_\star(C)\asymp K_\delta(C)=C^{s/\alpha}.
```

The source/optimization term is

```math
K_\star^{1-b}
\asymp
C^{-(b-1)s/\alpha}.
```

For this fixed learned count, minimize the remaining approximation and variance
terms:

```math
M^{1-b}+\frac{K_\star M}{C}.
```

The unconstrained balance gives

```math
M\asymp (C/K_\star)^{1/b}
= C^{(1-s/\alpha)/b}.
```

The constraint `M >= K_star` holds because

```math
s/\alpha < 1/(m+1)\le 1/(b+1).
```

At this choice of `M`, the approximation and variance terms scale as

```math
C^{-(b-1)(1-s/\alpha)/b}.
```

Since `s/alpha < 1/(b+1)`,

```math
\frac{(b-1)s}{\alpha}
<
\frac{(b-1)(1-s/\alpha)}{b},
```

so the capped source term dominates. Therefore

```math
R_\star(C)-\sigma^2
\asymp
C^{-(b-1)s/\alpha}.
```

Combining the two cases proves the theorem.

## Allocation In The Cap-Active Regime

When `s < alpha/(m+1)`, the compute-optimal scalings are

```math
\boxed{
K_\star(C)\asymp C^{s/\alpha},
}
```

```math
\boxed{
M_\star(C)\asymp C^{(1-s/\alpha)/b},
\qquad
N_\star(C)\asymp C^{1-(1-s/\alpha)/b}.
}
```

The exponent is controlled by the weight-decay cap, not by the no-decay
visible-spectrum law.

## Experimental Implication

The script

```bash
python experiments/compute_optimal_weight_decay_schedule.py --quick
```

checks the formula above for several decay schedules `delta(C)=C^{-s}`. The main
run is

```bash
python experiments/compute_optimal_weight_decay_schedule.py \
  --a 3.0 \
  --b 1.4 \
  --theta-values 0,0.5,1 \
  --s-values 0,0.05,0.1,0.2,0.4,0.8 \
  --c-min 1e4 \
  --c-max 1e10 \
  --num-c 40 \
  --outdir experiments/results/compute_optimal_weight_decay_schedule
```

The fixed-decay case `s=0` should show a near-zero asymptotic risk exponent. As
`s` increases, the exponent should grow linearly as `(b-1)s/alpha` until it
reaches the no-decay exponent `(b-1)/(m+1)`, after which it saturates.
