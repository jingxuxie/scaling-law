# 19. Visibility phase transition in compute-optimal scaling

The visible-spectrum theorem gives an effective spectral exponent

\[
    \alpha(\theta)=a(1-\theta/2),
\]

where \(\theta\in[0,1]\) measures how much spectral structure is visible to coordinatewise second moments.  This note makes the compute-optimal phase transition explicit.

## 1. Compute-risk exponent

The simplified sharp risk law is

\[
    R(M,N,K)-\sigma^2
    \asymp
    M^{1-b}+K^{1-b}+\frac{K}{N},
\]

with compute constraint

\[
    C\asymp MN,
\]

and algorithmic learned-mode constraint

\[
    K\le N^{1/\alpha(\theta)}.
\]

Let

\[
    m(\theta)=\max\{\alpha(\theta),b\}.
\]

Then the compute-optimal risk exponent is

\[
\boxed{
    \beta(\theta)=\frac{b-1}{m(\theta)+1}
    =
    \frac{b-1}{\max\{a(1-\theta/2),b\}+1}.
}
\]

Thus

\[
    R_\star(C)-\sigma^2\asymp C^{-\beta(\theta)}.
\]

## 2. Critical visibility threshold

The phase boundary is

\[
    \alpha(\theta)=b.
\]

Solving gives

\[
\boxed{
    \theta_c=2\left(1-\frac{b}{a}\right).
}
\]

There are three regimes.

### Always variance-limited

If \(\theta_c\le0\), equivalently \(b\ge a\), then the problem is already variance-limited at \(\theta=0\).  More visible spectral information improves constants and learning-rate schedules, but it does not improve the asymptotic compute exponent:

\[
    \beta(\theta)=\frac{b-1}{b+1}
    \qquad\text{for all }\theta\in[0,1].
\]

### Transition inside the visible range

If \(0<\theta_c<1\), then

\[
\boxed{
\beta(\theta)=
\begin{cases}
\dfrac{b-1}{a(1-\theta/2)+1}, & 0\le \theta\le\theta_c,\\[10pt]
\dfrac{b-1}{b+1}, & \theta\ge\theta_c.
\end{cases}
}
\]

Visibility improves the compute exponent until \(\theta_c\), after which the risk is variance-limited and the exponent saturates.

### Always time-limited

If \(\theta_c\ge1\), equivalently \(b\le a/2\), then the whole interval \(\theta\in[0,1]\) is time-limited.  Visibility improves the compute exponent throughout:

\[
    \beta(\theta)=\frac{b-1}{a(1-\theta/2)+1}.
\]

## 3. Allocation exponents

The optimal allocation is

\[
\boxed{
    M_\star(C)\asymp C^{1/(m(\theta)+1)},
    \qquad
    N_\star(C)\asymp C^{m(\theta)/(m(\theta)+1)}.
}
\]

The optimal learned-mode count satisfies

\[
\boxed{
    K_\star(C)\asymp C^{1/(m(\theta)+1)}.
}
\]

The optimal effective horizon is

\[
\boxed{
    n_\star(C)\asymp C^{\alpha(\theta)/(m(\theta)+1)}.
}
\]

and the optimal effective learning-rate scale is

\[
\boxed{
    \gamma_\star(C)=\frac{n_\star(C)}{N_\star(C)}
    \asymp
    C^{(\alpha(\theta)-m(\theta))/(m(\theta)+1)}.
}
\]

So \(\gamma_\star\asymp 1\) in the hard-source/time-limited phase, while \(\gamma_\star\) decays with compute in the easy-source/variance-limited phase.

## 4. Interpretation of the compute-optimal phase-grid result

For the experiment with \(a=1.5,b=1.4\),

\[
    \theta_c=2(1-1.4/1.5)\approx0.133.
\]

Therefore only \(\theta=0\) is time-limited; all tested \(\theta\ge0.25\) lie in the saturated variance-limited phase.  The risk exponent should be approximately

\[
    \frac{b-1}{b+1}=\frac{0.4}{2.4}=0.1667
\]

for \(\theta\ge0.25\).  This is exactly what the compute-optimal experiment observes.

For \(a=3.0,b=1.4\),

\[
    \theta_c=2(1-1.4/3)>1,
\]

so the whole \(\theta\in[0,1]\) interval is time-limited.  The compute exponent should improve monotonically as visibility increases.  The hard-full-range phase-grid run checks precisely this regime.
