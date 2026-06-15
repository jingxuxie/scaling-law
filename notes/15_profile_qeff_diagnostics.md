# 15. Profile-exponent diagnostics estimate the effective Adam/RMSProp exponent

The previous notes identify three regimes: aligned coordinates, band-limited mixing, and globally mixed coordinates. This note gives the diagnostic theorem needed for experiments: the empirical slope of coordinate second moments predicts the effective Adam/RMSProp exponent.

The central message is

\[
\boxed{
    d_j\asymp \lambda_j^\theta
    \quad\Longrightarrow\quad
    q_{\rm eff}=\theta/2.
}
\]

Here \(d_j\) is the coordinate variance, or equivalently the population scale of the coordinatewise gradient second moment.

## 1. Band-profile model

Let \(H\) have eigenvalues

\[
    \lambda_i\asymp i^{-a},
    \qquad a>1.
\]

Partition coordinates into spectral bands \(B_\ell\). Inside band \(B_\ell\), assume

\[
    \lambda_i\asymp \Lambda_\ell,
    \qquad i\in B_\ell.
\]

Let the optimizer-coordinate covariance be \(\Sigma\), and suppose the coordinate variances in band \(B_\ell\) obey

\[
\boxed{
    \Sigma_{jj}\asymp \Delta_\ell,
    \qquad j\in B_\ell.
}
\]

Assume the variance profile has exponent \(\theta\):

\[
\boxed{
    \Delta_\ell\asymp \Lambda_\ell^\theta,
    \qquad 0\le \theta\le1.
}
\]

The cases are \(\theta=1\) for aligned or band-limited spectral coordinates, and \(\theta=0\) for flat/global mixing coordinates.

## 2. Effective spectrum theorem

RMSProp/Adam estimates coordinate second moments proportional to \(\Sigma_{jj}\), so after scalar normalization its diagonal preconditioner has entries

\[
    D_{jj}=(\Sigma_{jj}+\rho)^{-1/2}.
\]

The transformed covariance is

\[
    \widetilde\Sigma=D^{1/2}\Sigma D^{1/2}.
\]

Assume the coordinate mixing is band-limited, so within \(B_\ell\) the eigenvalues of \(\Sigma_{B_\ell}\) are the original \(\lambda_i\)'s in that band up to constants. Since \(D_{jj}\asymp(\Delta_\ell+\rho)^{-1/2}\) inside the band, the min-max principle gives

\[
    \widetilde\mu_i
    \asymp
    \lambda_i(\Delta_\ell+\rho)^{-1/2},
    \qquad i\in B_\ell.
\]

Using \(\Delta_\ell\asymp\Lambda_\ell^\theta\) and \(\lambda_i\asymp\Lambda_\ell\),

\[
\boxed{
    \widetilde\mu_i
    \asymp
    \lambda_i(\lambda_i^\theta+\rho)^{-1/2}.
}
\]

In the active, pre-damping regime \(\lambda_i^\theta\gg\rho\), this becomes

\[
\boxed{
    \widetilde\mu_i
    \asymp
    \lambda_i^{1-\theta/2}.
}
\]

Therefore, if \(\lambda_i\asymp i^{-a}\), the effective spectral exponent is

\[
\boxed{
    \alpha_{\rm eff}=a(1-\theta/2).
}
\]

Equivalently,

\[
\boxed{
    q_{\rm eff}=1-\frac{\alpha_{\rm eff}}a=\theta/2.
}
\]

## 3. Learned-mode count

The damping knee occurs when

\[
    \lambda_i^\theta\asymp \rho.
\]

Since \(\lambda_i\asymp i^{-a}\), the spectral index knee is

\[
    i_\rho\asymp \rho^{-1/(a\theta)}
\]

for \(\theta>0\). The corresponding effective eigenvalue scale is

\[
    \mu_{i_\rho}\asymp \rho^{1/\theta-1/2},
\]

so the effective horizon knee is

\[
    n_\rho\asymp \rho^{-(1/\theta-1/2)}.
\]

For \(\theta=0\), there is no spectral damping knee from the profile because \(\lambda_i^\theta=1\) is constant.

For \(0<\theta\le1\), the learned-mode count is

\[
\boxed{
K_{\rho,\theta}(n)
\asymp
\begin{cases}
    n^{1/[a(1-\theta/2)]},
    & n\lesssim \rho^{-(1/\theta-1/2)},\\[4pt]
    \rho^{-1/(2a)}n^{1/a},
    & n\gtrsim \rho^{-(1/\theta-1/2)}.
\end{cases}
}
\]

This interpolates between SGD and aligned Adam/RMSProp:

\[
\theta=0:\quad K(n)\asymp n^{1/a},
\]

while

\[
\theta=1:\quad K(n)\asymp n^{2/a}
\]

before the damping knee.

## 4. Bias and variance filters

Under the source condition

\[
    s_i=\lambda_i(w_i^\star)^2\asymp i^{-b},
\]

and in the rough/intermediate source range

\[
    1<b<a(1-\theta/2)+1,
\]

the sharp spectral-sum results from `09_matching_filter_lower_bounds.md` give

\[
\boxed{
    R_M-\sigma^2
    \asymp
    M^{1-b}
    +K_{\rho,\theta}(n)^{1-b}
    +\frac{K_{\rho,\theta}(n)}{N_{\rm eff}}.
}
\]

When \(b\ge a(1-\theta/2)+1\), the smooth-source saturation theorem applies and the bias exponent saturates at \(n^{-1}\), up to a logarithm at the boundary.

## 5. Diagnostic consistency theorem

Suppose an experiment estimates coordinate second moments \(\widehat v_j\) over a fitting window \(J\), and suppose for all \(j\in J\),

\[
    C^{-1}d_j\le \widehat v_j\le C d_j.
\]

Assume also that over the same window

\[
    d_j\asymp \lambda_j^\theta
\]

and the log-eigenvalue range is

\[
    L_J:=\max_{j\in J}\log\lambda_j-
    \min_{j\in J}\log\lambda_j.
\]

Let \(\widehat\theta\) be the least-squares slope of \(\log \widehat v_j\) versus \(\log \lambda_j\) on \(J\). Then

\[
\boxed{
    |\widehat\theta-\theta|\lesssim \frac{\log C}{L_J}.
}
\]

Consequently the empirical preconditioner exponent

\[
    \widehat q_{\rm eff}:=\widehat\theta/2
\]

satisfies

\[
\boxed{
    |\widehat q_{\rm eff}-q_{\rm eff}|\lesssim \frac{\log C}{2L_J}.
}
\]

### Proof

Write

\[
    \log \widehat v_j=\log d_j+\varepsilon_j,
    \qquad |\varepsilon_j|\le \log C.
\]

Since \(d_j\asymp\lambda_j^\theta\), also

\[
    \log d_j=c+\theta\log\lambda_j+\eta_j
\]

with bounded \(\eta_j\). Least squares is Lipschitz in the response with operator norm controlled by the inverse spread of the regressor. The perturbation of the slope is bounded by a constant times

\[
    \frac{\max_j|\varepsilon_j|+\max_j|\eta_j|}{L_J}.
\]

Absorbing the fixed profile-comparability constants gives the stated bound. Dividing by two gives the \(q_{\rm eff}\) estimate.

## 6. Experimental implication

The experiments should report three quantities:

1. the coordinate-variance slope \(\widehat\theta\);
2. the observed effective spectrum slope \(\widehat\alpha_{\rm eff}\);
3. the consistency check

\[
\boxed{
    \widehat\alpha_{\rm eff}\approx a(1-\widehat\theta/2),
    \qquad
    \widehat q_{\rm eff}\approx \widehat\theta/2.
}
\]

This turns the alignment story into a measurable prediction. A feature map that gives \(\widehat\theta\approx1\) should display Adam/RMSProp spectral gains. A feature map that gives \(\widehat\theta\approx0\) should behave more like scalar preconditioning. Intermediate profiles should interpolate continuously.
