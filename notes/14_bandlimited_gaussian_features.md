# 14. Band-limited Gaussian features preserve Adam/RMSProp spectral gains

The global-sketch theorem shows that a fully mixed Gaussian sketch can flatten coordinate variances and make diagonal Adam/RMSProp approximately scalar.  This note proves the complementary random-feature result: if random features are Gaussian **within comparable-eigenvalue spectral bands**, then coordinatewise Adam/RMSProp still sees band-scale second moments and preserves the aligned \(q_{\rm eff}=1/2\) scaling law.

## 1. Band-limited Gaussian feature model

Let

\[
    H=\operatorname{diag}(\lambda_1,\ldots,\lambda_D),
    \qquad
    \lambda_1\ge\lambda_2\ge\cdots>0.
\]

Partition eigencoordinates into contiguous bands

\[
    B_1,\ldots,B_L,
    \qquad |B_\ell|=d_\ell.
\]

Assume eigenvalues in band \(B_\ell\) are comparable: for a band scale \(\Lambda_\ell\) and constant \(\kappa\ge1\),

\[
\boxed{
    \kappa^{-1}\Lambda_\ell
    \le
    \lambda_i
    \le
    \kappa\Lambda_\ell,
    \qquad i\in B_\ell.
}
\]

For each band, draw a Gaussian feature block

\[
    S_\ell\in\mathbb R^{m_\ell\times d_\ell},
    \qquad
    (S_\ell)_{jk}\sim \mathcal N(0,1/d_\ell),
\]

independently across bands.  The full feature map is block diagonal:

\[
    S=\operatorname{blockdiag}(S_1,\ldots,S_L).
\]

The feature covariance is

\[
    \Sigma=S H S^\top
    =
    \operatorname{blockdiag}(S_1H_1S_1^\top,\ldots,S_LH_LS_L^\top),
\]

where \(H_\ell=H|_{B_\ell}\).  Assume the row aspect ratios satisfy

\[
\boxed{
    0<\gamma_{\min}\le \gamma_\ell:=m_\ell/d_\ell\le \gamma_{\max}<1.
}
\]

The condition \(\gamma_{\max}<1\) is a clean sufficient condition ensuring that the nonzero singular values of each Gaussian block are bounded above and below by constants with high probability.

## 2. Coordinate variances concentrate at the band scale

For feature coordinate \(j\) in band \(\ell\), let \(s_{\ell,j}\) be the corresponding row of \(S_\ell\).  The coordinate variance seen by diagonal RMSProp/Adam is

\[
    d_{\ell,j}=s_{\ell,j}^\top H_\ell s_{\ell,j}.
\]

Its mean is

\[
    \tau_\ell=\mathbb E d_{\ell,j}=\frac{\operatorname{tr}(H_\ell)}{d_\ell}.
\]

By band comparability,

\[
\boxed{
    \kappa^{-1}\Lambda_\ell
    \le
    \tau_\ell
    \le
    \kappa\Lambda_\ell.
}
\]

Since

\[
    r_{\rm eff}(H_\ell)
    =
    \frac{\operatorname{tr}(H_\ell)^2}{\operatorname{tr}(H_\ell^2)}
    \gtrsim_\kappa d_\ell,
\]

Hanson-Wright / weighted chi-square concentration gives, for \(0<\varepsilon<1\),

\[
    \Pr\left(|d_{\ell,j}-\tau_\ell|>\varepsilon\tau_\ell\right)
    \le
    2\exp(-c_\kappa\varepsilon^2d_\ell).
\]

Union bounding over all feature coordinates gives the event

\[
\boxed{
    (1-\varepsilon)\tau_\ell
    \le
    d_{\ell,j}
    \le
    (1+\varepsilon)\tau_\ell
    \qquad \forall \ell,j
}
\]

with probability at least

\[
    1-2\sum_{\ell=1}^L m_\ell\exp(-c_\kappa\varepsilon^2d_\ell).
\]

Thus the diagonal second moment retains the spectral **band scale**.

## 3. The sketched covariance has the same band spectrum

Inside band \(\ell\),

\[
    \kappa^{-1}\Lambda_\ell I
    \preceq H_\ell
    \preceq \kappa\Lambda_\ell I.
\]

Therefore

\[
    \kappa^{-1}\Lambda_\ell S_\ell S_\ell^\top
    \preceq
    S_\ell H_\ell S_\ell^\top
    \preceq
    \kappa\Lambda_\ell S_\ell S_\ell^\top.
\]

Gaussian singular-value concentration implies that, with probability at least \(1-2\exp(-c_\gamma d_\ell)\),

\[
    c_\gamma I_{m_\ell}
    \preceq
    S_\ell S_\ell^\top
    \preceq
    C_\gamma I_{m_\ell}.
\]

Consequently,

\[
\boxed{
    c_{\kappa,\gamma}\Lambda_\ell I_{m_\ell}
    \preceq
    \Sigma_\ell:=S_\ell H_\ell S_\ell^\top
    \preceq
    C_{\kappa,\gamma}\Lambda_\ell I_{m_\ell}.
}
\]

So the \(m_\ell\) nonzero feature eigenvalues contributed by band \(\ell\) are all comparable to \(\Lambda_\ell\).  If \(m_\ell\asymp d_\ell\), the feature covariance preserves the original spectral counting measure up to constants.

## 4. RMSProp/Adam preconditioning inside each band

By the coordinate-second-moment identity, RMSProp/Adam tracks

\[
    v_{\ell,j}\asymp d_{\rm res}\,d_{\ell,j}.
\]

Ignoring the scalar residual factor, the diagonal preconditioner is

\[
    D_\rho=\operatorname{diag}\big((d_{\ell,j}+\rho)^{-1/2}\big).
\]

On the coordinate-variance event, within band \(\ell\),

\[
    D_{\rho,\ell}
    \asymp_{\kappa,\varepsilon}
    (\Lambda_\ell+\rho)^{-1/2}I_{m_\ell}.
\]

The transformed feature covariance is

\[
    \widetilde\Sigma=D_\rho^{1/2}\Sigma D_\rho^{1/2}.
\]

Therefore, in each band,

\[
\boxed{
    c_{\kappa,\gamma,\varepsilon}
    \Lambda_\ell(\Lambda_\ell+\rho)^{-1/2}I_{m_\ell}
    \preceq
    \widetilde\Sigma_\ell
    \preceq
    C_{\kappa,\gamma,\varepsilon}
    \Lambda_\ell(\Lambda_\ell+\rho)^{-1/2}I_{m_\ell}.
}
\]

Thus every effective eigenvalue coming from band \(\ell\) is comparable to

\[
\boxed{
    \Lambda_\ell(\Lambda_\ell+\rho)^{-1/2}.
}
\]

Since \(\lambda_i\asymp\Lambda_\ell\) for \(i\in B_\ell\), this is the same spectrum as the oracle aligned Adam/RMSProp preconditioner:

\[
\boxed{
    \widetilde\mu_i\asymp \lambda_i(\lambda_i+\rho)^{-1/2}
}
\]

with band multiplicities \(m_\ell\asymp d_\ell\).

## 5. Main theorem

Assume:

1. eigenvalues are comparable within each band;
2. \(m_\ell/d_\ell\in[\gamma_{\min},\gamma_{\max}]\) with \(0<\gamma_{\min}\le\gamma_{\max}<1\);
3. the smallest band dimension satisfies \(d_{\min}\gtrsim \log(M/\delta)\);
4. band multiplicities preserve the power-law counting measure, \(m_\ell\asymp d_\ell\).

Then with probability at least \(1-\delta\), the band-limited Gaussian feature map has Adam/RMSProp effective eigenvalues comparable to

\[
    \lambda_i(\lambda_i+\rho)^{-1/2}
\]

up to constants depending only on the band comparability, aspect-ratio, and concentration parameters.

If

\[
    \lambda_i\asymp i^{-a},
    \qquad a>1,
\]

then the learned-mode count is

\[
\boxed{
K_{\rm bandGF}(n)
\asymp
\begin{cases}
    n^{2/a}, & n\lesssim \rho^{-1/2},\\[3pt]
    \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2}.
\end{cases}
}
\]

So band-limited Gaussian features preserve the aligned Adam/RMSProp exponent.

## 6. Interpretation

We now have a three-way model-level distinction:

\[
\boxed{
\begin{array}{lll}
\text{spectral/eigenbasis features} & \Rightarrow & q_{\rm eff}=1/2,\\[3pt]
\text{band-limited Gaussian features} & \Rightarrow & q_{\rm eff}=1/2 \text{ up to constants},\\[3pt]
\text{global isotropic Gaussian sketches} & \Rightarrow & q_{\rm eff}=0 \text{ under high-effective-rank flattening.}
\end{array}
}
\]

This is the current best bridge from the diagonal proof to random-feature/sketched models.  The key object is no longer merely the population spectrum; it is the amount of spectral information visible in coordinatewise second moments.

## 7. Experimental consequence

A direct experiment should compare three feature maps with the same population eigenvalues:

1. aligned/eigenbasis features;
2. band-limited Gaussian features;
3. global Gaussian features.

The predicted effective spectral slopes after RMSProp/Adam preconditioning are:

\[
\begin{array}{lll}
\text{aligned/eigenbasis} &:& -a/2,\\
\text{band-limited Gaussian} &:& -a/2,\\
\text{global Gaussian} &:& -a.
\end{array}
\]

This experiment would directly test the main mechanism needed for a high-impact paper: diagonal adaptive optimizers change scaling exponents only when feature coordinates expose spectral structure.
