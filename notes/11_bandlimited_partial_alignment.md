# 11. Band-limited partial alignment preserves Adam/RMSProp spectral flattening

The coordinate-alignment theorem shows two extremes:

\[
\text{eigenbasis coordinates} \Rightarrow q_{\rm eff}=1/2,
\]

while flat or fully mixed coordinates make diagonal adaptivity essentially scalar at exponent level.  This note proves the intermediate case needed for random-feature and sketched models: **mixing within comparable-eigenvalue bands preserves the Adam/RMSProp spectral exponent**.

The message is:

\[
\boxed{
\text{coordinate mixing within spectral bands preserves }q_{\rm eff}=1/2.
}
\]

Only mixing across widely separated spectral scales destroys the spectral information seen by coordinatewise second moments.

## 1. Block/band model of partial alignment

Let

\[
    H=\operatorname{diag}(\lambda_1,\ldots,\lambda_M),
    \qquad
    \lambda_1\ge\lambda_2\ge\cdots>0.
\]

Partition the coordinates into contiguous spectral bands

\[
    B_1,B_2,
    \ldots,B_L.
\]

Assume that each band contains only comparable eigenvalues: there are band scales \(\Lambda_\ell\) and a constant \(\kappa\ge1\) such that for every \(i\in B_\ell\),

\[
\boxed{
    \kappa^{-1}\Lambda_\ell
    \le
    \lambda_i
    \le
    \kappa\Lambda_\ell.
}
\]

Let the optimizer coordinate basis be obtained by an orthogonal matrix \(Q\) that is block diagonal with respect to the same partition:

\[
    Q=\operatorname{blockdiag}(Q_1,\ldots,Q_L).
\]

Thus \(Q\) may mix coordinates arbitrarily inside a spectral band, but it does not mix coordinates whose eigenvalues differ by more than the band factor \(\kappa^2\).  In optimizer coordinates,

\[
    \Sigma=Q^\top H Q.
\]

## 2. Coordinate variances remain band-spectral

For coordinate \(j\in B_\ell\),

\[
    \Sigma_{jj}=q_j^\top H q_j,
\]

where \(q_j\) is supported on \(B_\ell\).  Since \(H\) restricted to \(B_\ell\) satisfies

\[
    \kappa^{-1}\Lambda_\ell I
    \preceq
    H_{B_\ell}
    \preceq
    \kappa\Lambda_\ell I,
\]

we get

\[
\boxed{
    \kappa^{-1}\Lambda_\ell
    \le
    \Sigma_{jj}
    \le
    \kappa\Lambda_\ell,
    \qquad j\in B_\ell.
}
\]

So RMSProp/Adam second moments still identify the spectral band scale, even though they do not identify individual eigenvectors inside the band.

## 3. Effective covariance after diagonal adaptivity

By the coordinate-alignment lemma, RMSProp/Adam tracks

\[
    v_j\asymp d\,\Sigma_{jj},
\]

so after removing the scalar \(d^{-1/2}\), the diagonal preconditioner is

\[
    D_\rho=\operatorname{diag}\left((\Sigma_{jj}+\rho)^{-1/2}\right).
\]

The transformed covariance is

\[
    \widetilde\Sigma=D_\rho^{1/2}\Sigma D_\rho^{1/2}.
\]

Restrict to block \(B_\ell\).  Since \(\Sigma_{jj}\asymp_\kappa\Lambda_\ell\) inside this block,

\[
    D_{\rho,jj}\asymp_\kappa(\Lambda_\ell+\rho)^{-1/2}.
\]

Therefore, in Loewner order,

\[
\boxed{
    c_\kappa(\Lambda_\ell+\rho)^{-1/2}\Sigma_{B_\ell}
    \preceq
    \widetilde\Sigma_{B_\ell}
    \preceq
    C_\kappa(\Lambda_\ell+\rho)^{-1/2}\Sigma_{B_\ell}.
}
\]

Here \(c_\kappa,C_\kappa>0\) depend only on the within-band comparability constant \(\kappa\).

## 4. Main theorem: band-limited mixing preserves the oracle spectrum

The eigenvalues of \(\Sigma_{B_\ell}=Q_\ell^\top H_{B_\ell}Q_\ell\) are exactly the eigenvalues \(\lambda_i\) for \(i\in B_\ell\).  By the min-max principle applied to the Loewner sandwich above, for every \(i\in B_\ell\), the corresponding effective eigenvalue \(\widetilde\mu_i\) satisfies

\[
\boxed{
    c_\kappa
    \lambda_i(\Lambda_\ell+\rho)^{-1/2}
    \le
    \widetilde\mu_i
    \le
    C_\kappa
    \lambda_i(\Lambda_\ell+\rho)^{-1/2}.
}
\]

Since \(\lambda_i\asymp_\kappa\Lambda_\ell\) inside the band,

\[
    \Lambda_\ell+\rho\asymp_\kappa \lambda_i+\rho.
\]

Therefore

\[
\boxed{
    \widetilde\mu_i
    \asymp_\kappa
    \lambda_i(\lambda_i+\rho)^{-1/2}.
}
\]

This is exactly the oracle damped RMSProp/Adam spectrum, up to constants.

Consequently, if

\[
    \lambda_i\asymp i^{-a},
\]

then band-limited coordinate mixing gives the same learned-mode count as aligned coordinates:

\[
\boxed{
K_{\rho,1/2}(n)
\asymp
\begin{cases}
    n^{2/a}, & n\lesssim \rho^{-1/2},\\[3pt]
    \rho^{-1/(2a)}n^{1/a}, & n\gtrsim \rho^{-1/2}.
\end{cases}
}
\]

Thus the exponent-level result is stable to arbitrary rotations within comparable-eigenvalue bands.

## 5. Perturbative off-band leakage

The block-diagonal assumption is idealized.  Now suppose

\[
    \Sigma=\Sigma_{\rm band}+R,
\]

where \(\Sigma_{\rm band}\) is block diagonal with the spectral-band structure above and \(R\) is an off-band leakage term.

Let \(D_\rho\) be the diagonal preconditioner from \(\Sigma\), and define

\[
    \widetilde\Sigma=D_\rho^{1/2}\Sigma D_\rho^{1/2},
    \qquad
    \widetilde\Sigma_{\rm band}=D_\rho^{1/2}\Sigma_{\rm band}D_\rho^{1/2}.
\]

If

\[
\boxed{
    \left\|D_\rho^{1/2}R D_\rho^{1/2}\right\|_{\rm op}\le \varepsilon,
}
\]

then Weyl's inequality gives

\[
\boxed{
    \left|\lambda_k(\widetilde\Sigma)-
    \lambda_k(\widetilde\Sigma_{\rm band})\right|\le \varepsilon
    \qquad\text{for all }k.
}
\]

Therefore every learned-mode count away from the threshold by a margin larger than \(\varepsilon\) is unchanged.  More explicitly, if

\[
    K_-=\#\{k:\lambda_k(\widetilde\Sigma_{\rm band})\ge n^{-1}+\varepsilon\},
\]

and

\[
    K_+=\#\{k:\lambda_k(\widetilde\Sigma_{\rm band})\ge n^{-1}-\varepsilon\},
\]

then

\[
\boxed{
    K_-\le K_{\widetilde\Sigma}(n)\le K_+.
}
\]

Thus small off-band leakage perturbs constants but not exponents, provided the leakage is small compared with the spectral threshold being tested.

## 6. Interpretation

The alignment picture now has three regimes:

\[
\boxed{
\begin{array}{lll}
\text{aligned coordinates} & \Rightarrow & q_{\rm eff}=1/2,\\[3pt]
\text{band-limited mixing} & \Rightarrow & q_{\rm eff}=1/2\text{ up to constants},\\[3pt]
\text{flat/global mixing} & \Rightarrow & q_{\rm eff}=0\text{ at exponent level.}
\end{array}
}
\]

This is the bridge to sketched/random-feature models.  A random feature map may not align exactly with population eigenvectors, but if each trainable coordinate mostly mixes features from a narrow spectral band, Adam/RMSProp should still show the damped \(q=1/2\) exponent.  If the trainable coordinates mix all spectral scales evenly, the diagonal second moment becomes flat and the Adam/RMSProp exponent improvement should disappear.

## 7. Next theorem target

The next theoretical goal is to express the same band-limited condition for a Gaussian sketch or random-feature matrix.  The desired statement is a quantitative alignment parameter, computable from the feature map, that predicts an effective exponent between the two extremes:

\[
    q_{\rm eff}\in[0,1/2].
\]

A natural candidate is the power-law slope of the coordinate variance profile

\[
    d_j=(S H S^\top)_{jj}
\]

after sorting, together with an off-band leakage norm.  The experiments from `run_sweeps.py` should guide which alignment statistic is most predictive.
