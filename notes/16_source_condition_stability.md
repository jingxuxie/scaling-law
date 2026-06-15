# 16. Source-condition stability under visible spectral preconditioning

The visible-spectrum theorem in `15_visible_spectrum_scaling_law.md` assumes a source condition in the transformed covariance basis.  This note proves when that assumption is automatic.  The main message is:

\[
\boxed{
\text{If the adaptive preconditioner is spectral, or block-spectral on invariant bands, then the source exponent is preserved.}
}
\]

This closes an important gap between the covariance/eigenvalue theory and the bias term in the risk law.

## 1. Exact spectral preconditioners preserve source energies

Let

\[
    \Sigma v_i=\lambda_i v_i,
    \qquad
    \lambda_i>0,
\]

and suppose the original source energies obey

\[
    s_i:=\lambda_i\langle v_i,w^\star\rangle^2.
\]

Let the preconditioner be an exact spectral function

\[
    P=f(\Sigma),
    \qquad f(\lambda_i)>0.
\]

The transformed covariance is

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2}=\Sigma f(\Sigma),
\]

so its eigenvectors are still \(v_i\), and its eigenvalues are

\[
    \widetilde\mu_i=\lambda_i f(\lambda_i).
\]

The transformed parameter is

\[
    u^\star=P^{-1/2}w^\star.
\]

Therefore

\[
\begin{aligned}
    \widetilde s_i
    &:=
    \widetilde\mu_i\langle v_i,u^\star\rangle^2 \\
    &=
    \lambda_i f(\lambda_i)
    \left[f(\lambda_i)^{-1/2}\langle v_i,w^\star\rangle\right]^2 \\
    &=
    \lambda_i\langle v_i,w^\star\rangle^2
    =s_i.
\end{aligned}
\]

Thus:

\[
\boxed{
    \widetilde s_i=s_i
    \qquad\text{for every }i.
}
\]

In particular, if

\[
    s_i\asymp i^{-b},
\]

then the transformed source condition has exactly the same exponent \(b\).

This applies to the aligned diagonal Adam/RMSProp theorem and to any coordinate system where the diagonal adaptive preconditioner is genuinely comparable to a spectral function and commutes with \(\Sigma\).

## 2. Block-invariant preconditioners preserve block source mass

The noncommuting case is subtler.  A useful intermediate case is the band-limited model from `11_bandlimited_partial_alignment.md`.

Assume the covariance and preconditioner share a block decomposition:

\[
    \Sigma=\operatorname{blockdiag}(\Sigma_1,\ldots,\Sigma_L),
    \qquad
    P=\operatorname{blockdiag}(P_1,\ldots,P_L).
\]

Let

\[
    \widetilde\Sigma=P^{1/2}\Sigma P^{1/2},
    \qquad
    u^\star=P^{-1/2}w^\star.
\]

For block \(B_\ell\), define the original block source mass

\[
    S_\ell=(w^\star_\ell)^\top\Sigma_\ell w^\star_\ell
\]

and the transformed block source mass

\[
    \widetilde S_\ell=(u^\star_\ell)^\top\widetilde\Sigma_\ell u^\star_\ell.
\]

Then

\[
\begin{aligned}
    \widetilde S_\ell
    &=
    (P_\ell^{-1/2}w^\star_\ell)^\top
    (P_\ell^{1/2}\Sigma_\ell P_\ell^{1/2})
    (P_\ell^{-1/2}w^\star_\ell) \\
    &=
    (w^\star_\ell)^\top\Sigma_\ell w^\star_\ell
    =S_\ell.
\end{aligned}
\]

So even if \(P_\ell\) does not commute with \(\Sigma_\ell\), the total source mass inside an invariant block is exactly preserved:

\[
\boxed{
    \widetilde S_\ell=S_\ell.
}
\]

## 3. Band source conditions

For scaling laws, individual coordinates are often less important than dyadic or geometric spectral bands.  Let bands \(B_\ell\) be chosen so that eigenvalues inside a band are comparable:

\[
    \lambda_i\asymp \Lambda_\ell,
    \qquad i\in B_\ell.
\]

Define a band source condition by

\[
\boxed{
    S_\ell
    :=
    \sum_{i\in B_\ell}\lambda_i\langle v_i,w^\star\rangle^2
    \asymp
    \sum_{i\in B_\ell}i^{-b}.
}
\]

If the transformed covariance has the same invariant bands and the transformed block source masses satisfy \(\widetilde S_\ell=S_\ell\), then the transformed problem satisfies the same band source exponent \(b\).

This is enough for the spectral-sum proof of the bias term because the filters

\[
    \frac{1}{1+n\widetilde\mu_i}
\]

are constant up to factors inside comparable-eigenvalue bands.

## Theorem 1: band source stability implies the same bias exponent

Assume:

1. the transformed effective eigenvalues are two-slope with exponent \(\alpha\) before the damping knee and exponent \(a\) after it;
2. the spectral bands have bounded comparability;
3. the transformed block source masses obey the same band source condition with exponent \(b\);
4. \(1<b<\alpha+1\).

Then the bias filter satisfies

\[
\boxed{
    \sum_{i\le M}\frac{\widetilde s_i}{1+n\widetilde\mu_i}
    \asymp
    K(n)^{1-b}
}
\]

whenever \(K(n)<M\), where

\[
    K(n)=\#\{i:\widetilde\mu_i\gtrsim n^{-1}\}.
\]

### Proof sketch

Group the spectral sum by bands.  Inside each band, the filter value is constant up to a multiplicative constant because the effective eigenvalues are comparable.  The grouped sum is therefore

\[
    \sum_\ell
    \frac{\widetilde S_\ell}{1+n\widetilde\Lambda_\ell},
\]

where \(\widetilde\Lambda_\ell\) is the effective eigenvalue scale in band \(B_\ell\).  By block source stability, \(\widetilde S_\ell=S_\ell\), and by the band source condition \(S_\ell\asymp\sum_{i\in B_\ell}i^{-b}\).  This reduces the proof to the same integral comparison used in `09_matching_filter_lower_bounds.md`, giving \(K(n)^{1-b}\).

## 4. Consequences for the paper

The transformed source assumption in the visible-spectrum theorem is automatic in the following cases:

\[
\boxed{
\begin{array}{lll}
\text{exact spectral preconditioner} & \Rightarrow & \widetilde s_i=s_i,\\[3pt]
\text{aligned coordinates} & \Rightarrow & \widetilde s_i=s_i,\\[3pt]
\text{band-limited invariant coordinates} & \Rightarrow & \widetilde S_\ell=S_\ell.\\
\end{array}
}
\]

For globally mixed coordinates, the covariance spectrum alone is not enough.  One must check whether the target/source energy remains regular in the transformed eigenbasis.  This explains why future experiments should report not only \(q_{\rm eff}\), but also a transformed-source diagnostic.

## 5. Recommended diagnostic

For any experiment with a covariance \(\Sigma\), preconditioner \(P\), and target \(w^\star\):

1. compute \(\widetilde\Sigma=P^{1/2}\Sigma P^{1/2}\);
2. compute its eigenpairs \((\widetilde\mu_i,\widetilde v_i)\);
3. set \(u^\star=P^{-1/2}w^\star\);
4. measure

\[
    \widetilde s_i=\widetilde\mu_i\langle \widetilde v_i,u^\star\rangle^2;
\]

5. fit the slope of sorted or band-averaged \(\widetilde s_i\).

If this slope is stable across coordinate systems, then observed risk differences can be attributed mainly to the visible spectrum \(q_{\rm eff}\).  If it changes, then target/source rotation is also affecting the scaling law.
