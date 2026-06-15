# 10. Coordinate alignment: when diagonal adaptivity is spectral

The diagonal Gaussian theory proves that RMSProp/Adam learns a damped spectral preconditioner when optimizer coordinates are covariance eigenvectors.  This note proves the next structural point: **coordinatewise adaptivity is spectral only to the extent that coordinate variances reveal spectral directions**.

In arbitrary optimizer coordinates, the second-moment estimator tracks diagonal entries of the covariance matrix in those coordinates, not the eigenvalues.  If those diagonal entries are flat, RMSProp/Adam becomes essentially scalar preconditioning and the spectral exponent is not improved.

## 1. Arbitrary coordinate basis

Let the covariance in optimizer coordinates be

\[
    \Sigma=Q^\top H Q,
\]

where \(H=\operatorname{diag}(\lambda_i)\) is the eigenbasis covariance and \(Q\) is orthogonal.  The data in optimizer coordinates satisfy

\[
    z\sim \mathcal N(0,\Sigma).
\]

For squared loss, at error vector \(e=w-w^\star\), the stochastic gradient is

\[
    g=(\langle z,e\rangle-\xi)z.
\]

Define

\[
    c_e=\sigma^2+\|e\|_\Sigma^2.
\]

## Lemma 1: second moments track coordinate variances

For every coordinate \(j\),

\[
\boxed{
    \mathbb E[g_j^2\mid e]
    =
    c_e\Sigma_{jj}+2(\Sigma e)_j^2.
}
\]

Consequently,

\[
\boxed{
    c_e\Sigma_{jj}
    \le
    \mathbb E[g_j^2\mid e]
    \le
    3c_e\Sigma_{jj}.
}
\]

### Proof

Let

\[
    U=\langle z,e\rangle-\xi,
    \qquad
    Z_j=z_j.
\]

Conditionally on \(e\), \((U,Z_j)\) is centered Gaussian and

\[
    \mathbb E[U^2]=c_e,
    \qquad
    \mathbb E[Z_j^2]=\Sigma_{jj},
    \qquad
    \mathbb E[UZ_j]=(\Sigma e)_j.
\]

By Isserlis' identity,

\[
    \mathbb E[U^2Z_j^2]
    =
    \mathbb E[U^2]\mathbb E[Z_j^2]
    +2\mathbb E[UZ_j]^2
    =
    c_e\Sigma_{jj}+2(\Sigma e)_j^2.
\]

For the upper bound, use Cauchy-Schwarz in the \(\Sigma\)-inner product:

\[
    (\Sigma e)_j^2
    =
    (e^\top \Sigma e_j)^2
    \le
    (e^\top\Sigma e)(e_j^\top\Sigma e_j)
    =
    \|e\|_\Sigma^2\Sigma_{jj}
    \le
    c_e\Sigma_{jj}.
\]

This proves the claim.

## 2. The actual diagonal preconditioner in arbitrary coordinates

Lemma 1 implies that an RMSProp/Adam second-moment estimate satisfies

\[
    v_j\asymp d\,\Sigma_{jj},
\]

where \(d\) is a scalar residual-energy EMA.  Therefore the coordinatewise preconditioner has the form

\[
    P_j=(v_j+\epsilon)^{-1/2}
    \asymp
    d^{-1/2}(\Sigma_{jj}+\rho)^{-1/2},
    \qquad
    \rho=\epsilon/d.
\]

Ignoring the scalar \(d^{-1/2}\), define

\[
    D_\rho=\operatorname{diag}\left((\Sigma_{jj}+\rho)^{-1/2}\right).
\]

The transformed covariance is

\[
\boxed{
    \widetilde\Sigma
    =
    D_\rho^{1/2}\Sigma D_\rho^{1/2}.
}
\]

Thus diagonal adaptivity depends on the diagonal of \(\Sigma\), not directly on the eigenvalues of \(\Sigma\).

## 3. Aligned coordinates recover the damped \(q=1/2\) theorem

If \(Q=I\), then

\[
    \Sigma_{jj}=\lambda_j.
\]

So

\[
    D_\rho=(H+\rho I)^{-1/2},
\]

and

\[
    \widetilde\Sigma
    =
    (H+\rho I)^{-1/4}H(H+\rho I)^{-1/4}.
\]

The eigenvalues are

\[
\boxed{
    \mu_j=\lambda_j(\lambda_j+\rho)^{-1/2}.
}
\]

Therefore aligned coordinates give the Adam/RMSProp learned-mode count

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

This is the mechanism proved in the previous notes.

## 4. Flat diagonal coordinates destroy spectral adaptivity

Suppose instead that the optimizer-coordinate variances are flat up to constants:

\[
    d_{\min}\le \Sigma_{jj}\le d_{\max},
    \qquad
    \kappa_d:=d_{\max}/d_{\min}<\infty.
\]

Then every preconditioner entry satisfies

\[
    (d_{\max}+\rho)^{-1/2}\le D_{\rho,jj}\le (d_{\min}+\rho)^{-1/2}.
\]

Hence, in Loewner order,

\[
\boxed{
    (d_{\max}+\rho)^{-1/2}\Sigma
    \preceq
    \widetilde\Sigma
    \preceq
    (d_{\min}+\rho)^{-1/2}\Sigma.
}
\]

Consequently, every eigenvalue of \(\widetilde\Sigma\) is within a constant factor of the corresponding eigenvalue of \(\Sigma\):

\[
\boxed{
    (d_{\max}+\rho)^{-1/2}\lambda_k(\Sigma)
    \le
    \lambda_k(\widetilde\Sigma)
    \le
    (d_{\min}+\rho)^{-1/2}\lambda_k(\Sigma).
}
\]

Therefore diagonal RMSProp/Adam does **not** change the spectral exponent.  If

\[
    \lambda_k(\Sigma)\asymp k^{-a},
\]

then also

\[
    \lambda_k(\widetilde\Sigma)\asymp k^{-a}.
\]

The learned-mode count remains SGD-like:

\[
\boxed{
    K_{\rm flat}(n)\asymp n^{1/a}
}
\]

up to constants.  There is no \(a\mapsto a/2\) flattening.

## 5. Hadamard/flat rotation corollary

Assume dimension \(M\) admits a normalized Hadamard matrix \(Q\), so

\[
    Q_{ij}^2=1/M.
\]

Then

\[
    \Sigma=Q^\top H Q
\]

has exactly flat diagonal:

\[
\boxed{
    \Sigma_{jj}=\frac1M\operatorname{tr}(H)
    \qquad\text{for all }j.
}
\]

Thus RMSProp/Adam's coordinatewise second moment is the same in every coordinate, up to the scalar residual-energy factor, and the adaptive preconditioner is scalar:

\[
    D_\rho=c_\rho I.
\]

Therefore

\[
    \widetilde\Sigma=c_\rho\Sigma,
\]

and the eigenvalue decay is unchanged.  In a perfectly flat coordinate system, diagonal adaptivity cannot act as spectral preconditioning.

## 6. Random rotations

For a Haar-random orthogonal basis, the coordinate variances are

\[
    \Sigma_{jj}=q_j^\top Hq_j.
\]

They estimate the average spectral mass seen by coordinate \(q_j\), not the ordered eigenvalues \(\lambda_j\).  When the effective rank of \(H\),

\[
    r_{\rm eff}=\frac{\operatorname{tr}(H)^2}{\operatorname{tr}(H^2)},
\]

is large enough, standard concentration of quadratic forms on the sphere gives

\[
    \Sigma_{jj}\approx \operatorname{tr}(H)/M
\]

uniformly over \(j\), and the flat-diagonal theorem applies.  For trace-class power laws with small effective rank, the diagonals may fluctuate by constants or logarithms, but they still do not encode the ordered power law \(\lambda_j\asymp j^{-a}\).  The experiment `experiments/coordinate_alignment.py` checks this finite-dimensional behavior.

## 7. Main coordinate-alignment theorem

Combining the cases above:

\[
\boxed{
\begin{array}{ll}
\text{aligned optimizer coordinates} & \Rightarrow q_{\rm eff}=1/2,\\[3pt]
\text{flat/bounded-variance coordinates} & \Rightarrow q_{\rm eff}=0\text{ at exponent level.}
\end{array}
}
\]

Thus coordinatewise adaptivity changes scaling exponents only when the coordinatewise gradient second moments retain spectral information.  This is not a weakness of the diagonal theory; it is a testable prediction.

## 8. Consequence for experiments and random features

The next empirical check is to compare:

1. aligned diagonal Gaussian regression;
2. the same problem after a flat Hadamard rotation;
3. the same problem after a Haar-random rotation;
4. Gaussian-sketched/random-feature models where optimizer coordinates may or may not align with population eigenmodes.

The theory predicts that Adam/RMSProp should show the \(a/2\) effective exponent in the aligned case, but should revert toward the SGD exponent in flat or sufficiently mixed coordinates.  This is the main bridge from the diagonal proof to realistic models: the optimizer-dependent exponent depends on the alignment between optimizer coordinates and spectral structure.
