# 22. Consistency of the fitted q_eff diagnostic

Most experiments report an empirical optimizer exponent

\[
    \widehat q_{\rm eff}=1-\widehat\alpha_{\rm eff}/\widehat\alpha_{\rm orig},
\]

where \(\widehat\alpha_{\rm orig}\) and \(\widehat\alpha_{\rm eff}\) are log-log fitted spectral slopes before and after diagonal adaptive preconditioning.  This note proves a simple robustness lemma justifying that diagnostic.

## Lemma: slope fitting is stable under multiplicative spectral errors

Let \(i\in[L,U]\) be the fitting window, and suppose

\[
    \lambda_i=c_\lambda i^{-a}e^{\varepsilon_i},
    \qquad
    \mu_i=c_\mu i^{-\alpha}e^{\delta_i},
\]

with

\[
    |\varepsilon_i|\le \eta,
    \qquad
    |\delta_i|\le \eta.
\]

Let \(\widehat a\) and \(\widehat\alpha\) be ordinary least-squares slopes of \(\log \lambda_i\) and \(\log \mu_i\) against \(\log i\) on the same window, with signs chosen so that positive slopes mean decay exponents.  Define

\[
    q=1-\alpha/a,
    \qquad
    \widehat q=1-\widehat\alpha/\widehat a.
\]

If \(a>0\) and the logarithmic window width \(W=\log(U/L)\) is bounded away from zero, then

\[
\boxed{
    |\widehat q-q|
    \lesssim
    \frac{\eta}{aW}
    +
    \frac{\eta}{W},
}
\]

provided \(\widehat a\) remains bounded below by a positive constant.

## Proof

Write

\[
    x_i=\log i,
    \qquad
    y_i=\log\lambda_i=\log c_\lambda-a x_i+\varepsilon_i.
\]

The least-squares slope error obeys

\[
    \widehat a-a
    =
    -\frac{\sum_i (x_i-\bar x)\varepsilon_i}{\sum_i (x_i-\bar x)^2}.
\]

Since \(|\varepsilon_i|\le\eta\),

\[
    |\widehat a-a|
    \le
    \eta
    \frac{\sum_i |x_i-\bar x|}{\sum_i (x_i-\bar x)^2}
    \lesssim
    \frac{\eta}{W}.
\]

The same argument gives

\[
    |\widehat\alpha-\alpha|\lesssim \eta/W.
\]

Finally,

\[
\begin{aligned}
    |\widehat q-q|
    &=
    \left|\frac{\alpha}{a}-\frac{\widehat\alpha}{\widehat a}\right| \\
    &\le
    \frac{|\widehat\alpha-\alpha|}{|\widehat a|}
    +
    |\alpha|
    \left|\frac1a-\frac1{\widehat a}\right|,
\end{aligned}
\]

which is controlled by the two slope errors when \(a\) and \(\widehat a\) are bounded away from zero.  This proves the claim.

## Consequence for the experiments

The fitted \(q_{\rm eff}\) diagnostic is reliable when the original and effective spectra are approximately power-law over the same fitting window.  This is why the experiments report both effective spectral slopes and \(q_{\rm eff}\), rather than relying only on finite-time risk slopes.  Risk slopes include constants, learning-rate effects, and transients; \(q_{\rm eff}\) measures the spectral scaling mechanism directly.
