# Citlali convolve statistical contract: independent pre-comparison derivation

Date: 2026-07-31

This memo was completed before opening or reading the independent audit at
`codex/convolve-contract-audit` / `800e8ae433f87d3fb7521fcb1a7fdf1d32532949`.
It records a source-derived analysis of these fixed revisions:

- application mainline: `9aae0e669384c5c0c0dda93debc194d6b8dac787`;
- bounded predecessor: `fa8f8a5b939aa8e78f1b2fc0affc86857f2b2f9c`;
- candidate: `02a198cbfb379eaf6ab279c5a3d44ee73ff90435`.

No conclusion below is based on the independent audit.

## 1. Exact implemented signal and noise operators

Let the map domain be the finite periodic grid
`G = Z_R x Z_C`. Let `K` be the supplied filter template and

```
k_q = K_q / sum_r K_r,
(C_k x)_p = sum_q k_q x_(p-q mod (R,C)).
```

The kernel normalization is global, not local. The implementation rejects a
non-finite, zero-sum, or strongly cancellation-dominated kernel; specifically,
it requires `abs(sum K) / sum(abs K) >= 0.05`. The FFT operation is circular:
there is no zero padding, and convolve mode sets its denominator plane to one.

Let `e_p` be the realized edge window and `b` the finite median of the signal
inside the realized science mask when `fill_mode=core_median` (`b=0` for zero
fill). Both the sequential and OpenMP implementations perform

```
xg_p = b + e_p (x_p - b),
y_p  = e_p (C_k xg)_p
     = e_p [b + C_k(e (x-b))_p],
n'_rp = e_p C_k(e n_r)_p
```

for every noise realization `r`. The median fill is used for signal only;
noise realizations are zero-apodized. Signal filtering does not use the formal
weight eligibility mask and has no local-support normalization.

Conditional on fixed `e`, fixed `b`, and fixed `k`, the signal operator is
affine. Unconditionally it is not a fixed linear estimator because `b` is a
sample median and the science/edge mask is data-product dependent. The code
does not propagate the covariance of the estimated median or mask selection.

The sequential and OpenMP source paths implement the same operator. The
OpenMP noise path uses thread-local FFT state but the same `D_e C_k D_e`
mapping. Existing tests compile against one implementation at a time; a
direct equivalence regression remains desirable.

## 2. Conditional covariance and the diagonal formal target

Writing `D_e = diag(e)`, the conditional noise covariance is

```
Cov(n' | e,b,k) = D_e C_k D_e Sigma D_e C_k^T D_e.
```

For conditionally independent input pixels with
`Sigma = diag(1/W_i)`, its diagonal is

```
V_y(p) = e_p^2 sum_q k_q^2 e_(p-q)^2 / W_(p-q).
```

The filter necessarily induces off-diagonal output covariance. A stored
per-pixel weight or variance plane can describe only this conditional
diagonal, not the full covariance. It also omits uncertainty in `b`, in the
selected mask, and in the filter kernel.

For binary accepted edge windows (`e` in `{0,1}`), define

```
a_i = 1 iff the guarded input weight is finite and positive,
q_p = C_(k^2)(a / W)_p,
m_p = C_(k^2)(a)_p,
K2  = sum_q k_q^2.
```

On an output pixel with `e_p=1`, the desired conditional diagonal is `q_p`
provided every stochastic signal/noise input is represented by the same
eligibility rule. Its inverse is `1/q_p`. Pixels with `e_p=0` are invalidated
with zero stored signal and zero stored weight; zero weight there is a
sentinel, not the reciprocal of a physical zero variance.

For a fractional edge window, however, the implementation first stores
`Wg_i = W_i e_i^2`, propagates `1/Wg_i`, and then multiplies the output weight
by `e_p^2`. The reciprocal code weight is therefore

```
V_code(p) = e_p^(-2) sum_q k_q^2 a_(p-q) /
            [W_(p-q) e_(p-q)^2],
```

which has inverse taper powers and disagrees with `V_y`. A delta-kernel
example gives actual variance `e^4/W` but reported variance `1/(W e^4)`.
The candidate correctly makes accepted unit-sum convolve and Wiener-lowpass
configurations fail closed on cosine/fractional taper. Lower-level calls that
bypass typed validation can still execute the invalid combination.

## 3. Revision-by-revision formal variance

### Application mainline `9aae0e6`

The mainline code computes a `cov_cut`-thresholded eligibility mask, then

```
q_p = C_(k^2)(a/W)_p,
m_p = C_(k^2)(a)_p,
W_pre(p) = m_p / q_p,
W_out(p) = e_p^2 m_p / q_p.
```

Thus the represented variance is locally divided by squared-kernel support.
With full uniform coverage it reports the input variance rather than the
variance of the stored convolved amplitude (`V_in K2`). This is not the
variance of the implemented fixed, globally normalized convolution.

### Bounded predecessor `fa8f8a5`

The predecessor removes the local-support normalization:

```
W_pre(p) = 1/q_p,
W_out(p) = e_p^2/q_p.
```

It also changes the support check to the relative numerical guard
`m_p > 1e-6 K2`. However, it retains `cov_cut` in formal-variance input
eligibility. Positive low-weight pixels can therefore enter the stored signal
and filtered noise but be omitted from formal variance, violating the
same-estimator requirement.

### Candidate `02a198c`

The candidate retains `W_pre=1/q`, removes `cov_cut` from variance
eligibility, and uses every finite positive post-edge guarded weight. In the
accepted binary-window configuration this matches the conditional diagonal
variance of the implemented noise operator, subject to these preconditions:

- input pixels with zero/non-finite formal weight do not carry an unmodelled
  stochastic signal/noise contribution;
- input covariance is diagonal with variance `1/W`;
- the edge window and core-median fill are conditioned on as fixed;
- the stored formal product represents only a covariance diagonal;
- typed validation has not been bypassed to admit fractional taper.

The natural normalized squared-kernel overlap is

```
s2_p = m_p / K2.
```

The candidate uses `m_p > 1e-6 K2` only as a tiny numerical overlap guard.
It does not divide by `m_p`, so the guard is not a response normalization or
scientific confidence cut. Equality to the threshold is invalid. This guard
should remain distinct from any persisted support/confidence product.

A hand case with a 2x2 uniform kernel and weights `(100,100,100,1)` gives
candidate variance `(1/16)(3/100+1)` and weight `16/1.03`. The predecessor
drops the low-weight contributor under `cov_cut=0.5` and instead reports
`1600/3`, despite that pixel participating in signal/noise convolution.

## 4. Zero-weight and low-weight semantics

The filter core does not weight-mask the signal. Normal map preparation
usually zeros signal/noise values that fail its earlier validity threshold.
Consequently, a zero-weight pixel is treated by formal propagation as a
deterministic, non-random input. If a caller supplies a nonzero stochastic
signal/noise value with zero weight and `e=1`, the formal contract is false.
This must be an explicit precondition and regression case.

Every finite positive weight that survives map preparation and has `e=1`
enters both the signal/noise operator and candidate formal variance, including
weights below the science `cov_cut`. `cov_cut` still affects construction of
the science/edge mask and later product selection/calibration; it is not a
separate censor for the variance sum.

## 5. Empirical noise estimator

For mean-subtracted realizations, mainline uses a population central moment.
It consequently reports zero variance for `N=1`, and for `N=2` gives
`(n1-n2)^2/4`. The predecessor applies `N/(N-1)` but uses a cancellation-prone
`mean(n^2)-mean(n)^2` calculation. The candidate uses Welford accumulation and

```
V_emp = M2/(N-1),  N >= 2,
```

rejecting a mean-subtracted singleton. For `N=2`, this is
`(n1-n2)^2/2`. In the explicitly known-zero-mean path, the estimator remains
`mean(n^2)` and a singleton is permitted.

Interpreting `M2/(N-1)` as unbiased for a common marginal variance requires
independent, identically distributed or exchangeable realizations with a
common unknown mean and finite second moment; Gaussianity is unnecessary.
Actual jackknife construction may not provide independence, so scientific
calibration still requires empirical validation. The `N-1` issue is treated
as a known implementation error only; no further bias analysis is proposed.

Empirical products are per-pixel diagonals. They do not represent the
filter-induced spatial covariance or median/mask-selection uncertainty.

The empirical rescaling is scalar, not a pixelwise empirical inverse
variance. The code snapshots the incoming filtered `weight` as
`weight_formal`, calculates `V_emp`, estimates a robust scalar from products
`weight_formal * V_emp` over an eligible calibration set, and applies that
scalar to the formal plane. Depending on pipeline order, this incoming
`weight_formal` may itself already be an empirically rescaled upstream weight,
so “formal” does not necessarily mean pristine mapmaker variance.

## 6. Units and product meanings

If input signal has unit `U` (normally a surface/beam-calibrated astronomical
amplitude such as mJy/beam), then:

- stored filtered signal / convolved amplitude: `U`;
- conditional formal diagonal variance: `U^2`;
- formal or empirically scalar-calibrated weight: `U^-2`;
- empirical sample variance and point-source uncertainty squared: `U^2`;
- S/N, numerical support, masks, and responses: dimensionless;
- coverage: time (seconds in current products).

In the candidate, `sig2noise` and `sig2noise_pixel` are exact aliases formed
from stored signal and the current calibrated-formal weight. The point-source
flux alias is also numerically the stored convolved amplitude. For convolve
and lowpass modes, the candidate truthfully labels it a convolved amplitude
and removes the false `RESPNORM=1` declaration. Full Wiener mode retains its
separate response-normalized contract.

Unit-sum convolution preserves a uniform/DC field. It does not, by itself,
normalize a point-source peak or a template-fit amplitude. A
response-corrected sky estimator would require a named source template and a
named response, for example

```
R_p = (D_e C_k D_e template)_p,
Ahat_p = y_p/R_p,
Var(Ahat_p) = V_y(p)/R_p^2,
W_A(p) = R_p^2/V_y(p).
```

No such authoritative `R` is currently computed or applied. A filtered
kernel plane may be relevant evidence, but is not presently a defined
response normalization.

## 7. Support, validity, response, and confidence are distinct

The current `coverage_bool` plane is derived from final weight and a
`cov_cut` threshold. It is not raw detector coverage, the convolution input
mask, squared-kernel support `s2`, a point-source response, the empirical
calibration set, or a complete numerical-validity mask. At a zero threshold,
zero and non-finite edge cases can even be classified counterintuitively.
It must not be reused or relabelled as convolution support.

The contracts that must remain separate are:

1. numerical validity: finite positive propagated variance and overlap above
   the tiny implementation guard, plus a valid binary output edge pixel;
2. convolution support: dimensionless squared-kernel overlap such as
   `s2=m/K2`, if explicitly persisted;
3. template/source response: a separately defined `R` for a specified source
   model;
4. science confidence/selection: a validated policy using support, coverage,
   noise, and possibly response, not an alias of any one of them.

The smallest truthful product strategy is either to persist an explicit
dimensionless convolution-support plane with its own metadata, or to withhold
any filtered downstream product whose contract requires support until such a
plane exists. It is not acceptable to overload `coverage_bool`.

## 8. Fruit-loop consequence

The convolved amplitude is a plausible future detection/selection signal, but
it is not presently a response-corrected recovered sky estimator. Formal and
empirical planes contain only conditional diagonal uncertainty, and edge
support/response contracts are not complete. Therefore filtered amplitude or
weight must not yet be routed into fruit-loop feedback. Any new spatial
fruit-loop estimator should remain fail closed on filtered products pending
independent derivation, product-contract review, and empirical recovery tests.

## 9. Proposed minimal repair boundary and tests

The candidate’s two numerical corrections are directionally required:

- fixed-convolution diagonal variance with no local-support division;
- all finite positive surviving stochastic contributors included;
- stable `N-1` central sample variance and rejection of `N<2`;
- fail-closed rejection of fractional taper for accepted convolve/lowpass;
- truthful convolved-amplitude aliases rather than false response claims.

Before scientific acceptance, equation-based tests should cover at least:

- delta kernel identity for signal and formal variance;
- uniform kernel/full support (`V=V_in sum k^2`);
- a hand-computed nonuniform kernel and nonuniform weights;
- positive weight below `cov_cut` still included in variance;
- zero-weight deterministic-input precondition and mismatch case;
- binary edge mask, median fill, output mask, and circular wrap behavior;
- exact numerical-support threshold boundary (`>` versus `==`);
- explicit support-product normalization and metadata, if persisted;
- serial/OpenMP numerical equivalence;
- `N=1` mean-subtracted rejection, `N=2` exact result, large-offset
  stability, and known-zero-mean singleton behavior;
- alias equality and accurate FITS type/unit/estimator metadata;
- explicit distinction between stored-amplitude variance and any future
  response-corrected variance.

Passing unit/regression/build gates would validate implementation against this
stated conditional model. It would not establish astronomical source-recovery
bias, correlated-noise calibration, point-source response, edge completeness,
or scientific acceptance; those require controlled recovery experiments and
reviewed data-product semantics.
