# SCI-NOI v0.1 Stage B Scientific Rationale

Document identity: `SCI-NOI_SCIENTIFIC_RATIONALE v0.1/draft-r0.1`

Status: implementation-blind Stage B draft; not owner-accepted, frozen,
numerically available, validated, ready, or production-authorized.

Scientific owner: Grant Wilson

Scientific authoring basis: the exact 17-object author packet bound by
`SCI-NOI_AUTHOR_PACKET_MANIFEST v0.1/r0.18`, manifest SHA-256
`b6f8e7252e7f61f4506899cb3e8e26cf939887bb48464852713f8ce81ac77ca0`.

## 1. Why the three NOI roles must remain separate

SCI-NOI describes three different questions.

- `NOI-GEN` asks what finite collection of conditional randomizations was
  generated from one exact immutable parent and one exact operator graph.
- `NOI-UNC` asks what empirical quantity a named estimator infers from one
  admitted ensemble.
- `NOI-STD` asks how one immutable signal numerator is divided by one
  compatible positive uncertainty scale.

A realization ensemble is not an uncertainty estimate merely because it is
called a noise or jackknife ensemble. An empirical second moment is not a
covariance merely because it is stored as a pointwise field. A reciprocal
second-moment scale is not inverse variance or precision. A dimensionless
standardized map is not a Gaussian significance map. These distinctions are
the central scientific safety boundary of this contract.

## 2. The ordinary conditional-randomization question

The ordinary route is
`NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1`. It begins with exact retained
PTC occurrences and inserts an NOI-owned sign modifier at the exact PTC-to-MAP
numerical boundary. The complete frozen MAP accumulation then produces an NOI
realization map. MAP may apply the modifier inline, or the same operation may
be materialized, but the scientific object is the same only when the identical
modifier acts on the identical admitted occurrence before identical frozen
MAP arithmetic. The output is not an ordinary MAP science product.

For member `b`, the fixed-state relation is

```text
M_b = O_Theta0(R_b(parent)).
```

The learned state `Theta0` is fixed. A procedure that instead derives
`Theta_b` from each randomized parent asks another scientific question:

```text
Theta_b = LearnResolve_b(R_b(parent)),
M_b     = O_Theta_b(R_b(parent)).
```

Those members must have another method identity and cannot enter the same
uncertainty estimate as fixed-state members.

## 3. Coherence and network-local balance

The ordinary coherence unit is one stable realized detector/channel within
one exact observation. One assigned sign applies to every admitted PTC
occurrence for that detector in that observation. Time, scan, chunk, traversal,
worker, and container boundaries cannot change the assignment.

For each observation/readout-network stratum `h`, let `D_h` be the canonical
ordered detector population. The exact frozen MAP-admitted positive
contributions define the detector coefficient mass

```text
B_d = sum_p sum_{i in C_p, detector(i)=d} G_pi gamma_i,   B_d > 0.
```

For a candidate sign vector `s_h`, define the dimensionless imbalance

```text
Delta_h(s_h) = abs(sum_{d in D_h} s_d B_d) / sum_{d in D_h} B_d.
```

The Stage B design admits `s_h` exactly when

```text
abs(sum_d s_d B_d) <= tau_h sum_d B_d,
```

where `tau_h` is an explicitly requested exact rational in `[0,1)`. There is
no inferred value. The comparison is performed on the exact rational values
of the persisted numerical parent coefficients, not on traversal-dependent
floating-point reductions.

The target law is uniform over the admissible sign vectors in each stratum,
with independent product composition across strata and observations. The
admissible set is complement-closed because `Delta_h(s_h)=Delta_h(-s_h)`.
Therefore every detector has marginal sign probability `1/2`, while detector
signs within a balanced stratum are generally dependent. Equal positive and
negative detector counts are neither required nor implied.

This conditional law is realized by plan-bound rejection sampling from
uniform independent sign candidates. The exact random-bit generator,
algorithm version, key namespace, seed/key bytes, requested member count, and
positive retry cap are required plan facts. The contract supplies no default
for any of them. Candidate rejections are construction outcomes, not members.
Failure to find an admissible candidate within the cap fails design resolution
closed.

Members are sampled with replacement. Complement pairs are not forced.
Duplicates and complements remain distinct draws and are retained; exact
assignment equality, complement-orbit equality, counts, and design rank are
reported separately. Equal design weights `omega_b=1/B_resolved` are part of
this exact design, not inferred from a filename or count.

## 4. What the ensemble contains

The scientifically readable term selected here is
`source-bearing conditional randomization ensemble`.

"Source-bearing" means that the fixed parent may contain astronomical source
signal, structured residuals, and source-model error. "Conditional" means the
declared parent and operator state are held fixed. "Randomization" identifies
the assignment law, not repeated physical noise. The design intends to
suppress source signal in aggregate, but it does not by construction establish
source-free maps or pixelwise cancellation.

Writing a fixed parent as `x=s+n` for reasoning, the conditional randomization
second moment contains terms from `s s^T`, `n n^T`, and their cross terms after
the sign law and frozen operator act. A compact source can therefore leave a
localized second-moment imprint. A structured scan residual can leave a
coherence-dependent imprint. Removing such morphology is not sufficient to
prove physical-noise fidelity because a learned source model may also absorb
genuine noise modes.

## 5. The initial UNC estimand

Let the all-members-successful ensemble contain admitted realization maps
`M_b`. On the exact common domain where every admitted member provides a valid
finite value, the initial estimator is

```text
V_hat_cond(p) = sum_b omega_b M_b(p)^2,
sum_b omega_b = 1,
omega_b = 1 / B_resolved.
```

The center is the known design target zero. The finite ensemble mean is not
subtracted, and no `B-1` correction applies. Dependence, complement structure,
all distinct counts, design rank, use-specific effective information, and the
uncertainty of `V_hat_cond` or its unavailable state remain explicit.

`V_hat_cond` is a conditional randomization second moment in squared signal
units. It retains source imprint and structured residual content. Pointwise
shape does not make it physical-noise variance or MAP covariance. Unreported
covariance is unknown or unavailable, not zero or independence.

The exact reciprocal product is

```text
W_hat_cond(p) = 1 / V_hat_cond(p)
```

only where the parent is finite and strictly positive. It is an inverse
conditional second-moment scale in inverse squared signal units, not inverse
variance or precision. No floor, cap, epsilon, clipping, or shrinkage is
implicit.

## 6. The initial STD product

The initial standardized product binds the exact immutable normalized
real-observation MAP signal `q_MAP` associated with the same frozen operator
state:

```text
sigma_cond(p) = sqrt(V_hat_cond(p)),
S_cond(p)     = q_MAP(p) / sigma_cond(p).
```

It exists only on the exact compatible intersection where numerator and scale
are valid and the scale is finite and strictly positive. Its unit is exactly
`1`. Its complete claim is only: "MAP signal standardized by the stated
conditional randomization second-moment scale." Numerator/scale dependence is
part of the identity. No Gaussian, Student, z, N-sigma, probability,
detection, completeness, purity, or catalog interpretation follows.

## 7. Immutable successors and externally owned transforms

An externally owned deterministic transformation may be applied to every
compatible realization only under an exact content-bound authority and parity
interface supplied by the process that owns the transformed science product.
NOI does not choose, tune, simplify, relocate, or reinterpret it.

A Wiener transform frozen before realization application is one fixed
owner-transformed route. A Wiener transform learned or updated from `UNC_k`
begins a new immutable transformation, science-product, GEN, and UNC
generation. Per-member learning is a distinct relearned method.

The same discipline applies to FRUIT. Fixed residual or terminal state supports
only uncertainty conditional on that frozen FRUIT state. NOI-informed
continuation creates a successor generation, and per-member replay is a
separate relearned method. Prior NOI products remain dependent inputs rather
than independent validation.

## 8. Analytic predictions of the draft contract

These are mathematical predictions of the stated design and estimators. They
are not reports of implementation conformity or empirical validation.

- `NOI-PRED-001`: Globally complementing an admitted sign assignment preserves
  every stratum imbalance and produces an equally admissible assignment.
- `NOI-PRED-002`: Under the declared uniform complement-closed law, every
  detector has marginal sign probability `1/2` even though network-local signs
  are dependent.
- `NOI-PRED-003`: A one-detector stratum has `Delta_h=1`; it is infeasible for
  every permitted `tau_h<1` and therefore fails design resolution closed.
- `NOI-PRED-004`: Two equal-mass detectors with opposite signs have
  `Delta_h=0` and satisfy every permitted nonnegative tolerance.
- `NOI-PRED-005`: A candidate rejected before admission changes no requested,
  resolved, completed, unique, or UNC-admitted member count.
- `NOI-PRED-006`: Exact duplicate assignments reduce `B_unique` but remain
  distinct with-replacement draws and retain their declared equal weights.
- `NOI-PRED-007`: Complement assignments are distinct full-distribution draws
  but produce the same member outer product under a fixed linear operator.
- `NOI-PRED-008`: Any admitted-member failure makes the whole GEN ensemble
  unavailable for UNC; completed survivors produce no partial estimate.
- `NOI-PRED-009`: `V_hat_cond(p)` is nonnegative wherever it is available and
  is unavailable outside the common all-member domain.
- `NOI-PRED-010`: A positive source-bearing fixed component may contribute a
  source-shaped term to `V_hat_cond`; zero design mean does not remove that
  second-moment imprint.
- `NOI-PRED-011`: `W_hat_cond` is available only on the finite strictly
  positive domain of `V_hat_cond`; zero never becomes an estimated inverse.
- `NOI-PRED-012`: `S_cond` is dimensionless and unavailable wherever the
  numerator/scale join is incompatible or the scale is not finite and positive.
- `NOI-PRED-013`: Increasing member count can improve estimation of the stated
  finite-design target but cannot create exposure or reduce the immutable
  parent map's realized noise as `1/sqrt(B)`.
- `NOI-PRED-014`: A fixed owner-defined linear transformation propagates a
  conditional second moment as `F M F^T`; a per-member learned transformation
  does not obey that fixed-operator identity by implication.
- `NOI-PRED-015`: A pass for one NOI-owned admission profile neither realizes
  the next operation nor changes any producer-owned fact.

## 9. Draft claim boundary

This rationale makes no claim of implementation conformity, representation
fidelity, numerical validation, calibration, physical-noise validity,
covariance completeness, Gaussian significance, achieved performance,
readiness, freeze, production suitability, or production authorization. The
selected ordinary route and all numerical products remain unavailable pending
their exact parent gates, plan realization, later scientific-owner acceptance,
and separately governed downstream evidence.
