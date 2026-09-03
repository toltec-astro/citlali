# SCI-FLT-INF scientific-owner approval for ODQ-009

Decision identity: `SCI-FLT-INF-ODQ-009-APPROVAL v0.1`

Decision date: `2026-08-31`

Scientific owner: G. Wilson

Status: **Option 1 approved; ODQ-009 closed**

## Approved uncertainty policy

Base v0.1 uses a tiered conditional-uncertainty policy. Every matched-filtered
map product must state its uncertainty/covariance availability truthfully, but
a numerical uncertainty product is required only when its scientific
prerequisites are authoritative and available. The filtered signal may remain
valid and publishable when covariance is explicitly unavailable. Missing
uncertainty is never encoded as zero variance, infinite weight, diagonal
covariance, or independence.

The approved primary uncertainty estimand is covariance of the filtered map
conditional on the exact fixed realized method state. It is not posterior-sky
covariance and does not include uncertainty from relearning or selecting the
state unless a later separately contracted method explicitly does so.

## Exact fixed-state covariance identity

Let `L` denote the exact fixed-state response operator whose admitted rows are
the ODQ-008 functionals `L_x`, acting on the exact admitted parent-map vector.
When an authoritative parent covariance `C_parent` is available for that same
parent identity, population, domain, support, and fixed state, the exact
conditional output covariance is

```text
C_cond = L C_parent L^T.
```

Its entries have unit `unit(A_hat)^2`. The covariance product must bind the
exact observation-local or coadd-local parent identity, output and input
domains, admitted support, fixed method state, response operator, population,
rank and null space, regularization, approximations, omitted modes or
correlations, calibration treatment, and lifecycle. Observation and coadd
covariances remain separate identities; neither is inferred from the other.

If `C_parent`, required cross-covariance blocks, or another material premise is
missing or only partially authoritative, the affected covariance entry,
block, projection, or entire product is explicitly limited or unavailable. It
is not filled with zeros and does not establish independence.

## Normalization and marginal variance

If and only if ODQ-004 later selects `Q=C_parent^-1` on the exact admitted
fixed-state domain and the complete generalized-least-squares premises hold,
the ODQ-006 normalization gives

```text
Var[A_hat(x) | fixed state] = D(x)^-1.
```

This is a **marginal conditional variance** at `x`. It does not make `D` a
universal precision product, does not imply that `C_cond` is diagonal, and
does not imply independent filtered-map locations. Under any weaker weighting
identity or incomplete premise, `D` remains the estimator normalization only;
neither `D` nor `D^-1` may be labeled covariance, variance, inverse variance,
precision, weight, or significance by shape, positivity, unit, or historical
usage.

## Covariance representation assignment

A dense covariance matrix is not universally required. The future
implementation-blind author must develop the smallest bounded set of
persistence and representation options needed by the selected science in both
the Scientific Rationale and Contract and the Engineering Conformance
Specification. Both views must use the same option identifiers and state the
same scientific consequences. Candidate forms may include an exact explicit
operator or matrix, an exact structured representation, an exact named
projection for an authorized consumer, a lineage-resolvable representation,
or explicit unavailability.

Each option must state which entries or operations it can recover, whether it
is exact or approximate, its domain and rank/null behavior, its supported
consumers, and its failure and unavailable states. A retained NOI ensemble is
not automatically a covariance representation. The scientific owner must
select or dispose of the authored options before freeze or publication of a
numerical covariance route.

## Frozen-state NOI companion

An optional transformed-NOI companion may be produced only through the exact
frozen SCI-NOI authority. The same exact owner-frozen filter state and
transformation must be applied to every admitted NOI member with the required
operator and product parity. On the common all-member valid domain, the
ordinary frozen-NOI product

```text
V_hat_cond(p) = sum_b omega_b M_b(p)^2
```

retains its frozen-authority identity as a conditional randomization second
moment, with unit `unit(A_hat)^2`. It is not thereby physical-noise variance,
covariance, precision, inverse variance, calibrated significance, or a
substitute for `C_cond`. Its reciprocal, if retained at all, is only an inverse
conditional-second-moment scale.

Fixed-state and relearned members cannot be mixed. A full-procedure empirical
uncertainty in which PSD, support, template, approximation, selection, or
other consequential state is re-estimated belongs to the separate ODQ-010
generation method and population. The absence of a NOI companion does not by
itself invalidate the filtered signal.

## Calibration and total uncertainty

Fixed-state statistical covariance and calibration/nuisance uncertainty are
distinct components. Parent/template calibration dependence remains joint
under ODQ-008: shared factors are not presumed independent, cancelling, or
reduced by averaging. Calibration terms and cross-covariances that are not
authoritative remain unavailable.

A total calibrated uncertainty may be claimed only when every material
statistical, calibration, nuisance, and cross-covariance term is quantified or
explicitly not applicable for the declared estimand. Otherwise total
uncertainty is unavailable even when `C_cond` or a NOI second moment exists.

## Permitted consumer use

For an authorized fixed linear measurement `g^T A_hat`, the conditional
variance is

```text
Var[g^T A_hat | fixed state] = g^T C_cond g
```

only when the representation supplies every required covariance operation or
entry under the same state and population. Marginal variances alone do not
authorize independent-pixel aperture, integrated-flux, source, peak,
detection, or catalog uncertainties. No product selected here is calibrated
significance, a source-analysis result, or posterior-sky covariance.

## Decision consequences

`SCI-FLT-INF-ODQ-009` is closed on this basis. The next owner gate is
`SCI-FLT-INF-ODQ-010`, which must classify the generation and relearning graph
for every consequential state component. The covariance-representation option
set is an explicit future-author assignment with later owner disposition; this
approval does not select one representation or authorize a numerical route.

This decision does not select the ODQ-004 noise/covariance authority, approve
an ODQ-006 approximation envelope, define a full-procedure uncertainty,
authorize source analysis or significance, establish posterior/Wiener
reconstruction, approve the final product bundle, launch Stage B, or alter any
protected SCI-FLT-FIXED authority.
