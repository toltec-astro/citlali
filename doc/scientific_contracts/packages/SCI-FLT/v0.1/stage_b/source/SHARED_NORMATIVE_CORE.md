# SCI-FLT-FIXED v0.1 Shared Normative Scientific Core

Document identity: `SCI-FLT-FIXED-NORMATIVE-CORE v0.1/draft-r0.1`

Status: implementation-blind Stage B scientific-contract draft; scientific-owner review required

Stage B date: `2026-08-30`

## 1. Authority, scope, and normative language

This document is the sole shared normative scientific core for the
SCI-FLT-FIXED v0.1 Stage B draft. It was authored from the exact 17-object
packet bound by `AUTHOR_PACKET_MANIFEST.md` identity
`SCI-FLT-FIXED_AUTHOR_PACKET v0.1/r0.1`. The external SHA-256 of that manifest
is `7f2d03f182258ac9770f7dba869e9ae0b5018efdcdb93b18b299a9b9c6df1e4d`.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
MAY, and OPTIONAL state the strength of a scientific-contract requirement.
Unknown, unavailable, disabled, failed, and zero are distinct typed states.

SCI-FLT is the recovery tranche. SCI-FLT-FIXED is the v0.1 package. No
SCI-FLT-INF contract is defined here.

## 2. Scientific object and estimand

For one exact admitted parent map vector `m`, the scientific object is

```text
y = A m,
A = J_full L_Theta.
```

`L_Theta` is a finite same-grid linear operator completely resolved and frozen
before application to the parent random field. `J_full` selects the exact
scientific output rows whose complete required footprint is admitted and
finite. The output `y` is the transformed parent-map amplitude on that row
domain. There is no additive term.

Fixed convolution is the admitted structured family. Fixed low-pass
convolution is only a qualified subtype of fixed convolution. It is not a
second generic family and it is not implied by a friendly name, a width, or
the word smoothing.

## 3. Parent roles and parent ordering

One transformation binds exactly one immutable parent in exactly one role:

- `FLT-PARENT-MAP-OBS`: one complete base or unfiltered MAP observation bundle;
- `FLT-PARENT-MAP-COADD`: one complete base or unfiltered centered-integer
  common-grid MAP coadd bundle; or
- `FLT-PARENT-JINC-OBS`: one complete atomic JINC observation bundle.

The roles are not interchangeable even when shape, WCS values, or numerical
payloads happen to agree. MAP observation, MAP coadd, and JINC observation
successors remain separately identified. SCI-FLT-FIXED does not coadd and
does not presume that filtering and coaddition commute.

The exact parent binding carries the complete parent package, revision,
product and application generations, role, membership, stable array or group,
quantity, units and originating nominal-beam convention, WCS, frame,
topology, grid, pixel metric, shape, row domain, support, validity and causes,
response and covariance availability, null and additive-reference state,
exposure, causal influence, lifecycle, failure, and provenance facts required
by the applicable MAP or JINC boundary. For JINC, all five required numerical
roles form one atomic parent. A partial bundle is not a parent.

The Stage A boundary records make ordinary numerical MAP and TolTEC JINC
parents unavailable at this draft's launch state. The contract defines their
typed successor routes but does not manufacture numerical parents.

## 4. Requested, effective, resolved, and applied state

The contract distinguishes:

- requested scientific purpose and method;
- effective selection or disablement;
- complete externally resolved immutable plan;
- observation-resolved operator generation;
- exact applied transformation; and
- realized atomic successor product.

Every coefficient, kernel parameter, cutoff or width, WCS fact, normalization,
support rule, transfer qualification, and lifecycle fact is external to the
parent application and fixed before use. No parent data or NOI member may
select, learn, tune, or re-resolve any of these facts.

## 5. Same-grid finite operator

Let `S_in` be the exact finite parent row domain. For output row `p`,

```text
y_p = sum over q in S_in of L[p,q] m_q,  for p in S_out.
```

The operator identity binds all of the following:

- exact input and output row domains;
- identical parent and output WCS, frame, topology, metric, shape, indexing,
  and pixel-area convention, apart from restriction to `S_out`;
- operator family, version, parameter set, complete sampled coefficients, and
  content digest;
- coefficient coordinate domain, units, ordering, numerical representation,
  orientation, handedness, center, extent, even or odd tie convention, phase,
  subpixel convention, and finite offset support;
- normalization and every qualified transfer fact;
- full-footprint row selection and unavailable-row cause rules;
- response, covariance, mode, influence, support, and validity states; and
- requested through realized lifecycle generation, failure, and provenance.

Reprojection, resampling, mosaicking, and deconvolution are not same-grid
SCI-FLT-FIXED methods. FFT, direct, separable, cached, or threaded evaluation
is scientifically immaterial only when it realizes the identical declared
finite operator.

## 6. Fixed convolution and low-pass qualification

For the exact finite offset set `K_Theta`, fixed convolution is

```text
(L_Theta m)_p = sum over r in K_Theta of k_Theta(r) m_(p-r).
```

The sampled kernel, rather than a continuous ideal or family name, constructs
the scientific operator. Signed sum, absolute support, squared support, and
geometric support are distinct quantities.

A low-pass claim is available only if the resolved plan binds the exact
spatial-frequency domain and WCS metric, DC gain, passband, transition region,
stopband or attenuation criterion, phase, isotropic or anisotropic state,
finite-grid and edge limitations, sampled kernel, normalization, and parameter
identity, source, and provenance. If any one of these is unavailable, the
fixed-convolution identity may remain available but the low-pass qualification
is unavailable.

## 7. Full-footprint scientific output domain

The sole v0.1 scientific output-row method is

```text
S_out = {p: for every r in K_Theta,
            p-r lies in S_in,
            m_(p-r) is admitted for this exact FLT use,
            m_(p-r) is available and finite,
            and every required predicate passes}.
```

Rows outside `S_out` are scientifically unavailable, not zero. A stored array
may preserve parent shape and WCS only if every unavailable row carries its
typed cause and remains outside the scientific vector.

Base v0.1 admits no boundary extension, periodic wrapping, truncated
convolution, support-conditioned renormalization, inpainting, reflection,
clamping, mirroring, edge completion, padding-based admission, or replacement
of a missing or non-finite value. Each would require a separately named and
versioned successor method.

Numerical computability, complete kernel support, FLT input admission,
FLT-local output validity, confidence, and downstream eligibility are
distinct states.

## 8. Quantity, units, beam, response, and modes

`FLT-SIG` is the transformed parent-map quantity. Its units are derived from
the exact parent units and operator coefficient units. For a MAP parent, the
originating `mJy/beam` nominal-beam and calibration lineage remain identified;
the output is not relabeled as `mJy/filtered-beam`.

Kernel normalization is an explicit convention, such as unit signed sum or DC
gain, unit peak, unit angular integral, unit L2 norm, or another fully stated
convention. No convention is inferred from a name. Numerical unit preservation
does not establish calibration, point-source peak preservation, integrated
flux, extended-source fidelity, target PSF, or a new beam convention.

For an available exact compatible parent response `R_parent`,

```text
R_out = J_full L_Theta R_parent.
```

The response uses the identical operator, centering, orientation, phase,
support, edge, missing-data, row-domain, and normalization rules as the signal.
If no compatible parent response exists, `FLT-RSP` is unavailable. The kernel
alone is not the complete source response or PSF.

`FLT-TRANSFER` records the exact local sampled spatial or Fourier transfer on
the declared finite-grid domain where scientifically defined, or typed
unavailability. `FLT-MODE` records exact null modes, invariant modes,
attenuation statements or bounds, and sampled phase where defined. Local
operator modes are not automatically upstream sky-to-output modes.

`FLT-INFLUENCE` is the exact coefficient and support relation between parent
and output rows. It is not physical exposure. Filtering creates no new
physical exposure claim.

## 9. Deterministic covariance and NOI boundary

For an available compatible declared parent covariance `C_parent`,

```text
C_out = J_full L_Theta C_parent transpose(L_Theta) transpose(J_full).
```

The deterministic covariance state is explicitly one of complete,
diagonal-input propagated, structured or operator, partial, marginal, or
unavailable. Every available representation states its domain, rank, null
space, omitted terms, and supported operations as applicable. Unknown cross
terms remain unknown and are not zero.

For an explicitly diagonal independent parent covariance with marginal
variances `V_j`,

```text
Var(y_i) = sum_j k[i,j]^2 V_j,
Cov(y_i,y_l) = sum_j k[i,j] k[l,j] V_j.
```

The output generally has off-diagonal covariance. A marginal variance plane is
not full covariance and does not authorize independent-pixel multi-pixel
inference.

SCI-NOI, not SCI-FLT-FIXED, owns empirical uncertainty, empirical covariance,
conditional inverse scale, standardized signal, and significance inference.
For every compatible admitted NOI member `M_b`, exact fixed-state parity is

```text
M_b_out = J_full L_Theta M_b.
```

The real parent and every member use the identical operator, parameter set,
grid, support, edge rule, row domain, and lifecycle generation. Filtering a
variance, standard deviation, precision, reciprocal, weight, standardized
map, or significance field is not this operation. Per-member selection or
re-resolution is a different inference-bearing method and cannot enter the
fixed-state ensemble.

Parameter, kernel, cutoff, beam, WCS, selection, model, and calibration
uncertainty remain separate from covariance conditional on fixed `L_Theta`.

## 10. Atomic product and lifecycle

Every realized bundle contains these role records, including explicit
unavailable states where allowed:

- `FLT-PARENT`;
- `FLT-PLAN`;
- `FLT-OPERATOR`;
- `FLT-SIG`;
- `FLT-UNIT-BEAM`;
- `FLT-TRANSFER`;
- `FLT-RSP`;
- `FLT-MODE`;
- `FLT-INFLUENCE`;
- `FLT-SUP`;
- `FLT-VALID`;
- `FLT-COV-FORMAL`;
- optional, separately owned `NOI-UNC[FLT-SIG]`; and
- `FLT-LINEAGE`.

A missing required record is not an atomic bundle. An honestly unavailable
response or covariance record may satisfy the base role only where the role
permits absence; it cannot satisfy a response-qualified or
covariance-qualified request.

Lifecycle states are `not_requested`, `requested`, `effective`, `disabled`,
`unavailable`, `resolved`, `applied`, `failed`, `realized_identity`,
`realized_zero`, `realized`, and `superseded`. Only a complete atomic bundle
with a successful publication decision is realized. Disabled makes no
product. Identity and zero operators produce real separately parented products
after resolution and application. Failure of a required transformation or
bundle step propagates and yields no complete product.

Any change to parent, request or effective purpose, operator, kernel,
parameter, transfer qualification, normalization, WCS, grid, row domain,
support, validity, response or covariance role, lifecycle, or failure policy
creates a new immutable transformation and product generation. A later NOI
attachment is a separate immutable companion and does not mutate FLT or the
parent.

## 11. FLT policy and VAL boundary

The draft input-admission policy is `SCI-FLT-FIXED:input_admission@1`. It is
requested only by an accepted plan explicitly requesting SCI-FLT-FIXED. It is
applicable only to one supported parent role and the strict-linear same-grid
full-footprint method. Eligibility requires every exact parent, plan,
operator, unit and beam, WCS and grid, support and validity, response and
covariance availability, lifecycle, failure, and provenance fact required by
the applicable boundary. Missing, unavailable, or conflicting required facts
fail the affected route closed and preserve causes.

The draft output-publication policy is
`SCI-FLT-FIXED:output_publication@1`. It applies only to one realized atomic
SCI-FLT-FIXED bundle. Eligibility requires every role and honest companion
state in Section 10. Disabled, partial, placeholder, or inferred bundles are
not eligible. Identity and zero products remain eligible only through their
realized lifecycle states.

SCI-VAL may bind and evaluate an immutable owner-approved successor of these
profiles. VAL does not author producer facts, FLT policy, arithmetic, or
scientific claims. These draft profiles are not registered and do not create
a numerical route.

## 12. Consumer and ownership boundaries

MAP and JINC own the parent estimand and parent claims. CAL owns absolute
calibration, passband and color correction, and calibration covariance.
SCI-FLT-FIXED owns the exact local transformation, transformed signal, output
unit derivation, composed-response state, local transfer and modes, influence,
support and validity, deterministic covariance state, lifecycle, failure, and
provenance. SCI-NOI owns empirical uncertainty inference and applies but does
not choose the exact FLT transformation. SCI-BEAM and future source or mode
contracts own physical source, beam, Pointing, and OOF interpretations.
SCI-FRUIT owns iterative feedback science.

The availability of an SCI-FLT-FIXED product authorizes no generic Beammap,
Pointing, OOF, source-fit, catalog, NOI, or FRUIT use. Each consumer owns an
exact use policy.

## 13. Exclusions and nonclaims

This contract excludes affine offsets, background or template subtraction,
additive correction, reprojection, resampling, mosaicking, deconvolution,
boundary extension, truncated or renormalized convolution, missing-value
replacement, data-derived kernel or cutoff selection, automatic method
selection, Wiener transformation, matched or generalized least-squares
template-amplitude estimation, source-learned operation, data-derived mode
selection or map-domain destriping, per-member relearning, FLT coaddition,
source or mode interpretation, FRUIT recurrence, and RTC timestream filtering.

This Stage B draft makes no implementation-conformity, algorithm-change,
validation, calibration, achieved-response, achieved-covariance, numerical
adequacy, performance, readiness, scientific-freeze, production, or Unity
claim.

## 14. Normative requirements

### SCI-FLT-FIXED-REQ-001 - Package identity

The package SHALL be identified as SCI-FLT-FIXED v0.1 within the SCI-FLT
tranche and SHALL NOT be presented as SCI-FLT-INF or as a generic filter
contract.

### SCI-FLT-FIXED-REQ-002 - Strict linearity

The transformation MUST be exactly `y = J_full L_Theta m` with no additive
term. Any offset, background, template subtraction, or additive correction
MUST be rejected as outside v0.1.

### SCI-FLT-FIXED-REQ-003 - Exact parent role

Each transformation MUST bind exactly one complete immutable
`FLT-PARENT-MAP-OBS`, `FLT-PARENT-MAP-COADD`, or `FLT-PARENT-JINC-OBS` parent.
No role substitution by shape, WCS, name, or payload similarity is permitted.

### SCI-FLT-FIXED-REQ-004 - Complete parent identity

The parent binding MUST include all applicable package, revision, product,
application, observation or membership, quantity, unit, nominal-beam, WCS,
grid, row-domain, support, validity, response, covariance, null, exposure,
lifecycle, failure, and provenance facts stated in Section 3. A partial JINC
five-role bundle MUST be rejected.

### SCI-FLT-FIXED-REQ-005 - Honest upstream availability

An unavailable MAP or JINC numerical parent MUST remain unavailable. Algebra,
defaults, approximate identity, or a finite payload MUST NOT manufacture a
parent or successor.

### SCI-FLT-FIXED-REQ-006 - Parent ordering

Filtering an observation and filtering a coadd MUST create distinct successor
identities. SCI-FLT-FIXED MUST NOT coadd or claim filter/coadd commutation
without a separately approved exact bounded relation.

### SCI-FLT-FIXED-REQ-007 - External fixed resolution

Every operator coefficient, parameter, grid fact, normalization, support rule,
and transfer qualification MUST be resolved externally and frozen before
application to the parent or any NOI member.

### SCI-FLT-FIXED-REQ-008 - Same-grid boundary

The output MUST preserve the exact parent WCS, frame, topology, metric, shape,
pixel indexing, and pixel-area convention, apart from scientific-row
restriction. Reprojection, resampling, and approximate-WCS joins MUST be
rejected.

### SCI-FLT-FIXED-REQ-009 - Complete operator identity

The applied operator MUST bind every identity and discretization fact listed
in Section 5, including complete sampled coefficients and their content
digest. A friendly kernel name or continuous ideal is insufficient.

### SCI-FLT-FIXED-REQ-010 - Fixed convolution construction

A `FLT-FIXED-CONV` method MUST construct the complete finite operator from one
exact finite sampled kernel and its declared offset set, orientation, center,
phase, normalization, and coefficient representation.

### SCI-FLT-FIXED-REQ-011 - Low-pass qualification

A `FLT-FIXED-CONV-LOWPASS` claim MUST bind every transfer fact listed in
Section 6. If any is missing, the low-pass claim MUST be unavailable even when
the fixed-convolution identity remains available.

### SCI-FLT-FIXED-REQ-012 - Full-footprint admission

`S_out` MUST contain exactly those rows for which every required kernel
location is in the parent domain, admitted for the exact FLT use, available,
finite, and passing every required predicate.

### SCI-FLT-FIXED-REQ-013 - Unavailable row semantics

A row outside `S_out` MUST be unavailable rather than zero. Parent-shaped
storage MAY be retained only with explicit unavailable-row state and causes.

### SCI-FLT-FIXED-REQ-014 - Sole edge method

Boundary extension, wrapping, truncation, support renormalization, inpainting,
reflection, clamping, mirroring, padding-based admission, edge completion, and
missing or non-finite replacement MUST NOT be used by v0.1.

### SCI-FLT-FIXED-REQ-015 - Output quantity and units

`FLT-SIG` MUST be identified as transformed parent-map amplitude on `S_out`,
with output units derived from exact parent and coefficient units. It MUST NOT
be relabeled as flux, fitted amplitude, or a response-corrected quantity.

### SCI-FLT-FIXED-REQ-016 - Beam and calibration boundary

The originating nominal-beam and calibration lineage MUST be retained. No new
filtered-beam, target-PSF, absolute-calibration, peak-preservation,
integrated-flux, or extended-source claim may be inferred from normalization
or units.

### SCI-FLT-FIXED-REQ-017 - Response composition

When an exact compatible parent response is available, `FLT-RSP` MUST equal
`J_full L_Theta R_parent` using the identical applied transformation. When it
is not available or compatible, `FLT-RSP` MUST be typed unavailable.

### SCI-FLT-FIXED-REQ-018 - Transfer and mode state

`FLT-TRANSFER` and `FLT-MODE` MUST publish the exact local finite-grid transfer,
null, invariant, attenuation, and phase facts where defined, or honest typed
unavailability. They MUST NOT be promoted to unproved whole-chain claims.

### SCI-FLT-FIXED-REQ-019 - Influence is not exposure

`FLT-INFLUENCE` MUST describe the exact parent-to-output coefficient relation
and MUST NOT be labeled or interpreted as physical exposure.

### SCI-FLT-FIXED-REQ-020 - Distinct support and validity states

Numerical computability, complete footprint, FLT input admission, FLT-local
validity, parent validity, confidence, and downstream eligibility MUST remain
distinct with exact causes.

### SCI-FLT-FIXED-REQ-021 - Deterministic covariance propagation

An available compatible parent covariance MUST be propagated by
`C_out = J_full L_Theta C_parent transpose(L_Theta) transpose(J_full)` on the
exact scientific row domain.

### SCI-FLT-FIXED-REQ-022 - Covariance state honesty

`FLT-COV-FORMAL` MUST distinguish complete, diagonal-input propagated,
structured or operator, partial, marginal, and unavailable states. Unknown
cross terms MUST NOT be set to zero or interpreted as independence.

### SCI-FLT-FIXED-REQ-023 - Induced covariance

For an explicitly diagonal independent parent covariance, the exact output
off-diagonal terms MUST be retained in any complete covariance claim. A
marginal plane MUST NOT be called full covariance.

### SCI-FLT-FIXED-REQ-024 - Empirical uncertainty ownership

SCI-FLT-FIXED MUST NOT infer empirical uncertainty, empirical covariance,
conditional inverse scale, standardized signal, or significance. Such an
attachment remains SCI-NOI-owned and separate from `FLT-COV-FORMAL`.

### SCI-FLT-FIXED-REQ-025 - Fixed-state NOI parity

Every admitted NOI member for the exact transformed product MUST receive the
identical `J_full L_Theta`, parameters, grid, support, edge rule, row domain,
and lifecycle generation used for the real parent.

### SCI-FLT-FIXED-REQ-026 - Relearning rejection

Any per-member parameter selection, operator resolution, or relearning MUST be
rejected from the fixed-state ensemble and routed to a separately named
inference-bearing method.

### SCI-FLT-FIXED-REQ-027 - Atomic product roles

A realized product MUST contain every required role record in Section 10,
including explicit unavailable companion records where the base role permits
absence. A partial bundle MUST NOT be published.

### SCI-FLT-FIXED-REQ-028 - Lifecycle

The lifecycle MUST distinguish all states in Section 10 and MUST preserve
exact causes, failures, and immutable generation bindings.

### SCI-FLT-FIXED-REQ-029 - Disabled, identity, and zero states

Disabled MUST produce no product. Requested and applied identity and zero
operators MUST produce distinct realized products and MUST NOT be represented
as disabled or unavailable.

### SCI-FLT-FIXED-REQ-030 - Generation identity

Any change listed in Section 10 MUST create a new immutable transformation and
product generation. An NOI companion MUST NOT mutate the FLT product.

### SCI-FLT-FIXED-REQ-031 - Failure and fallback

Missing, conflicting, or unavailable state required by the requested identity
MUST fail the affected route closed with exact causes. No silent fallback,
default, or same-name substitution may retain that requested identity.

### SCI-FLT-FIXED-REQ-032 - Input admission policy

The `SCI-FLT-FIXED:input_admission@1` draft semantics in Section 11 MUST govern
request, applicability, eligibility, unavailable decision, exclusions, and
fail-closed action until superseded by an owner-approved immutable profile.

### SCI-FLT-FIXED-REQ-033 - Output publication policy

The `SCI-FLT-FIXED:output_publication@1` draft semantics in Section 11 MUST
govern publication of the exact atomic bundle. Honest absence is permitted
only for a role that explicitly allows it.

### SCI-FLT-FIXED-REQ-034 - VAL boundary

SCI-VAL MAY bind and evaluate an owner-approved immutable policy but MUST NOT
author FLT facts or policy, perform the transformation, or convert this draft
into a registered or numerical route.

### SCI-FLT-FIXED-REQ-035 - Consumer boundary

Product availability MUST NOT authorize generic downstream use. Beammap,
Pointing, OOF, source-fit, catalog, NOI, and FRUIT consumers MUST supply their
own exact owner-approved use policies.

### SCI-FLT-FIXED-REQ-036 - Excluded methods and nonclaims

Every method in Section 13 MUST remain outside v0.1. The package MUST preserve
all Stage B nonclaims in Section 13.

## 15. Falsifiable predictions

Each prediction is conditional on an exact admitted parent and fully resolved
operator unless the prediction explicitly tests unavailability.

### SCI-FLT-FIXED-PRED-001 - Identity operator

For the exact identity operator, `FLT-SIG` equals the parent signal on `S_out`.
An available compatible response and covariance are unchanged on that domain.
The result is `realized_identity`, not disabled and not an unparented copy.

### SCI-FLT-FIXED-PRED-002 - Zero operator

For the exact zero operator, every admitted `FLT-SIG` value is zero and any
available compatible transformed response and covariance are zero on `S_out`.
Unavailable parent response or covariance remains typed unavailable. The
product is `realized_zero`, not disabled, invalid, or evidence of precision.

### SCI-FLT-FIXED-PRED-003 - Input scaling

For any finite scalar `a`, applying the same fixed operator and admissions to
`a m` produces `a y`. Failure of this relation falsifies strict linearity.

### SCI-FLT-FIXED-PRED-004 - Constant input and DC gain

For constant admitted input `m_q = c`, every full-footprint convolution row is
`c` times the exact signed kernel sum, equivalently `c` times the declared DC
gain. Constant preservation occurs exactly when that gain is one; no other
normalization implies it.

### SCI-FLT-FIXED-PRED-005 - Impulse response

An admitted single-pixel unit impulse produces the exact sampled kernel shifted
according to the declared center, orientation, handedness, phase, and indexing,
then restricted by `S_out`. Any implicit recentering, interpolation, reversal,
or periodic copy falsifies the operator identity.

### SCI-FLT-FIXED-PRED-006 - Parent-response composition

Applying the transformation to an exact compatible parent unit-source response
produces exactly `J_full L_Theta R_parent`. A kernel-only response or a response
using different edge, phase, or normalization rules fails this prediction.

### SCI-FLT-FIXED-PRED-007 - Signed kernel

A signed kernel follows its exact signed coefficients, including cancellation.
Its geometric, signed, absolute, and squared support summaries remain unequal
when the coefficients make them unequal; none may substitute for another.

### SCI-FLT-FIXED-PRED-008 - Zero-sum kernel

For a complete constant input and exact zero signed-sum kernel, every admitted
output row is zero. This nulling does not by itself make the row unavailable,
uncertain, source-free, or statistically significant.

### SCI-FLT-FIXED-PRED-009 - Full-footprint boundary

For every output row whose required footprint is complete, admitted, finite,
and predicate-passing, the row is in `S_out`. Removing, invalidating, or making
non-finite any one required parent location removes every dependent output row
from `S_out` with an exact cause.

### SCI-FLT-FIXED-PRED-010 - Deferred edge methods

Any result that uses extension, wrapping, truncation, local support
renormalization, inpainting, reflection, clamp, mirror, padding-based admission,
edge completion, or value replacement is rejected as a v0.1 product even when
it is finite or preserves a constant.

### SCI-FLT-FIXED-PRED-011 - Missing and non-finite input

No output row depending on a missing, unavailable, non-admitted, or non-finite
required parent value belongs to `S_out`. Such a row is unavailable rather than
zero and records the applicable cause.

### SCI-FLT-FIXED-PRED-012 - Complete covariance transform

For an exact available compatible `C_parent`, the published complete covariance
equals `J_full L_Theta C_parent transpose(L_Theta) transpose(J_full)` on the
declared row ordering. A different domain, ordering, or omitted required term
fails the complete-covariance claim.

### SCI-FLT-FIXED-PRED-013 - Off-diagonal covariance from diagonal input

For explicitly independent parent pixels, overlapping nonzero output stencils
produce the exact cross term `sum_j k[i,j] k[l,j] V_j`. Publishing only
marginals cannot pass a full-covariance check when this term is nonzero.

### SCI-FLT-FIXED-PRED-014 - Unavailable parent companion

When the parent response or required covariance is unavailable, the
corresponding FLT companion is unavailable. A zero array, diagonal guess,
weight, denominator, or kernel-only surrogate fails the prediction.

### SCI-FLT-FIXED-PRED-015 - WCS or grid mismatch

Any mismatch in parent and output WCS, frame, topology, metric, shape, pixel
indexing, or pixel-area convention makes the same-grid route unavailable.
Approximate equality or successful numerical resampling does not pass.

### SCI-FLT-FIXED-PRED-016 - Observation and coadd identity

A filtered observation, filtered MAP coadd, and filtered JINC observation have
distinct parent and successor identities even if their arrays are numerically
equal. No coadd of filtered observations is an SCI-FLT-FIXED v0.1 product.

### SCI-FLT-FIXED-PRED-017 - Exact NOI parity

For each compatible admitted NOI member, recomputing with the exact signal
operator produces `J_full L_Theta M_b` on the identical row domain. Any
different coefficient, parameter, grid, support, edge rule, generation, or row
selection makes the empirical-uncertainty route unavailable.

### SCI-FLT-FIXED-PRED-018 - Per-member re-resolution

If any NOI member selects or re-resolves a kernel, cutoff, support, threshold,
edge state, or other parameter, that member is rejected from the fixed-state
ensemble rather than mixed with it.

### SCI-FLT-FIXED-PRED-019 - Disabled, identity, zero, and failure

A disabled route emits no FLT product; identity emits a separately parented
`realized_identity` product; zero emits a separately parented `realized_zero`
product; and a required failure emits no complete product and propagates its
cause. Collapsing any pair of these states fails the lifecycle contract.

### SCI-FLT-FIXED-PRED-020 - Upstream unavailable parent

At the launch state recorded by the admitted boundaries, an attempted ordinary
numerical MAP or TolTEC JINC transformation remains unavailable unless a
separately authorized upstream successor supplies every named gate. The FLT
contract alone cannot make that route numerical.

### SCI-FLT-FIXED-PRED-021 - Low-pass claim completeness

Removing any required low-pass transfer fact from an otherwise complete fixed
convolution makes only the low-pass qualification unavailable. Retaining a
low-pass label with an incomplete transfer specification fails the contract.
