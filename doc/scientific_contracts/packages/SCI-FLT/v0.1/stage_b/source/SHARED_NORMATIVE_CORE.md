# SCI-FLT-FIXED v0.1 Shared Normative Scientific Core

Document identity: `SCI-FLT-FIXED-NORMATIVE-CORE v0.1/freeze-candidate`

Status: implementation-blind conditional scientific-owner freeze candidate; owner signature required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

## 1. Authority, scope, and normative language

This document is the sole shared normative scientific core for the
SCI-FLT-FIXED v0.1 conditional freeze candidate. It incorporates the exact
r0.2 formal-closure directive, the exact r0.3 final targeted scientific-
closure and source-preflight directive, the exact r0.4 final formal-closure
and freeze-preflight directive, and the exact final micro-repair and
conditional-freeze directive bound by the Stage B build record.
It preserves the scientific authority of the exact 17-object
packet bound by `AUTHOR_PACKET_MANIFEST.md` identity
`SCI-FLT-FIXED_AUTHOR_PACKET v0.1/r0.1`. The external SHA-256 of that manifest
is `7f2d03f182258ac9770f7dba869e9ae0b5018efdcdb93b18b299a9b9c6df1e4d`.

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
MAY, and OPTIONAL state the strength of a scientific-contract requirement.
Unknown, unavailable, disabled, failed, and zero are distinct typed states.

SCI-FLT is the recovery tranche. SCI-FLT-FIXED is the v0.1 package. No
SCI-FLT-INF contract is defined here.

## 2. Scientific object and estimand

Define the exact parent fact domain and the scientific numerical signal domain
separately:

```text
S_parent_fact
  = exact parent row-identity and fact domain;

D_m
  = {q in S_parent_fact:
       an available finite real signal payload exists at q};

m : D_m -> R.
```

A missing, unavailable, or non-finite stored payload is a typed row fact in
`S_parent_fact`; it is not an element of the scientific numerical function
`m`. Parent-shaped storage is a representation containing payload and state
records, not the scientific vector domain. The transformation MUST establish
`q in D_m` before evaluating `m_q`.

For one exact admitted parent, the scientific object is

```text
J_full = J(plan,
           immutable S_parent_fact and row membership,
           FLT input admission,
           typed availability and finiteness facts defining D_m,
           support,
           required predicates).
```

```text
A_Theta,J = J_full L_Theta,
y = A_Theta,J m,
y : S_out -> R.
```

The base scalar field is real on its typed function domains:

```text
m : D_m -> R; y : S_out -> R;
k_Theta(r), L_Theta, and A_Theta,J are real-valued.
```

Every admitted sampled coefficient is finite, real, unit-typed, present in one
canonical exact representation, and content-bound before application. A
missing, non-finite, complex, unrepresentable, or conflicting coefficient
makes plan resolution unavailable. The prospective numerical-comparison
policy cannot repair or admit such a coefficient.

The only numerically admitted base family is `FLT-FIXED-CONV`. `L_Theta` is
the complete matrix representation of one exact realized sampled convolution;
it is not a separately selectable arbitrary dense linear-operator family.
`FLT-FIXED-CONV-LOWPASS` is only a qualified subtype of
`FLT-FIXED-CONV`.

`L_Theta` and the externally resolved plan are complete and frozen before
application to the parent random field. Parent payload amplitudes do not
select, learn, tune, or alter any plan or operator fact. `J_full` is resolved
once from only declared immutable parent identity and row membership, exact
parent-row admission, declared typed row-state facts defining `D_m`, support,
and exact required predicates. Reading those row-state facts is structural
screening, not evaluation of `m_q`, convolution arithmetic, or permission to
tune the plan. The resulting `A_Theta,J` is
the complete applied operator. The output `y` is transformed parent-map
amplitude on the selected row domain. There is no additive term.

Strict linearity is conditional on the exact frozen parent membership and
`J_full`. It is not a global-linearity claim across parents whose domain,
support, admission, availability, finiteness, or validity differs. Response,
covariance, noise realizations, and NOI members receive the identical frozen
`A_Theta,J`; `J_full` is never re-resolved for a response perturbation,
covariance draw, noise realization, or fixed-state NOI member. A member that
cannot supply one frozen required footprint is unavailable on the affected row
rather than receiving another selector. Selection and support uncertainty are
excluded from conditional linear propagation unless separately supplied as a
typed uncertainty object.

Fixed convolution is the admitted structured family. Fixed low-pass
convolution is only a qualified subtype of fixed convolution. It is not a
second generic family and it is not implied by a friendly name, a width, or
the word smoothing.

## 3. Parent roles and parent ordering

One transformation binds exactly one immutable parent in exactly one role.
The exact signal-role disposition is:

```text
Parent role             Exact m used by FLT-SIG
FLT-PARENT-MAP-OBS      base/unfiltered MAP observation signal
FLT-PARENT-MAP-COADD    base/unfiltered MAP coadd signal
FLT-PARENT-JINC-OBS     normalized jinc_map on admitted local support
```

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
roles form one atomic parent:

```text
jinc_signal_numerator
jinc_signed_normalization
jinc_quadratic_accumulator
jinc_map with local support/validity
jinc_coefficient_squared_time
```

Only `jinc_map` is the JINC signal vector `m`. The numerator, signed
normalization, quadratic accumulator, and coefficient-squared temporal
accounting are diagnostics or parent facts, not alternate FLT signals.

Only the selected signal role receives `L_Theta` as `FLT-SIG`. Parent response
receives its separately typed response composition and parent covariance its
separately typed covariance composition. Support and validity are predicates
and records, not convolved signals. Exposure is lineage and is not convolved
into physical exposure. Normalization and coefficient diagnostics remain
parent facts. No other map-shaped role is transformed without a separately
named method. A partial five-role JINC bundle is not a parent.

The Stage A boundary records make ordinary numerical MAP and TolTEC JINC
parents unavailable at this draft's launch state. The contract defines their
typed successor routes but does not manufacture numerical parents.

## 4. Requested, effective, resolved, and applied state

The contract distinguishes:

- requested scientific purpose and method;
- effective selection or disablement;
- complete externally resolved immutable plan;
- observation-resolved operator generation;
- exact frozen row selector and applied transformation;
- `complete_publication_disposition_candidate` with its exact variant;
- publication decision; and
- realized atomic successor product.

Every coefficient, kernel parameter, cutoff or width, WCS fact, normalization,
support rule, transfer qualification, threshold, and lifecycle fact is
resolved before application under one exact FLT-owned plan and fixed before
use. "Externally resolved" describes timing and separation from payload
arithmetic; it does not transfer FLT policy or transformation ownership to an
unnamed producer. Parent payload amplitudes MUST NOT select, learn, tune, or
alter a kernel coefficient, cutoff or width, normalization, transfer
qualification, edge method, support rule, threshold, or other plan fact. No
amplitude-thresholded, source-sensitive, spectral-selection, or otherwise
data-derived method is admitted. `J_full` nevertheless MUST use the declared
immutable parent facts named in Section 2. Response perturbations, covariance
draws, noise realizations, NOI members, and other compatible companions reuse
that resolved selector and MUST NOT re-resolve it from their own values.

## 5. Same-grid finite operator and identity layers

The earlier undifferentiated `S_in` terminology is replaced by
`S_parent_fact` for typed parent facts and `D_m` for the numerical function
domain. For output row `p`,

```text
y_p = sum over q in D_m of L[p,q] m_q,  for p in S_out,
y : S_out -> R.
```

Only entries required by the exact scientific operator may contribute. The
operator MUST establish `q in D_m` before evaluating `m_q`.

One SCI-FLT-FIXED product applies one exact resolved sampled convolution. A
plan may supply a final kernel constructed elsewhere, but the FLT product
makes no claim about intermediate transformations. Ordered composition,
multiple convolution application, and implicit operator reordering are not
admitted in base v0.1. In particular, no equality between a sequence with
intermediate selectors and one final selected convolution is presumed.

The `FLT-OPERATOR` record contains two exact identity layers.

The scientific operator identity binds all of the following:

- exact input `D_m` and output `S_out` scientific domains and their relation to
  `S_parent_fact`;
- identical parent and output WCS, frame, topology, metric, shape, indexing,
  and pixel-area convention, apart from restriction to `S_out`;
- operator family, version, parameter set, canonical coordinate-domain
  offset-to-coefficient relation, exact coefficient values and units, and
  scientific-operator digest;
- `K_geom_science`, `K_nonzero`, `K_req`, coefficient coordinate domain,
  orientation, handedness, center, extent, even or odd tie convention, phase,
  subpixel convention, coordinate-domain method, and finite scientific support
  sets;
- normalization and every qualified transfer fact;
- full-footprint row selection and unavailable-row cause rules;
- response, covariance, mode, influence, support, and validity states; and
- requested through realized lifecycle generation, failure, and provenance.

Its digest MUST be independent of dense, sparse, cropped, padded, compressed,
container, field-order, byte-order, or other serialization choices.

The separate representation identity binds `K_store`, dense or sparse and
cropped or padded encoding, field ordering, byte serialization,
compression/container representation, representation digest, and
representation generation. A representation-only change MAY create a new
representation artifact or representation generation, but it MUST retain the
same scientific operator identity, FLT product identity, and scientific
transformation and product generation. It MUST NOT alter `S_out`, arithmetic,
response, covariance, influence, lifecycle, or any scientific claim. A change
to the canonical scientific coefficient map or any other scientific operator
fact creates a new scientific transformation and product generation.

Reprojection, resampling, mosaicking, and deconvolution are not same-grid
SCI-FLT-FIXED methods. FFT, direct, separable, cached, or threaded evaluation
is scientifically immaterial only when it realizes the identical declared
finite operator under a preregistered numerical-comparison policy.

## 6. Fixed convolution and low-pass qualification

For the ordinary exact finite offset representation, fixed convolution is

```text
(L_Theta m)_p = sum over r in K_nonzero of k_Theta(r) m_(p-r)
              = sum over r in K_req of k_Theta(r) m_(p-r).
```

The sampled kernel, rather than a continuous ideal or family name, constructs
the scientific operator. The scientific owner adopts this exact disposition:

- `K_geom_science`: exact scientific geometric footprint, independent of
  serialization;
- `K_store`: storage or serialization footprint, scientifically
  nonauthoritative;
- `K_nonzero`: offsets whose canonical coefficient is exactly nonzero, with
  exact zero decided only from that canonical representation; and
- `K_req = K_nonzero` for the ordinary fixed-convolution method.

`K_geom_science` is a representation-invariant geometric description, not the
arithmetic dependency set. `K_store` is serialization only. An exact-zero
coefficient contributes no arithmetic term, requires no parent payload,
creates no influence or covariance contribution, cannot cause row exclusion,
and is never classified through a floating threshold. The ordinary method
MUST NOT evaluate or dereference a missing, unavailable, or non-finite parent
payload at an exact-zero coefficient.

Dense, sparse, cropped, or zero-padded storage of one scientific kernel MUST
NOT change `K_geom_science`, `K_nonzero`, `K_req`, `S_out`, response,
covariance, scientific operator identity, FLT product identity, or scientific
generation. Such encodings have distinct representation identities when their
bound serialization facts differ. A floating tolerance MUST NOT classify
exact zero. A zero-valued offset may enter `K_req` only under a separately
named method with a scientific support reason independent of storage layout.
Geometric, storage, nonzero, required-dependency, signed, absolute, and
squared support remain distinct objects.

The identity convolution has `K_req = {0}`. Its scientific row domain is the
exact admitted finite parent-signal row domain. A nonzero convolution is
defined exactly by `K_nonzero` not being empty.

For the exact zero operator,

```text
K_nonzero_zero = empty set;
K_req_zero     = empty set;

S_out_zero
  = exact admitted finite parent-signal row domain
    under the zero-operator request and predicates;

y_p = 0,  for p in S_out_zero.
```

`S_out_zero` is constructed independently from the empty arithmetic and
required-dependency sets. The zero operator does not describe a parent row
domain as a `K_req` domain and does not receive arbitrary storage rows through
a vacuous universal predicate. It retains exact parent identity and row
support, has local fixed-state derivative zero, and has parent-payload
conditional covariance contribution zero. A complete source-domain response
or systematic uncertainty may remain typed unavailable.

A convolution declares one exact coordinate-domain method: pixel-index domain,
affine tangent-plane angular domain, or another exact identified domain. A
fixed shift-invariant pixel kernel is not one spatially uniform angular kernel
unless that relation is established over the declared WCS.

An angular-frequency low-pass claim is available only when the resolved plan
binds either one exact affine or constant tangent-plane pixel metric, or
another exact method proving the stated angular transfer. Every qualified
plan content-binds all of the following:

- the exact discrete or Fourier transform sign convention and normalization;
- coordinate and spatial-frequency units;
- zero-frequency origin and ordering;
- positive-frequency, negative-frequency, and Nyquist-frequency treatment;
- the exact frequency sample grid;
- whether the declared response is complex `H`, amplitude `|H|`, power
  `|H|^2`, or another exact quantity;
- linear-ratio or decibel attenuation;
- passband, transition, and stopband region geometry;
- complex-phase branch or unwrapping convention;
- anisotropy convention and WCS metric relation; and
- DC gain, finite-grid and edge limitations, sampled kernel, parameter
  identity, source, and provenance.

For a pixel-index kernel an allowed reference relation is

```text
H(nu) = sum over r of k(r) exp[-2 pi i nu dot r],
```

but the exact adopted convention MUST be content-bound and MUST NOT be
inferred from this example. If any required fact is unavailable,
fixed-convolution identity may remain available but low-pass qualification is
unavailable.

`H(nu)` is a possibly complex frequency-domain representation of the real map
operator. It does not authorize a complex-valued `FLT-SIG`. A future complex
signal method requires separately named authority for complex signal units,
conjugation conventions, Hermitian covariance, and `A C A^dagger`
propagation.

The sampled-kernel or interior translation-invariant transfer is distinct from
the complete finite row-restricted operator `A_Theta,J`. Base v0.1 makes no
global Fourier-transfer claim for that complete finite operator unless an
exact theorem bound to its finite domain and selector establishes one.

## 7. Full-footprint scientific output domain

The sole v0.1 scientific output-row method is

```text
S_out = {p: for every r in K_req,
            p-r is in S_parent_fact,
            p-r is admitted for this exact FLT use,
            p-r is in D_m,
            and every required predicate passes}.

y : S_out -> R.
```

`K_req`, not `K_store` or `K_geom_science`, governs ordinary
scientific admission. Structural finite/non-finite screening uses only
authorized required parent locations; an exact-zero geometric offset is not a
required location and its payload MUST NOT be evaluated or dereferenced. Rows
outside `S_out` are scientifically unavailable, not zero. A stored array may
preserve parent shape and WCS only if every unavailable row carries its typed
cause and remains outside the scientific numerical function. FLT MUST
establish `p-r in D_m` before evaluating `m_(p-r)`.

For a requested nonzero convolution that resolves and applies but has
`S_out = empty set`, the exact application state is
`applied_no_scientific_output_support`. The plan, operator, parent, causes, and
application evidence are preserved. The ordinary base-signal publication
action is `not_produced` with exact cause
`no_full_footprint_output_rows`. This is not disabled, failed execution, the
zero operator, or a successful empty signal product. Identity retains its
separately declared support rule; zero uses independently constructed
`S_out_zero`.

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

Response families are separately typed and none may silently stand for
another:

- Fixed-state linear parent response:

```text
R_out^fixed = A_Theta,J R_parent^fixed;
```

- An already realized parent-grid response companion receives `A_Theta,J`
  exactly once.

- A parent full-procedure finite difference with FLT fixed follows:

```text
Delta y_parent-FP = A_Theta,J Delta m_parent-FP,
```

  Baseline and perturbed parent-procedure products MUST provide one exact
  compatible parent-grid difference on the frozen FLT domain. If row
  membership, availability, WCS, quantity, or support differs so the
  difference is not defined on the frozen required footprint, `J_full` is not
  re-resolved, the parent state-change record is retained, and the affected
  transformed full-procedure response is unavailable. Equal array shape alone
  does not establish a valid procedure difference.

- An FLT re-resolved procedure response is outside SCI-FLT-FIXED when kernel,
  cutoff, normalization, support, selector, edge state, or method is
  re-resolved.

Every admitted response uses the identical frozen `A_Theta,J`, centering,
orientation, phase, support, edge, missing-data, row-domain, and normalization
rules as the signal. If its required source basis, domain, or compatible
parent-response identity is unavailable, the requested `FLT-RSP` family is
typed unavailable. The kernel alone is not the complete source response or
PSF.

For the exact zero operator, the local fixed-state derivative and the
parent-payload conditional covariance contribution are exactly zero on its
declared scientific row domain. A complete source-domain response may still be
unavailable when its source basis, domain, or parent-response identity is
unavailable. Total systematic uncertainty remains separately typed. Local
zero response is neither hidden nor promoted into an unsupported complete
sky-response claim.

`FLT-TRANSFER` records the exact local sampled spatial or Fourier transfer on
the declared finite-grid domain where scientifically defined, or typed
unavailability. `FLT-MODE` records exact null modes, invariant modes,
attenuation statements or bounds, and sampled phase where defined. Local
operator modes are not automatically upstream sky-to-output modes.

`FLT-INFLUENCE` is the exact coefficient and support relation between parent
and output rows. It is not physical exposure. Filtering creates no new
physical exposure claim. `FLT-EXPOSURE-LINEAGE` records the exact parent
exposure-product identity or typed absence and states that FLT creates no
physical exposure. Convolving an exposure plane is not authorized as the
physical exposure of filtered signal.

## 9. Deterministic covariance and NOI boundary

For an available compatible declared parent covariance `C_parent`,

```text
C_out = A_Theta,J C_parent transpose(A_Theta,J).
```

Parent stochastic authority and output representation are separate axes.
Parent authority is exactly one of complete covariance, explicit
independent-diagonal model, marginal variances only, structured or partial
model, or unavailable. Output representation is exactly one of complete
matrix, exact linear or operator representation, structured representation,
marginal plane, summary only, or unavailable.

Every result states the parent model on which it is conditional, output
representation, domain and ordering, rank and null space, omitted terms,
supported operations, and selection, kernel, beam, WCS, calibration, and model
uncertainty exclusions. A full matrix propagated from an explicitly declared
independent-diagonal parent model is complete relative to that conditional
model; it does not imply that unknown real parent cross terms are zero.

The exact authority and representation compatibility table is:

```text
Parent authority              Authorized FLT covariance result
complete covariance           exact A C A^T on the complete domain in any
                              mathematically exact declared representation
independent-diagonal model    full covariance relative to that exact model,
                              including induced off-diagonal terms
marginal variances only       exact conditional marginal for a row with
                              exactly one nonzero parent coefficient;
                              otherwise unavailable or explicitly partial
structured or partial model   only operations proved exact for that exact
                              representation and domain
unavailable                   unavailable, except separately stated local
                              zero-operator parent-payload facts
```

Marginal variances alone do not imply independence. For an output row with
exactly one nonzero parent coefficient `A_ij`, marginal-only parent authority
is sufficient for that row's conditional marginal:

```text
Var(y_i) = A_ij^2 Var(m_j).
```

When a row mixes two or more parent variables, marginal-only authority does
not determine its exact output marginal. For a general parent covariance,

```text
Var(y_i) = sum_j A_ij^2 Var(m_j)
           + 2 sum over j<k of A_ij A_ik Cov(m_j,m_k).
```

A separately named diagonal-contribution diagnostic may report the first sum,
but it MUST NOT be called variance, covariance, uncertainty, or precision.

For an explicitly diagonal independent parent covariance with marginal
variances `V_j`,

```text
A_ij = (J_full L_Theta)_ij,
Var(y_i) = sum_j A_ij^2 V_j,
Cov(y_i,y_l) = sum_j A_ij A_lj V_j.
```

The output generally has off-diagonal covariance. A marginal variance plane is
not full covariance and does not authorize independent-pixel multi-pixel
inference. One exact row marginal authorizes neither cross-row covariance nor
independence. The zero operator retains its separately typed zero
parent-payload covariance contribution.

SCI-NOI, not SCI-FLT-FIXED, owns empirical uncertainty, empirical covariance,
conditional inverse scale, standardized signal, and significance inference.
For every compatible admitted NOI member `M_b`, exact fixed-state parity is

```text
M_b_out = A_Theta,J M_b.
```

The real parent and every member use the identical frozen `A_Theta,J`,
parameter set, grid, support, edge rule, row domain, and lifecycle generation.
`J_full` is never re-resolved per member. Filtering a
variance, standard deviation, precision, reciprocal, weight, standardized
map, or significance field is not this operation. Per-member selection or
re-resolution is a different inference-bearing method and cannot enter the
fixed-state ensemble.

Parameter, kernel, cutoff, beam, WCS, selection, model, and calibration
uncertainty remain separate from covariance conditional on fixed
`A_Theta,J` unless a separately supplied typed model includes them.

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
- `FLT-EXPOSURE-LINEAGE`;
- `FLT-SUP`;
- `FLT-VALID`;
- `FLT-COV-FORMAL`;
- `FLT-NOI-COMPATIBILITY`; and
- `FLT-LINEAGE`.

A missing required record is not an atomic bundle. An honestly unavailable
response or covariance record may satisfy the base role only where the role
permits absence; it cannot satisfy a response-qualified or
covariance-qualified request.

`FLT-NOI-COMPATIBILITY` is immutable FLT state, never the NOI product. It binds
the exact FLT product, operator, and row-domain identities; exact
FLT-to-NOI boundary and profile compatibility; fixed-state transformation
semantics; request state known at FLT publication; and typed compatibility or
unavailability. It contains no identity of a future NOI product. A later NOI
product references the immutable FLT parent. A recorded
`not_requested_at_FLT_publication` state is historical provenance and does not
prohibit a later independently requested SCI-NOI child. That child owns its
own request, applicability, eligibility, realization, generation, and failure,
and still requires exact compatible FLT boundary and profile state. Any
optional reverse
`SCI-FLT-FIXED_TO_SCI-NOI-RELATION` is a separately versioned artifact outside
FLT atomic completion and MUST NOT mutate the FLT bundle.

Publication policy consumes one exact
`complete_publication_disposition_candidate` with exactly one of two variants:

- `product_candidate`, containing the complete atomic FLT product bundle; or
- `no_output_support_candidate`, containing the exact request, parent, plan,
  scientific operator and representation identities, `K_req`, attempted
  output-domain construction, proof that `S_out` is empty, exact row/cause
  accounting, application generation,
  `applied_no_scientific_output_support` state, and prescribed publication
  cause.

The `no_output_support_candidate` contains no realized `FLT-SIG` and is not an
atomic FLT product bundle. The publication lifecycle is

```text
requested
  -> effective
  -> resolved
  -> applied
  -> complete_publication_disposition_candidate
       (product_candidate | no_output_support_candidate)
  -> publication_decision
  -> realized | failed | not_produced.
```

For a requested nonzero convolution with empty `S_out`, the branch is

```text
applied
  -> applied_no_scientific_output_support
  -> complete_publication_disposition_candidate
       (no_output_support_candidate)
  -> publication_decision
  -> not_produced (cause: no_full_footprint_output_rows).
```

`SCI-FLT-FIXED:output_publication@1` evaluates the complete publication
disposition candidate, not a pre-existing realized bundle. For the named
publication use of a requested nonzero convolution with empty `S_out`, the
exact axes are `request = requested`, `applicability = applicable`,
`eligibility = ineligible`, `realization = not_produced`, and
`cause = no_full_footprint_output_rows`. The earlier input/application use MAY
have been eligible and successfully applied. A successful publication action
for a `product_candidate` creates `realized_identity`, `realized_zero`, or
`realized` as applicable.
Disabled is `not_produced`. Failure of a required transformation, candidate,
or publication step propagates and creates no complete product. `unavailable`
remains a typed route or role state rather than a synonym for disabled, failed,
zero, or not produced. `superseded` records immutable succession after a
realized generation exists.

Any change to parent, request or effective purpose, scientific operator,
canonical coefficient map, parameter, transfer qualification, normalization,
WCS, grid, scientific row domain, support, validity, response or covariance
role, lifecycle, or failure policy creates a new immutable scientific
transformation and product generation. A representation-only change creates
at most a new representation artifact or generation and MUST NOT create a new
scientific transformation, FLT product identity, or scientific generation. A
later NOI attachment is a separate immutable companion and does not mutate FLT
or the parent.

## 11. Typed FLT policy objects and VAL boundary

The draft defines three distinct policy domains:

- `SCI-FLT-FIXED:input_bundle_admission@1` evaluates one request, one complete
  parent bundle, and one exact resolved FLT plan;
- `SCI-FLT-FIXED:input_parent_row_admission@1` evaluates each exact parent row
  for the named FLT use; and
- `SCI-FLT-FIXED:output_publication@1` evaluates one exact
  `complete_publication_disposition_candidate`, either `product_candidate` or
  `no_output_support_candidate`, and defines its disposition and prescribed
  consumer action.

`J_full` and `S_out` are deterministic FLT-owned constructions from the
parent-row decisions, `K_req`, `S_parent_fact`, `D_m`, and required predicates.
Output-row arithmetic support is not a producer fact. Membership in `D_m` MUST
be established before a signal payload is evaluated.
SCI-VAL may bind and evaluate an owner-approved profile and produce a decision
artifact; it does not convolve data or perform publication. The FLT publisher
performs or declines the prescribed action and FLT owns final realization and
FLT-local validity.

Every immutable profile binds scientific-policy owner, exact source and
boundary versions, request, applicability, eligibility, realization,
restrictions, decisive exclusions, exceptions, missing or conflict behavior,
lifecycle, and exact consumer action. The exact draft records are part of the
bound Stage B source set.

Structural identity or source conflict yields `applicability_unknown` and
`decision_unavailable`. A known applicable failed restriction yields
`ineligible`. A not-requested or effectively disabled route makes no
eligibility proposition and yields `not_produced`. An eligible transformation
failure yields `eligible` and `realization_failed`. An eligible
`product_candidate` becomes realized only after successful publication action.
An applied nonzero transformation with empty `S_out` yields
`applied_no_scientific_output_support` and a complete
`no_output_support_candidate`; for that publication use policy records
`request = requested`, `applicability = applicable`,
`eligibility = ineligible`, `realization = not_produced`, and cause
`no_full_footprint_output_rows`, and publishes no empty base-signal product.

Companion qualification is request-specific:

- a base signal request requires exact response and covariance state records
  but permits an honest `unavailable` state where their role allows it;
- a response-qualified request requires an available exact compatible response;
- a covariance-qualified request requires the exact declared compatible
  covariance representation; and
- a response-and-covariance-qualified request requires both.

An optional unavailable companion never blocks an otherwise valid base-signal
product. It does block the corresponding qualified request.

SCI-VAL may bind and evaluate an immutable owner-approved successor of these
profiles. VAL does not author producer facts, FLT policy, arithmetic, or
scientific claims. The profile records first drafted at r0.3 and their current
exact source bindings remain drafts, are not owner-approved Registry entries,
and create no claim that Registry evaluation occurred.

## 12. Consumer and ownership boundaries

MAP and JINC own the parent estimand and parent claims. CAL owns absolute
calibration, passband and color correction, and calibration covariance.
SCI-FLT-FIXED owns the exact local transformation, transformed signal, output
unit derivation, composed-response state, local transfer and modes, influence,
support and validity, deterministic covariance state, lifecycle, failure, and
provenance. It also owns its plan, selector, application, and publication
policy even when a final kernel is constructed elsewhere. SCI-NOI owns
empirical uncertainty inference and applies but does not choose the exact FLT
transformation. SCI-BEAM and future source or mode contracts own physical
source, beam, Pointing, and OOF interpretations. SCI-FRUIT owns iterative
feedback science.

`confidence` is not a generic FLT scalar. Where the applicable upstream
boundary supplies a named confidence or quality state, FLT carries its exact
typed identity and meaning as parent state and may reference it in a named row
predicate. Otherwise FLT confidence is `not_defined`; it must not be inferred
from finiteness, support, covariance, a weight, or downstream eligibility.

The availability of an SCI-FLT-FIXED product authorizes no generic Beammap,
Pointing, OOF, source-fit, catalog, NOI, or FRUIT use. Each consumer owns an
exact use policy.

## 13. Numerical-conformance policy boundary

Future finite-precision implementation evidence must be evaluated under one
preregistered immutable numerical-comparison policy. Before observing a
candidate result, that policy binds an independent exact or high-precision
oracle; absolute, relative, and near-zero behavior; signed-cancellation and
zero-sum cases; conditioning and operation-count dependence; covariance
comparison; sequential and parallel agreement; overflow, underflow, and
non-finite handling; simultaneous row-level comparison; lifecycle; and
provenance. Bounds must not be changed after observing a failure.

The policy preserves exact operator coefficients and scientific identity. It
authors no filter science and supplies no implementation-conformity,
validation, numerical-adequacy, or performance claim. The current draft policy
is bound as a separate Stage B source artifact and is not a preregistered
future evidence decision.

## 14. Exclusions and nonclaims

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

## 15. Normative requirements

### SCI-FLT-FIXED-REQ-001 - Package identity

The package SHALL be identified as SCI-FLT-FIXED v0.1 within the SCI-FLT
tranche. `FLT-FIXED-CONV` SHALL be the only numerically admitted base family,
with `FLT-FIXED-CONV-LOWPASS` only a qualified subtype. It SHALL NOT be
presented as SCI-FLT-INF, an arbitrary dense linear-operator family, or a
generic filter contract.

### SCI-FLT-FIXED-REQ-002 - Strict linearity

The transformation MUST be exactly `y = A_Theta,J m`, where
`A_Theta,J = J_full L_Theta`, with no additive term. Strict linearity MUST be
conditioned on one exact frozen parent membership and selector, with
`m : D_m -> R` and `y : S_out -> R`. `S_parent_fact`, `D_m`, and `S_out` MUST
remain separately typed. Any offset, background, template subtraction, or
additive correction MUST be rejected as outside v0.1.

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
and transfer qualification MUST be resolved under one exact FLT-owned plan and
frozen before application. Parent payload amplitudes MUST NOT select, learn,
tune, or alter any plan fact. The selector MUST be resolved once from only the
declared immutable `S_parent_fact`, row membership and admission, typed facts
defining `D_m`, support, and predicates in Section 2 before convolution
arithmetic. FLT MUST establish membership in `D_m` before evaluating a signal
payload and MUST NOT re-resolve the selector for response, covariance, noise,
or NOI members.

### SCI-FLT-FIXED-REQ-008 - Same-grid boundary

The output MUST preserve the exact parent WCS, frame, topology, metric, shape,
pixel indexing, and pixel-area convention, apart from scientific-row
restriction. Reprojection, resampling, and approximate-WCS joins MUST be
rejected.

### SCI-FLT-FIXED-REQ-009 - Complete operator identity

The applied operator MUST bind the separate scientific-operator and
representation identities in Sections 5 and 6. Scientific identity MUST bind
the canonical exact offset-to-coefficient map and units,
`K_geom_science`, `K_nonzero`, `K_req`, scientific domains, coordinate and WCS
facts, normalization, transfer qualification, edge rule, and all downstream
science. Representation identity MUST separately bind `K_store`, encoding,
field and byte ordering, container/compression, representation digest, and
representation generation. `K_store` MUST be scientifically nonauthoritative;
the scientific digest and every scientific support set MUST be
representation-invariant. Each coefficient MUST satisfy the real, finite,
unit-typed, canonical, content-bound conditions in Section 2.

### SCI-FLT-FIXED-REQ-010 - Fixed convolution construction

A `FLT-FIXED-CONV` method MUST construct the complete finite operator from one
exact finite sampled kernel and its declared offset sets, orientation, center,
phase, normalization, coordinate-domain method, and coefficient
representation. Its ordinary sum MUST range exactly over
`K_nonzero = K_req`, never `K_geom_science` or `K_store`. One product MUST
apply this convolution exactly once and MUST make no intermediate
transformation or reordered-composition claim.

### SCI-FLT-FIXED-REQ-011 - Low-pass qualification

A `FLT-FIXED-CONV-LOWPASS` claim MUST bind every coordinate, metric, frequency,
and transfer fact listed in Section 6 and distinguish sampled-kernel transfer
from the complete finite row-restricted operator. If any fact is missing, the
low-pass claim MUST be unavailable even when fixed-convolution identity
remains available.

### SCI-FLT-FIXED-REQ-012 - Full-footprint admission

`S_out` MUST use ordinary-method `K_req = K_nonzero` and contain exactly those
rows for which every scientifically required parent location is in
`S_parent_fact`, admitted for the exact FLT use, in `D_m`, and passing every
required predicate. Membership in `D_m` MUST be established before `m` is
evaluated. Exact-zero coefficient status MUST be read from the canonical
representation, never a floating threshold. Storage layout MUST NOT change row
admission. An exact-zero offset MUST require no parent payload, MUST create no
influence or covariance contribution, MUST NOT remove a row, and MUST NOT
cause its payload to be evaluated or dereferenced.

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

Each `FLT-RSP` MUST identify exactly one response family in Section 8. An
available compatible fixed-state response, realized parent-grid response, or
parent full-procedure finite difference MUST receive the identical frozen
`A_Theta,J` exactly once as specified. A full-procedure difference MUST be
defined on one exact compatible frozen parent-grid domain; a membership,
availability, WCS, quantity, or support mismatch MUST yield affected-response
unavailability without re-resolving `J_full`. FLT re-resolution is outside
v0.1. A missing required basis, domain, or parent-response identity MUST yield
typed unavailability rather than substitution.

### SCI-FLT-FIXED-REQ-018 - Transfer and mode state

`FLT-TRANSFER` and `FLT-MODE` MUST publish the exact sampled-kernel or local
finite-grid transfer, null, invariant, attenuation, and phase facts where
defined, or honest typed unavailability. They MUST NOT be promoted to an
unproved global finite-operator, angular-WCS, or whole-chain claim.

### SCI-FLT-FIXED-REQ-019 - Influence is not exposure

`FLT-INFLUENCE` MUST describe the exact parent-to-output coefficient relation
and MUST NOT be labeled or interpreted as physical exposure.
`FLT-EXPOSURE-LINEAGE` MUST identify the exact parent exposure product or typed
absence and MUST state that FLT creates no physical exposure.

### SCI-FLT-FIXED-REQ-020 - Distinct support and validity states

Numerical computability, `K_geom_science`, `K_store`, `K_nonzero`, `K_req`, complete footprint,
FLT bundle and row admission, FLT-local validity, parent validity, named
upstream confidence or `not_defined`, and downstream eligibility MUST remain
distinct with exact causes.

### SCI-FLT-FIXED-REQ-021 - Deterministic covariance propagation

An available compatible parent covariance MUST be propagated by
`C_out = A_Theta,J C_parent transpose(A_Theta,J)` on the exact scientific row
domain with the identical frozen selector and only under the exact authority
and representation compatibility rules in Section 9.

### SCI-FLT-FIXED-REQ-022 - Covariance state honesty

`FLT-COV-FORMAL` MUST separately state parent stochastic authority and output
representation using Section 9's two axes, plus conditional model, domain,
ordering, rank, null space, omissions, supported operations, and excluded
uncertainties. Unknown cross terms MUST NOT be set to zero or interpreted as
independence.

### SCI-FLT-FIXED-REQ-023 - Induced covariance

For an explicitly diagonal independent parent model, the exact output
off-diagonal terms using `A_ij` MUST be retained in any complete-relative-to-
that-model covariance claim. Marginal parent variances MUST NOT infer
independence or determine an exact output marginal for a row mixing multiple
parent variables. A marginal plane MUST NOT be called full covariance, and a
diagonal-contribution diagnostic MUST NOT be called variance, covariance,
uncertainty, or precision.

### SCI-FLT-FIXED-REQ-024 - Empirical uncertainty ownership

SCI-FLT-FIXED MUST NOT infer empirical uncertainty, empirical covariance,
conditional inverse scale, standardized signal, or significance. Such an
attachment remains SCI-NOI-owned and separate from `FLT-COV-FORMAL`.

### SCI-FLT-FIXED-REQ-025 - Fixed-state NOI parity

Every admitted NOI member for the exact transformed product MUST receive the
identical frozen `A_Theta,J`, parameters, grid, support, edge rule, row domain,
and lifecycle generation used for the real parent. A member failing a frozen
required footprint MUST be unavailable there rather than receiving a new
selector.

### SCI-FLT-FIXED-REQ-026 - Relearning rejection

Any per-member parameter selection, operator resolution, or relearning MUST be
rejected from the fixed-state ensemble and routed to a separately named
inference-bearing method.

### SCI-FLT-FIXED-REQ-027 - Atomic product roles

A realized product MUST contain every required role record in Section 10,
including explicit unavailable companion records where the base role permits
absence. Immutable `FLT-NOI-COMPATIBILITY`, not an SCI-NOI product, future NOI
identity, or later-changing attachment state, MUST occupy the FLT relation
role. A partial bundle MUST NOT be published.

### SCI-FLT-FIXED-REQ-028 - Lifecycle

The lifecycle MUST evaluate publication policy against one exact
`complete_publication_disposition_candidate`, either `product_candidate` or
`no_output_support_candidate`, before realization and MUST distinguish all
states and transitions in Section 10 while preserving exact causes, failures,
and immutable generation bindings.

### SCI-FLT-FIXED-REQ-029 - Disabled, identity, and zero states

Disabled MUST be `not_produced`. Requested and applied identity and zero
operators MUST use their explicit support rules, `product_candidate`, and
publication decision to produce distinct realized products and MUST NOT be
represented as disabled or unavailable. Zero MUST use
`K_nonzero_zero = empty set`, `K_req_zero = empty set`, and independently
constructed `S_out_zero`; it MUST NOT acquire rows through a vacuous
empty-footprint test.

### SCI-FLT-FIXED-REQ-030 - Generation identity

Any change to a scientific operator fact listed in Section 10 MUST create a new
immutable scientific transformation and product generation. A
representation-only change MAY create a representation artifact or generation
but MUST retain the same scientific operator identity, FLT product identity,
and scientific generation. A separately realized NOI product or optional
reverse relation MUST reference the immutable FLT product, MUST NOT determine
FLT completeness, and MUST NOT mutate the FLT product.

### SCI-FLT-FIXED-REQ-031 - Failure and fallback

Missing, conflicting, or unavailable state required by the requested identity
MUST fail the affected route closed with exact causes. No silent fallback,
default, or same-name substitution may retain that requested identity.

### SCI-FLT-FIXED-REQ-032 - Input admission policy

The `SCI-FLT-FIXED:input_bundle_admission@1` and
`SCI-FLT-FIXED:input_parent_row_admission@1` draft semantics in Section 11 MUST
govern their distinct bundle and exact parent-row domains, typed applicability
and eligibility, unavailable decision, exclusions, and fail-closed action
until superseded by owner-approved immutable profiles. `J_full` and `S_out`
MUST remain deterministic FLT constructions, not profile-authored output-row
arithmetic.

### SCI-FLT-FIXED-REQ-033 - Output publication policy

The `SCI-FLT-FIXED:output_publication@1` draft semantics in Section 11 MUST
govern the exact `complete_publication_disposition_candidate`, its
`product_candidate` and `no_output_support_candidate` variants, disposition,
and prescribed consumer action. A policy or VAL evaluation MUST NOT itself
perform publication; the FLT publisher performs or declines that action.
Honest absence is permitted only for a role and request qualification that
explicitly allows it.

### SCI-FLT-FIXED-REQ-034 - VAL boundary

SCI-VAL MAY bind and evaluate an owner-approved immutable policy and produce a
decision artifact but MUST NOT author FLT facts or policy, perform the
transformation or publication, or convert these unapproved draft profile bytes
into a Registry-evaluated or numerical route.

### SCI-FLT-FIXED-REQ-035 - Consumer boundary

Product availability MUST NOT authorize generic downstream use. Beammap,
Pointing, OOF, source-fit, catalog, NOI, and FRUIT consumers MUST supply their
own exact owner-approved use policies.

### SCI-FLT-FIXED-REQ-036 - Excluded methods and nonclaims

Every method in Section 14 MUST remain outside v0.1. The package MUST preserve
all Stage B nonclaims in Section 14.

### SCI-FLT-FIXED-REQ-037 - Frozen-selector conditioning

`J_full` MUST be resolved exactly once before convolution arithmetic from only
declared immutable `S_parent_fact`, row membership, exact parent-row admission,
typed facts defining `D_m`, support, and exact required predicates. Membership
in `D_m` MUST be established before `m_q` is evaluated. Structural screening
of declared typed row-state facts MUST NOT evaluate `m_q` or tune the plan. Response
perturbations, covariance draws, noise realizations, and NOI members MUST reuse
the resulting `A_Theta,J`; selection and support uncertainty MUST remain
excluded unless separately supplied as typed uncertainty.

### SCI-FLT-FIXED-REQ-038 - Required-dependency support

Every kernel MUST distinguish representation-invariant `K_geom_science`,
nonauthoritative `K_store`, `K_nonzero`, and `K_req` and bind their exact
relations. The ordinary method MUST use `K_req = K_nonzero`; identity uses
`K_req = {0}`; and the zero operator uses `K_nonzero_zero = empty set`,
`K_req_zero = empty set`, and independently constructed `S_out_zero` equal to
the exact admitted finite parent-signal row domain under its request and
predicates. A parent row domain MUST NOT be described as a `K_req` domain. An
exact-zero ordinary
coefficient contributes no arithmetic dependency, influence, covariance, or
row exclusion and its parent payload MUST NOT be dereferenced. A zero-valued
required offset requires a separately named scientific method independent of
storage.

### SCI-FLT-FIXED-REQ-039 - Typed policy and companion qualification

All three policy objects in Section 11 MUST bind their exact domains and VAL-
congruent state semantics. Base, response-qualified, covariance-qualified, and
jointly qualified requests MUST apply their distinct companion requirements;
an optional unavailable companion MUST NOT block a valid base signal.

### SCI-FLT-FIXED-REQ-040 - Response-family separation

Fixed-state linear response, already realized parent-grid response, parent
full-procedure finite difference with FLT fixed, and FLT re-resolved procedure
response MUST remain separately identified. No response family or local zero
derivative may be promoted into another or into an unsupported complete
source-domain claim.

### SCI-FLT-FIXED-REQ-041 - Single-convolution composition

One SCI-FLT-FIXED v0.1 product MUST apply one exact resolved sampled
convolution exactly once. Intermediate transformations, multiple application,
selector collapse, or operator reordering MUST NOT be claimed by the product.

### SCI-FLT-FIXED-REQ-042 - Numerical-comparison policy

Any future finite-precision conformance evidence MUST preregister every
comparison-policy field in Section 13 before candidate results are observed.
Comparison bounds MUST NOT change after a failure, and the policy MUST NOT
author filter science or claim validation, adequacy, or performance.

### SCI-FLT-FIXED-REQ-043 - Exposure lineage

Every complete candidate MUST carry `FLT-EXPOSURE-LINEAGE` with the exact
parent exposure identity or typed absence. FLT MUST create no physical
exposure, MUST keep influence distinct, and MUST NOT authorize a convolved
exposure plane as filtered-signal physical exposure.

### SCI-FLT-FIXED-REQ-044 - External-resolution ownership

"Externally resolved" MUST mean complete before application under one exact
FLT-owned plan. It MUST NOT transfer FLT policy, transformation, selector,
application, or publication ownership to an unnamed producer.

### SCI-FLT-FIXED-REQ-045 - Exact parent signal role

Each parent role MUST bind the exact signal vector in Section 3. Only that
signal role may receive `L_Theta` as `FLT-SIG`. JINC numerator, signed
normalization, quadratic accumulator, and coefficient-squared temporal
accounting; MAP exposure; support; validity; covariance; and response MUST NOT
be transformed as `FLT-SIG` by the ordinary method.

### SCI-FLT-FIXED-REQ-046 - Covariance authority compatibility

Every covariance result MUST follow the Section 9 authority/representation
table. Complete, independent-diagonal, marginal-only, structured or partial,
and unavailable parent authorities MUST remain distinct. Unknown cross terms
MUST keep any marginal for a row mixing multiple parent variables, and any
unsupported cross-row covariance, unavailable or explicitly partial. A row
with exactly one nonzero parent coefficient MAY publish its exact conditional
marginal without implying independence.

### SCI-FLT-FIXED-REQ-047 - Immutable NOI compatibility

`FLT-NOI-COMPATIBILITY` MUST contain only FLT identity, operator and row-domain
identity, boundary/profile compatibility, fixed-state semantics, publication-
time request state, and typed compatibility or unavailability. It MUST NOT
contain a future NOI identity. A later NOI child or reverse relation MUST NOT
mutate FLT. `not_requested_at_FLT_publication` MUST be historical provenance
only and MUST NOT prohibit a later independently requested compatible SCI-NOI
child with its own lifecycle.

### SCI-FLT-FIXED-REQ-048 - Policy actor separation

Input profiles MUST evaluate complete bundles and exact parent rows. FLT MUST
construct `J_full` and `S_out`. Publication policy MUST evaluate either exact
candidate variant and define a disposition and prescribed action; VAL MAY
produce a decision artifact; only the FLT publisher may perform or decline
publication and establish FLT realization and local validity.

### SCI-FLT-FIXED-REQ-049 - Low-pass transform convention

Every low-pass-qualified plan MUST content-bind every transform, frequency,
quantity, attenuation, region, phase, anisotropy, and WCS-metric convention in
Section 6. A different convention, even with complete metadata, MUST fail the
qualification.

### SCI-FLT-FIXED-REQ-050 - Full-procedure response domain

A transformed parent full-procedure response MUST use one exact compatible
baseline/perturbed difference on the frozen FLT domain. Membership,
availability, WCS, quantity, or support incompatibility MUST retain the parent
state-change record, MUST NOT re-resolve `J_full`, and MUST make affected
response rows unavailable.

### SCI-FLT-FIXED-REQ-051 - Consolidated authority preflight

`AUTHORITY_MANIFEST.json`, the proposed-freeze authority manifest, MUST bind the complete Stage A packet,
all owner directives, every Stage B scientific and policy source, exact build
and verification records, reports, and rendered PDFs by path, bytes, and
SHA-256, plus scientific/process/representation role, authority state,
compatibility or supersession state, and generated-view relation. Any
unreproducible dependency MUST route its dependent claim to typed
unavailability rather than freeze disposition.

### SCI-FLT-FIXED-REQ-052 - Real scalar and coefficient admissibility

The base functions MUST be `m : D_m -> R` and `y : S_out -> R`, with
`S_parent_fact`, `D_m`, and `S_out` separately typed; `k_Theta(r)`, `L_Theta`,
and `A_Theta,J` MUST be real-valued.
Every sampled coefficient MUST be finite, real, unit-typed, canonically exactly
represented, and content-bound before application. A missing, non-finite,
complex, unrepresentable, or conflicting coefficient MUST make plan resolution
unavailable and MUST NOT be repaired by numerical-comparison policy. Complex
`H(nu)` MUST remain only a representation of the real map operator.

### SCI-FLT-FIXED-REQ-053 - Empty scientific output support

For a requested nonzero convolution that resolves and applies with
`S_out = empty set`, FLT MUST record
`applied_no_scientific_output_support`, preserve its bound evidence in a
complete `no_output_support_candidate`, and prescribe the publication-use axes
`request = requested`, `applicability = applicable`,
`eligibility = ineligible`, `realization = not_produced`, and cause
`no_full_footprint_output_rows`. The candidate MUST contain no realized
`FLT-SIG` and MUST NOT be an atomic bundle. FLT MUST NOT publish a successful
empty signal product or relabel the state as not requested, disabled, failed
execution, decision unavailable, identity, or zero.

## 16. Falsifiable predictions

Each prediction is conditional on an exact admitted parent and fully resolved
operator unless the prediction explicitly tests unavailability.

### SCI-FLT-FIXED-PRED-001 - Identity operator

For the exact identity operator, `FLT-SIG` equals the parent signal on `S_out`.
An available compatible response and covariance are unchanged on that domain.
`K_req = {0}` and `S_out` equals the exact admitted finite parent-row domain.
The result is `realized_identity` only after successful publication, not
disabled and not an unparented copy.

### SCI-FLT-FIXED-PRED-002 - Zero operator

For the exact zero operator, `K_nonzero_zero` and `K_req_zero` are empty, while
`S_out_zero` is independently the exact admitted finite parent-signal row
domain under the request and predicates. Every `FLT-SIG` value on
`S_out_zero` is zero; the local fixed-state derivative and parent-payload
conditional covariance contribution are zero there. No arbitrary storage row
is admitted by vacuity. A missing complete source basis or parent response
remains typed unavailable, and total systematic uncertainty remains separate.
The product is `realized_zero` only after publication, not disabled, invalid,
or evidence of precision.

### SCI-FLT-FIXED-PRED-003 - Input scaling

For any finite scalar `a`, applying the same frozen `A_Theta,J` to `a m`
produces `a y`. Changing parent membership or re-resolving `J_full` is outside
this conditional-linearity comparison. Failure on the fixed domain falsifies
strict linearity.

### SCI-FLT-FIXED-PRED-004 - Constant input and DC gain

For constant admitted input `m_q = c`, every full-footprint convolution row is
`c` times the exact signed kernel sum, equivalently `c` times the declared DC
gain. Constant preservation occurs exactly when that gain is one; no other
normalization implies it.

### SCI-FLT-FIXED-PRED-005 - Impulse response

An admitted single-pixel unit impulse produces the exact sampled kernel shifted
according to the declared center, orientation, handedness, phase, and indexing,
with arithmetic terms only at exact `K_nonzero = K_req` offsets, then
restricted by `S_out`. Any implicit recentering, interpolation, reversal,
periodic copy, or geometric exact-zero term falsifies the operator identity.

### SCI-FLT-FIXED-PRED-006 - Parent-response composition

Applying the transformation to an exact compatible fixed-state response,
already realized response companion, or parent full-procedure difference
produces exactly its separately identified `A_Theta,J` composition. A
kernel-only surrogate, a second application, a re-resolved selector, or a
response using different edge, phase, or normalization rules fails.

### SCI-FLT-FIXED-PRED-007 - Signed kernel

A signed kernel follows its exact signed coefficients, including cancellation.
Its geometric, signed, absolute, and squared support summaries remain unequal
when the coefficients make them unequal; none may substitute for another.

### SCI-FLT-FIXED-PRED-008 - Zero-sum kernel

For a complete constant input and exact zero signed-sum kernel, every admitted
output row is zero. This nulling does not by itself make the row unavailable,
uncertain, source-free, or statistically significant.

### SCI-FLT-FIXED-PRED-009 - Full-footprint boundary

For every output row whose exact `K_req` footprint lies in `S_parent_fact`, is
admitted, lies in `D_m`, and passes every predicate, the row is in `S_out`.
Removing a required location from `S_parent_fact` or `D_m`, or failing its
admission or predicate, removes every dependent output row with an exact
cause; `m_q` is never evaluated before `q in D_m` is established. Dense and
sparse representations with distinct representation identities but the
identical canonical coefficient map produce the identical scientific operator
identity, FLT product identity, `K_req`, and `S_out`. An unavailable or
non-finite payload at an exact-zero geometric offset leaves the row and value
unchanged; making that coefficient exactly nonzero activates the dependency
and removes the row.

### SCI-FLT-FIXED-PRED-010 - Deferred edge methods

Any result that uses extension, wrapping, truncation, local support
renormalization, inpainting, reflection, clamp, mirror, padding-based admission,
edge completion, or value replacement is rejected as a v0.1 product even when
it is finite or preserves a constant.

### SCI-FLT-FIXED-PRED-011 - Missing and non-finite input

No output row depending on a required location outside `S_parent_fact`, outside
`D_m`, not admitted, or failing a required predicate belongs to `S_out`. Such a
row is unavailable rather than zero and records the applicable cause. A stored
missing, unavailable, or non-finite payload remains a typed fact and is never
evaluated as an element of `m`.

### SCI-FLT-FIXED-PRED-012 - Complete covariance transform

For an exact available compatible `C_parent`, the published complete covariance
equals `A_Theta,J C_parent transpose(A_Theta,J)` on the declared row ordering.
A different domain, ordering, or omitted required term fails the complete-
covariance claim.

### SCI-FLT-FIXED-PRED-013 - Off-diagonal covariance from diagonal input

For an explicitly independent-diagonal parent model, overlapping nonzero
output rows produce the exact cross term `sum_j A_ij A_lj V_j`. Publishing
only marginals cannot pass a complete-relative-to-that-model covariance check
when this term is nonzero, and that conditional completeness says nothing
about unknown real parent cross terms.

### SCI-FLT-FIXED-PRED-014 - Unavailable parent companion

For a base signal request, unavailable parent response or covariance yields an
honest unavailable state record without blocking an otherwise complete signal.
The same state fails a corresponding response-qualified or covariance-qualified
request. A zero array, diagonal guess, weight, denominator, or kernel-only
surrogate fails.

### SCI-FLT-FIXED-PRED-015 - WCS or grid mismatch

Any mismatch in parent and output WCS, frame, topology, metric, shape, pixel
indexing, or pixel-area convention makes the same-grid route unavailable.
Approximate equality or successful numerical resampling does not pass.

### SCI-FLT-FIXED-PRED-016 - Observation and coadd identity

A filtered observation, filtered MAP coadd, and filtered JINC observation have
distinct parent and successor identities even if their arrays are numerically
equal. No coadd of filtered observations is an SCI-FLT-FIXED v0.1 product.

### SCI-FLT-FIXED-PRED-017 - Exact NOI parity

For each compatible admitted NOI member, applying the exact frozen signal
operator produces `A_Theta,J M_b` on the identical row domain. A member missing
one frozen required footprint is unavailable there; a different coefficient,
parameter, grid, support, edge rule, generation, or selector makes the route
unavailable.

### SCI-FLT-FIXED-PRED-018 - Per-member re-resolution

If any NOI member selects or re-resolves a kernel, cutoff, support, threshold,
edge state, or other parameter, that member is rejected from the fixed-state
ensemble rather than mixed with it.

### SCI-FLT-FIXED-PRED-019 - Disabled, identity, zero, and failure

A disabled route is `not_produced`; identity and zero each pass through a
complete `product_candidate` and publication decision before emitting their
separately parented realized products; and a required failure emits no complete
product and propagates its cause. Collapsing any pair fails the lifecycle
contract.

### SCI-FLT-FIXED-PRED-020 - Upstream unavailable parent

At the launch state recorded by the admitted boundaries, an attempted ordinary
numerical MAP or TolTEC JINC transformation remains unavailable unless a
separately authorized upstream successor supplies every named gate. The FLT
contract alone cannot make that route numerical.

### SCI-FLT-FIXED-PRED-021 - Low-pass claim completeness

Removing any required low-pass transfer fact from an otherwise complete fixed
convolution makes only the low-pass qualification unavailable. Retaining a
low-pass label with an incomplete transfer specification fails the contract.

### SCI-FLT-FIXED-PRED-022 - Exact-zero and storage invariance

For the ordinary method, an exact-zero canonical coefficient is absent from
`K_nonzero` and `K_req`. Dense and sparse encodings bind different
representation identities and generations when their representation facts
differ, while preserving the identical scientific operator identity, FLT
product identity, scientific generation, `K_geom_science`, `K_nonzero`,
`K_req`, `S_out`, arithmetic, influence, covariance, response, and claim. Its
parent payload is not evaluated or dereferenced. Making that coefficient any
exact nonzero value changes the canonical scientific map and activates a new
scientific operator identity, arithmetic, row dependency, and scientific
generation. A floating threshold or storage layout that changes this result
falsifies the identity separation.

### SCI-FLT-FIXED-PRED-023 - Zero-operator row support

The exact zero operator has `K_nonzero_zero = empty set` and
`K_req_zero = empty set` and produces rows exactly on independently
constructed `S_out_zero`, the admitted finite parent-signal row domain under
its request and predicates. An empty required footprint that admits storage
rows outside `S_out_zero`, or a zero array with no parent identity and row
support, fails the contract. Its empty arithmetic sum creates no
parent-payload covariance contribution and is not the empty `S_out` state of a
nonzero convolution, which by definition has nonempty `K_nonzero`.

### SCI-FLT-FIXED-PRED-024 - Independent low-pass transfer check

For a qualified low-pass subtype, an independent exact sampled-transfer
evaluation using the identical content-bound transform sign, normalization,
units, origin, ordering, signed/Nyquist treatment, frequency grid, response
quantity, attenuation units, region geometry, phase, anisotropy, and WCS
metric reproduces the declared DC gain and transfer. Complete metadata under a
different convention fails. PRED-021 metadata completeness without this
numerical agreement is insufficient.

### SCI-FLT-FIXED-PRED-025 - Non-signal role rejection

An ordinary-method attempt to transform JINC numerator, signed normalization,
quadratic accumulator, coefficient-squared temporal accounting, MAP exposure,
support, validity, covariance, or response as `FLT-SIG` is rejected. The same
objects may enter only their separately typed facts or compositions.

### SCI-FLT-FIXED-PRED-026 - Covariance cross-term sensitivity

For two compatible parent covariance matrices with identical diagonals and
different cross terms, any filtered output row that mixes the affected parent
variables has different variance when the corresponding weighted cross term
is nonzero. A marginal-only parent selects neither result and leaves the exact
output marginal unavailable or explicitly partial. In contrast, a row with
exactly one nonzero coefficient has conditional marginal
`A_ij^2 Var(m_j)` from marginal-only authority; this authorizes no cross-row
covariance or independence. The exact-zero row keeps its separately typed zero
parent-payload contribution.

### SCI-FLT-FIXED-PRED-027 - NOI child nonmutation

Realizing, replacing, superseding, or removing a later SCI-NOI child or an
optional reverse relation leaves the immutable FLT bundle bytes, completeness,
realization, and `FLT-NOI-COMPATIBILITY` unchanged. Any mutation fails the
atomic product contract. A `not_requested_at_FLT_publication` record does not
block a later independently requested compatible child; that child owns its
own request, applicability, eligibility, realization, generation, and failure.

### SCI-FLT-FIXED-PRED-028 - Full-procedure domain incompatibility

If baseline and perturbed parent-procedure products share array shape but
differ in row membership, availability, WCS, quantity, or required support,
the affected transformed full-procedure response is unavailable, `J_full`
remains frozen, and the parent state-change record remains visible. Producing a
numerical difference by shape alone fails.

### SCI-FLT-FIXED-PRED-029 - Empty scientific output support

For a requested nonzero convolution wider than its parent, or otherwise having
no complete admitted footprint, application yields `S_out = empty set` and
`applied_no_scientific_output_support`. It creates a complete
`no_output_support_candidate` containing the exact attempted-domain proof and
evidence but no realized `FLT-SIG` and no atomic bundle. For the publication
use, the exact axes are requested, applicable, ineligible, and not produced,
with cause `no_full_footprint_output_rows`. A not-requested, disabled,
execution-failed, decision-unavailable, zero-operator, or realized empty-signal
result fails.

### SCI-FLT-FIXED-PRED-030 - Invalid coefficient rejection

A kernel containing any missing, non-finite, complex, unrepresentable, or
conflicting coefficient makes plan resolution unavailable before application.
No numerical-comparison tolerance repairs it. A kernel with all coefficients
finite, real, unit-typed, canonically represented, and content-bound passes
this coefficient-admissibility fixture without thereby making any low-pass,
response, covariance, or implementation claim.
