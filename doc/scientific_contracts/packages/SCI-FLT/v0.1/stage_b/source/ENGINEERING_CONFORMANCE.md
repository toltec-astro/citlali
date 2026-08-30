# SCI-FLT-FIXED v0.1 Engineering-Conformance Specification

Document identity: `SCI-FLT-FIXED-ENGINEERING-CONFORMANCE v0.1/draft-r0.2`

Status: implementation-blind Stage B conformance-target draft; no conformity finding

Scientific owner: Grant Wilson

Normative import: the complete
`SCI-FLT-FIXED-NORMATIVE-CORE v0.1/draft-r0.2`, source SHA-256
`{{NORMATIVE_CORE_SHA256}}`, is incorporated without modification. This view
adds no scientific rule. If this view and the imported core differ, the core
controls.

## 1. Conformance target and evidence boundary

This specification translates the shared scientific core into observable
engineering obligations without describing, inspecting, or judging an
implementation. A future conformance review must bind one candidate realization
to one exact core revision, parent generation, resolved operator generation,
and product generation.

A claim passes only when positive evidence establishes the exact required
identity and behavior. A missing, conflicting, approximate, same-name, or
inferred fact yields unavailable or failed status as directed by the core. A
finite array, successful process exit, or familiar label is not sufficient.

Conformance, scientific validation, calibration, performance, readiness, and
production are separate evidence classes. This specification defines only a
future conformance target.

## 2. Required input and plan records

A candidate realization must accept an immutable request record, effective
decision, complete parent binding, and externally resolved fixed plan. The
records must expose enough content to reconstruct and compare exact identities,
not merely display friendly names.

The parent record must distinguish MAP observation, MAP coadd, and JINC
observation roles. It must bind the complete applicable upstream identity,
quantity, units, nominal-beam, WCS, support, validity, response, covariance,
null, exposure, lifecycle, failure, and provenance state. JINC requires all five
atomic numerical roles. Upstream unavailability must remain explicit.

The plan record must distinguish requested, effective, disabled, unavailable,
and resolved states. A resolved plan must identify `FLT-FIXED-CONV` as the
sole base family, bind one exact sampled convolution, and bind every
coefficient, parameter, coordinate domain, grid and metric fact,
normalization, support and edge rule, transfer qualification, and provenance
fact before application. "Externally resolved" does not transfer FLT policy
or application ownership.

## 3. Operator reconstruction record

`FLT-OPERATOR` must provide a content-bound representation sufficient to
reconstruct the exact finite `L_Theta`, frozen `J_full`, and complete
`A_Theta,J`, including:

- ordered input and output row domains;
- exact WCS, frame, topology, metric, shape, indexing, and pixel-area facts;
- complete sampled coefficients, coefficient units and ordering, numerical
  representation, and digest;
- `K_geom`, `K_nonzero`, `K_req`, their relations, exact-zero representation,
  center, extent, tie, phase, subpixel convention, orientation, and
  handedness;
- declared normalization, DC gain, and distinct signed, absolute, squared, and
  geometric support summaries;
- full-footprint predicates, exact inherited zero-operator row domain, and
  cause vocabulary;
- qualified transfer facts or explicit unavailability; and
- immutable plan and operator generation identifiers.

The record must show that one product applies this exact sampled convolution
once and makes no intermediate or reordered-composition claim. An alternate
computational mechanism is comparable only after reconstructing the identical
declared operator. A different coefficient representation that
changes a declared coefficient, ordering, phase, row domain, or numerical
operator is not scientific equivalence.

## 4. Admission and application gate

Bundle admission must evaluate
`SCI-FLT-FIXED:input_bundle_admission@1` against one exact request, parent, and
resolved plan. Row admission must evaluate
`SCI-FLT-FIXED:input_row_admission@1` for every candidate row against the
frozen plan and exact `K_req`. Each returns the typed request, applicability,
eligibility, missing/conflict, action, and cause state defined by its bound
profile. Neither performs payload arithmetic.

Application may proceed only for an eligible resolved route. The selector must
be resolved once from immutable membership before payload arithmetic. For every
output row, it verifies all `K_req` locations against the exact parent domain,
FLT input admission, availability, finiteness, and required predicates. The
scientific row domain must equal the passing set. Response perturbations,
covariance draws, noise realizations, and NOI members must reuse that selector;
a member failure produces row unavailability, not re-selection.

No extension, periodic wrap, truncation, local renormalization, inpainting,
reflection, clamp, mirror, padding-based admission, edge completion, or value
replacement is conforming in v0.1. Rows failing the full-footprint gate remain
unavailable with causes, even if a storage payload is finite.

## 5. Signal, response, transfer, mode, and influence records

Application must compute `FLT-SIG` from the exact resolved operator on the
exact scientific row ordering. The output-unit record must derive units from
parent and coefficient units and retain the originating nominal-beam and
calibration lineage without introducing a filtered-beam label.

Each response record must name fixed-state linear, already realized parent-
grid, parent full-procedure finite difference with FLT fixed, or FLT re-
resolved procedure family. The first three must apply the identical frozen
`A_Theta,J` exactly once; the fourth is outside v0.1. Missing basis, domain, or
parent-response identity is explicitly unavailable. The exact zero operator
records its local zero derivative without promoting it to an unavailable
complete source response. Kernel-only, approximate, multiply applied,
differently centered, differently normalized, or differently edged surrogates
do not pass.

Transfer and mode records must bind the exact coordinate domain, WCS metric,
frequency grid, finite-grid domain, phase, normalization, null, invariant,
anisotropy, and attenuation facts that are scientifically defined. They must
distinguish sampled-kernel transfer from the complete row-restricted operator.
Unsupported facts remain unavailable. Influence exposes the parent-row
coefficient relation; `FLT-EXPOSURE-LINEAGE` binds parent exposure identity or
typed absence and confirms that neither influence nor a convolved exposure
plane is physical exposure.

## 6. Covariance and NOI records

For an available compatible parent covariance, a covariance-qualified candidate
must realize the exact two-sided frozen-operator propagation on `S_out`. Its
record must state parent stochastic authority separately from output
representation and name the conditional model, omitted terms, excluded
selection/kernel/beam/WCS/calibration/model uncertainties, domain, ordering,
rank, null space, and supported operations.

If an explicitly diagonal parent covariance is propagated, any complete output
representation must include induced cross-row covariance. A marginal-only
representation must say so. Missing cross terms may not be encoded as zero or
independence.

`FLT-NOI-ATTACHMENT-STATE` must be an FLT relation record, not the separately
owned NOI product and not an FLT completion dependency. Every admitted NOI
member must receive the identical frozen `A_Theta,J`; member footprint failure
is unavailable. The route rejects filtering uncertainty-like products,
approximate transfer, relocation, commutation, substitution, and any per-member
re-resolution.

## 7. Atomic publication and lifecycle records

Publication must evaluate `SCI-FLT-FIXED:output_publication@1` against one
complete publication candidate. The candidate contains all required roles from
the imported core, including request-specific honest unavailable companion
records. A partial candidate, placeholder, or inferred companion fails
publication; a successful publication action creates the realized product.

Lifecycle evidence must distinguish `not_requested`, `requested`, `effective`,
`disabled`, `unavailable`, `resolved`, `applied`,
`complete_publication_candidate`, `publication_decision`, `not_produced`,
`realization_failed`, `failed`, `realized_identity`, `realized_zero`,
`realized`, and `superseded`. Disabled is not produced. Identity and zero use
the same candidate and decision sequence. Required failure propagates and
emits no complete product.

Every product must bind immutable parent, plan, operator, output, companion,
cause, lifecycle, failure, and provenance generations. Any core-defined
identity change creates a new generation. A later NOI attachment is a separate
companion and cannot mutate an existing FLT bundle.

## 8. Requirement conformance matrix

The following matrix routes every stable core requirement to a future
observable conformance decision. The exact normative text remains in the
imported core.

- `SCI-FLT-FIXED-REQ-001`: verify package, tranche, convolution-only base, and
  qualified low-pass subtype identity; reject arbitrary-linear or inference-
  bearing identity.
- `SCI-FLT-FIXED-REQ-002`: reconstruct `A` and verify no additive term or
  additive-state dependency.
- `SCI-FLT-FIXED-REQ-003`: verify exactly one supported immutable parent role.
- `SCI-FLT-FIXED-REQ-004`: verify complete applicable parent fields and atomic
  JINC roles.
- `SCI-FLT-FIXED-REQ-005`: verify upstream unavailability remains fail-closed.
- `SCI-FLT-FIXED-REQ-006`: verify observation, coadd, and JINC successor
  identities and absence of FLT coaddition or inferred commutation.
- `SCI-FLT-FIXED-REQ-007`: verify all plan and selector state predates payload
  arithmetic and is independent of response, covariance, noise, and NOI
  member payloads.
- `SCI-FLT-FIXED-REQ-008`: compare exact WCS and grid facts and reject
  approximate joins or resampling.
- `SCI-FLT-FIXED-REQ-009`: reconstruct exact finite coefficients, coordinate
  domain, three support sets, domains, discretization, and digest.
- `SCI-FLT-FIXED-REQ-010`: verify one-time convolution construction from the
  exact sampled kernel without intermediate or reordered claims.
- `SCI-FLT-FIXED-REQ-011`: verify every low-pass transfer fact or mark only the
  qualification unavailable.
- `SCI-FLT-FIXED-REQ-012`: recompute the full-footprint row set from exact
  `K_req` and predicates, including exact-zero stored coefficients.
- `SCI-FLT-FIXED-REQ-013`: verify all excluded rows are unavailable with
  causes, never promoted by storage shape.
- `SCI-FLT-FIXED-REQ-014`: detect and reject every deferred edge or replacement
  method.
- `SCI-FLT-FIXED-REQ-015`: verify transformed-amplitude identity and exact unit
  derivation without amplitude-category relabeling.
- `SCI-FLT-FIXED-REQ-016`: verify nominal-beam and CAL lineage retention and
  absence of unsupported beam or flux claims.
- `SCI-FLT-FIXED-REQ-017`: identify the response family, compare its one-time
  frozen-operator composition, or verify honest unavailability.
- `SCI-FLT-FIXED-REQ-018`: verify finite-domain transfer, mode, and phase facts
  or explicit unavailable states without whole-chain promotion.
- `SCI-FLT-FIXED-REQ-019`: compare influence to coefficients, verify it is not
  exposure, and verify exact exposure-lineage state.
- `SCI-FLT-FIXED-REQ-020`: verify distinct computability, support, admission,
  validity, confidence, and downstream states.
- `SCI-FLT-FIXED-REQ-021`: compare covariance to exact two-sided propagation.
- `SCI-FLT-FIXED-REQ-022`: verify separate parent-authority and output-
  representation axes, conditional model, exclusions, and unknown cross-term
  handling.
- `SCI-FLT-FIXED-REQ-023`: test induced off-diagonal terms and reject marginal
  planes presented as complete covariance.
- `SCI-FLT-FIXED-REQ-024`: verify empirical uncertainty and significance are
  absent from FLT ownership.
- `SCI-FLT-FIXED-REQ-025`: compare every NOI member's full operator state to
  the signal state.
- `SCI-FLT-FIXED-REQ-026`: detect and reject per-member selection or
  re-resolution.
- `SCI-FLT-FIXED-REQ-027`: verify every atomic role, exposure lineage, allowed
  honest unavailable record, and NOI attachment-state relation without an NOI
  completion dependency.
- `SCI-FLT-FIXED-REQ-028`: verify the complete candidate-before-publication
  lifecycle vocabulary, causes, failure, and generation bindings.
- `SCI-FLT-FIXED-REQ-029`: test distinct disabled, identity, and zero outcomes.
- `SCI-FLT-FIXED-REQ-030`: mutate each identity-defining fact in isolation and
  verify a new immutable generation; verify NOI nonmutation.
- `SCI-FLT-FIXED-REQ-031`: remove or conflict each required fact and verify
  fail-closed behavior without fallback.
- `SCI-FLT-FIXED-REQ-032`: evaluate distinct bundle and row admission request,
  applicability, eligibility, unavailable, exclusion, and cause branches.
- `SCI-FLT-FIXED-REQ-033`: evaluate the complete publication candidate,
  request-specific honest-unavailable, partial, disabled, identity, zero, and
  failed publication branches.
- `SCI-FLT-FIXED-REQ-034`: verify VAL only binds or evaluates an immutable
  approved policy and authors no FLT fact or arithmetic.
- `SCI-FLT-FIXED-REQ-035`: verify no generic downstream admission follows from
  product availability.
- `SCI-FLT-FIXED-REQ-036`: verify every excluded family and every Stage B
  nonclaim remains outside the product claim set.
- `SCI-FLT-FIXED-REQ-037`: freeze `J_full` once and verify identical selector
  reuse plus explicit exclusion of unsupplied selection uncertainty.
- `SCI-FLT-FIXED-REQ-038`: verify `K_geom`, `K_nonzero`, `K_req`, exact-zero,
  identity, and declared zero-operator support semantics.
- `SCI-FLT-FIXED-REQ-039`: evaluate all three typed policy objects and each
  request-specific companion qualification.
- `SCI-FLT-FIXED-REQ-040`: distinguish every response family and prevent local
  zero derivative from becoming an unsupported complete response claim.
- `SCI-FLT-FIXED-REQ-041`: verify exactly one sampled convolution application
  and absence of composition, collapse, or reordering claims.
- `SCI-FLT-FIXED-REQ-042`: verify preregistered numerical comparison fields,
  frozen bounds, independent oracle, and nonclaim boundary.
- `SCI-FLT-FIXED-REQ-043`: verify exact inherited exposure lineage and reject
  influence or convolved exposure as physical exposure.
- `SCI-FLT-FIXED-REQ-044`: verify external timing while retaining exact
  FLT-owned plan, selector, application, and publication ownership.

## 9. Falsifiable prediction suite

Each future prediction evaluation must bind exact inputs, operator state,
expected row domain, expected companions, expected lifecycle, comparison
semantics, and causes before execution. The normative expected outcomes are in
the imported core.

- `SCI-FLT-FIXED-PRED-001`: identity operator and identity lifecycle.
- `SCI-FLT-FIXED-PRED-002`: zero operator, inherited parent-support domain,
  local zero derivative and covariance, honest unavailable complete response,
  and zero lifecycle.
- `SCI-FLT-FIXED-PRED-003`: scalar input linearity on one frozen selector.
- `SCI-FLT-FIXED-PRED-004`: constant input under exact DC gain and every
  authorized normalization.
- `SCI-FLT-FIXED-PRED-005`: impulse, center, orientation, phase, and indexing.
- `SCI-FLT-FIXED-PRED-006`: exact separately typed response-family
  composition with one frozen operator application.
- `SCI-FLT-FIXED-PRED-007`: signed coefficients and distinct support summaries.
- `SCI-FLT-FIXED-PRED-008`: zero-sum constant-mode null without claim
  promotion.
- `SCI-FLT-FIXED-PRED-009`: full-footprint inclusion and dependent-row removal.
- `SCI-FLT-FIXED-PRED-010`: rejection of every deferred edge and replacement
  method, including support-conditioned renormalization.
- `SCI-FLT-FIXED-PRED-011`: missing, unavailable, non-admitted, and non-finite
  required parent locations.
- `SCI-FLT-FIXED-PRED-012`: exact complete covariance on ordered `S_out`.
- `SCI-FLT-FIXED-PRED-013`: induced cross-row covariance conditional on an
  explicit independent-diagonal parent model.
- `SCI-FLT-FIXED-PRED-014`: unavailable parent response or covariance and
  rejection of surrogates.
- `SCI-FLT-FIXED-PRED-015`: exact WCS and grid mismatch failure.
- `SCI-FLT-FIXED-PRED-016`: distinct observation, coadd, and JINC identities
  with no FLT coadd product.
- `SCI-FLT-FIXED-PRED-017`: exact frozen-selector NOI member parity and
  member-level unavailable footprint state.
- `SCI-FLT-FIXED-PRED-018`: per-member re-resolution rejection.
- `SCI-FLT-FIXED-PRED-019`: disabled, identity, zero, and failure lifecycle
  distinctions.
- `SCI-FLT-FIXED-PRED-020`: upstream unavailable MAP and JINC route.
- `SCI-FLT-FIXED-PRED-021`: low-pass qualification completeness.
- `SCI-FLT-FIXED-PRED-022`: exact-zero stored-coefficient required dependency.
- `SCI-FLT-FIXED-PRED-023`: zero-operator inherited row-support domain.
- `SCI-FLT-FIXED-PRED-024`: independent exact sampled low-pass transfer on the
  declared frequency grid and metric.

## 10. Numerical-comparison evidence record

A future candidate comparison must bind one immutable policy before results are
observed. The evidence record must identify its independent exact or high-
precision oracle; absolute, relative, and near-zero rules; signed-cancellation
and zero-sum cases; conditioning and operation-count dependence; covariance
comparison; sequential and parallel agreement; overflow, underflow, and non-
finite handling; simultaneous row decisions; lifecycle; and provenance.

The record must prove that bounds were not changed after a failure and that
scientific coefficients and identity remained exact. Passing such a policy
would be future conformance evidence only. This specification supplies no
candidate result, numerical-adequacy finding, validation, or performance claim.

## 11. Traceability, identity, and reproducibility record

`TRACEABILITY.json` maps every requirement and prediction identifier to its
normative-core section, scientist-rationale section, engineering-conformance
section, exact admitted Stage A object sections, and r0.2 owner-directive
sections where the revision changes semantics. The durable verifier must reject
missing, duplicate, extra, malformed, or unadmitted trace entries.

`BUILD_BINDING.json` records the packet manifest binding, all 17 admitted
object hashes and byte counts, exact r0.2 owner-directive binding, every Stage
B source and build-tool hash, deterministic build environment, output PDF
identities, page counts, byte sizes, and SHA-256 digests. The source-closure,
owner-decision parity, core/view parity, semantic-change, profile, rebuild, and
visual-QA records are required. A clean separate rebuild must reproduce the
three PDF digests exactly.

PDF text must expose the document identity, exact source digest, exact shared
core digest, exact packet-manifest digest, and exact builder digest. PDF page
rendering and visual QA are required in addition to content and hash checks.

## 12. Nonclaims

This conformance specification does not identify, inspect, modify, or judge an
implementation. It records no implementation conformity, algorithm change,
validation result, calibration state, achieved response or covariance,
numerical adequacy, performance, readiness, scientific freeze, production
status, or Unity activity.
