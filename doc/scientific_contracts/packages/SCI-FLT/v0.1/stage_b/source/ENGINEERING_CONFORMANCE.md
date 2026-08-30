# SCI-FLT-FIXED v0.1 Engineering-Conformance Specification

Document identity: `SCI-FLT-FIXED-ENGINEERING-CONFORMANCE v0.1/draft-r0.1`

Status: implementation-blind Stage B conformance-target draft; no conformity finding

Normative import: the complete
`SCI-FLT-FIXED-NORMATIVE-CORE v0.1/draft-r0.1`, source SHA-256
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
and resolved states. A resolved plan must bind every coefficient, parameter,
grid and metric fact, normalization, support and edge rule, transfer
qualification, and provenance fact before any parent or NOI member is applied.

## 3. Operator reconstruction record

`FLT-OPERATOR` must provide a content-bound representation sufficient to
reconstruct the exact finite `J_full L_Theta`, including:

- ordered input and output row domains;
- exact WCS, frame, topology, metric, shape, indexing, and pixel-area facts;
- complete sampled coefficients, coefficient units and ordering, numerical
  representation, and digest;
- kernel offset set, center, extent, tie, phase, subpixel convention,
  orientation, and handedness;
- declared normalization, DC gain, and distinct signed, absolute, squared, and
  geometric support summaries;
- full-footprint predicates and cause vocabulary;
- qualified transfer facts or explicit unavailability; and
- immutable plan and operator generation identifiers.

An alternate computational mechanism is comparable only after reconstructing
the identical declared operator. A different coefficient representation that
changes a declared coefficient, ordering, phase, row domain, or numerical
operator is not scientific equivalence.

## 4. Admission and application gate

Input admission must evaluate the draft
`SCI-FLT-FIXED:input_admission@1` semantics against one exact request, parent,
and resolved plan. It must return distinct request, applicability, eligibility,
unavailable-decision, and cause state. It performs no transformation.

Application may proceed only for an eligible resolved route. For every output
row, the row selector must verify all required kernel locations against the
exact parent domain, FLT input admission, payload availability, finiteness, and
all required predicates. The scientific row domain must equal the set of rows
that pass every check.

No extension, periodic wrap, truncation, local renormalization, inpainting,
reflection, clamp, mirror, padding-based admission, edge completion, or value
replacement is conforming in v0.1. Rows failing the full-footprint gate remain
unavailable with causes, even if a storage payload is finite.

## 5. Signal, response, transfer, mode, and influence records

Application must compute `FLT-SIG` from the exact resolved operator on the
exact scientific row ordering. The output-unit record must derive units from
parent and coefficient units and retain the originating nominal-beam and
calibration lineage without introducing a filtered-beam label.

When an exact compatible parent response is available, the response record must
equal the identical operator applied to that response on the identical row
domain. Otherwise the record must be explicitly unavailable. Kernel-only,
approximate, differently centered, differently normalized, or differently
edged response surrogates do not pass.

Transfer and mode records must bind the exact finite-grid domain, phase,
normalization, null, invariant, and attenuation facts that are scientifically
defined. Unsupported facts must remain unavailable. Influence must expose the
parent-row coefficient relation and must remain separate from physical
exposure.

## 6. Covariance and NOI records

For an available compatible parent covariance, a covariance-qualified candidate
must realize the exact two-sided operator propagation on `S_out`. Its record
must state whether it is complete, diagonal-input propagated, structured,
partial, or marginal; name all omitted terms; and expose domain, ordering,
rank, null space, and supported operations where applicable.

If an explicitly diagonal parent covariance is propagated, any complete output
representation must include induced cross-row covariance. A marginal-only
representation must say so. Missing cross terms may not be encoded as zero or
independence.

An optional NOI attachment must be separately owned and immutably bound to the
exact FLT product. Every admitted NOI member must receive the identical
operator state and row selection as the signal. The route must reject filtering
of uncertainty-like products, approximate transfer, relocation, commutation,
substitution, and any per-member re-resolution.

## 7. Atomic publication and lifecycle records

Publication must evaluate the draft
`SCI-FLT-FIXED:output_publication@1` semantics against one complete bundle. The
bundle must contain all required roles from the imported core, including honest
unavailable companion records where permitted. A partial bundle, placeholder,
or inferred companion fails required publication.

Lifecycle evidence must distinguish `not_requested`, `requested`, `effective`,
`disabled`, `unavailable`, `resolved`, `applied`, `failed`,
`realized_identity`, `realized_zero`, `realized`, and `superseded`. Disabled
emits no product. Identity and zero are realized transformations. Required
failure propagates and emits no complete product.

Every product must bind immutable parent, plan, operator, output, companion,
cause, lifecycle, failure, and provenance generations. Any core-defined
identity change creates a new generation. A later NOI attachment is a separate
companion and cannot mutate an existing FLT bundle.

## 8. Requirement conformance matrix

The following matrix routes every stable core requirement to a future
observable conformance decision. The exact normative text remains in the
imported core.

- `SCI-FLT-FIXED-REQ-001`: verify package and tranche identity; reject generic
  or inference-bearing identity.
- `SCI-FLT-FIXED-REQ-002`: reconstruct `A` and verify no additive term or
  additive-state dependency.
- `SCI-FLT-FIXED-REQ-003`: verify exactly one supported immutable parent role.
- `SCI-FLT-FIXED-REQ-004`: verify complete applicable parent fields and atomic
  JINC roles.
- `SCI-FLT-FIXED-REQ-005`: verify upstream unavailability remains fail-closed.
- `SCI-FLT-FIXED-REQ-006`: verify observation, coadd, and JINC successor
  identities and absence of FLT coaddition or inferred commutation.
- `SCI-FLT-FIXED-REQ-007`: verify all operator state predates application and
  is independent of parent and member payloads.
- `SCI-FLT-FIXED-REQ-008`: compare exact WCS and grid facts and reject
  approximate joins or resampling.
- `SCI-FLT-FIXED-REQ-009`: reconstruct exact finite coefficients, domains,
  discretization, and digest.
- `SCI-FLT-FIXED-REQ-010`: verify convolution construction from the exact
  sampled kernel and offset set.
- `SCI-FLT-FIXED-REQ-011`: verify every low-pass transfer fact or mark only the
  qualification unavailable.
- `SCI-FLT-FIXED-REQ-012`: recompute the full-footprint row set from exact
  predicates.
- `SCI-FLT-FIXED-REQ-013`: verify all excluded rows are unavailable with
  causes, never promoted by storage shape.
- `SCI-FLT-FIXED-REQ-014`: detect and reject every deferred edge or replacement
  method.
- `SCI-FLT-FIXED-REQ-015`: verify transformed-amplitude identity and exact unit
  derivation without amplitude-category relabeling.
- `SCI-FLT-FIXED-REQ-016`: verify nominal-beam and CAL lineage retention and
  absence of unsupported beam or flux claims.
- `SCI-FLT-FIXED-REQ-017`: compare response output to exact operator
  composition or verify honest unavailability.
- `SCI-FLT-FIXED-REQ-018`: verify finite-domain transfer, mode, and phase facts
  or explicit unavailable states without whole-chain promotion.
- `SCI-FLT-FIXED-REQ-019`: compare influence to coefficients and verify it is
  not labeled exposure.
- `SCI-FLT-FIXED-REQ-020`: verify distinct computability, support, admission,
  validity, confidence, and downstream states.
- `SCI-FLT-FIXED-REQ-021`: compare covariance to exact two-sided propagation.
- `SCI-FLT-FIXED-REQ-022`: verify covariance representation type, omissions,
  domain, and unknown cross-term handling.
- `SCI-FLT-FIXED-REQ-023`: test induced off-diagonal terms and reject marginal
  planes presented as complete covariance.
- `SCI-FLT-FIXED-REQ-024`: verify empirical uncertainty and significance are
  absent from FLT ownership.
- `SCI-FLT-FIXED-REQ-025`: compare every NOI member's full operator state to
  the signal state.
- `SCI-FLT-FIXED-REQ-026`: detect and reject per-member selection or
  re-resolution.
- `SCI-FLT-FIXED-REQ-027`: verify every atomic role and allowed honest
  unavailable record.
- `SCI-FLT-FIXED-REQ-028`: verify the complete lifecycle vocabulary, causes,
  failure, and generation bindings.
- `SCI-FLT-FIXED-REQ-029`: test distinct disabled, identity, and zero outcomes.
- `SCI-FLT-FIXED-REQ-030`: mutate each identity-defining fact in isolation and
  verify a new immutable generation; verify NOI nonmutation.
- `SCI-FLT-FIXED-REQ-031`: remove or conflict each required fact and verify
  fail-closed behavior without fallback.
- `SCI-FLT-FIXED-REQ-032`: evaluate all input-admission request,
  applicability, eligibility, unavailable, exclusion, and cause branches.
- `SCI-FLT-FIXED-REQ-033`: evaluate complete, honest-unavailable, partial,
  disabled, identity, zero, and failed publication branches.
- `SCI-FLT-FIXED-REQ-034`: verify VAL only binds or evaluates an immutable
  approved policy and authors no FLT fact or arithmetic.
- `SCI-FLT-FIXED-REQ-035`: verify no generic downstream admission follows from
  product availability.
- `SCI-FLT-FIXED-REQ-036`: verify every excluded family and every Stage B
  nonclaim remains outside the product claim set.

## 9. Falsifiable prediction suite

Each future prediction evaluation must bind exact inputs, operator state,
expected row domain, expected companions, expected lifecycle, comparison
semantics, and causes before execution. The normative expected outcomes are in
the imported core.

- `SCI-FLT-FIXED-PRED-001`: identity operator and identity lifecycle.
- `SCI-FLT-FIXED-PRED-002`: zero operator, zero signal, honest unavailable
  companions, and zero lifecycle.
- `SCI-FLT-FIXED-PRED-003`: scalar input linearity.
- `SCI-FLT-FIXED-PRED-004`: constant input under exact DC gain and every
  authorized normalization.
- `SCI-FLT-FIXED-PRED-005`: impulse, center, orientation, phase, and indexing.
- `SCI-FLT-FIXED-PRED-006`: exact compatible parent-response composition.
- `SCI-FLT-FIXED-PRED-007`: signed coefficients and distinct support summaries.
- `SCI-FLT-FIXED-PRED-008`: zero-sum constant-mode null without claim
  promotion.
- `SCI-FLT-FIXED-PRED-009`: full-footprint inclusion and dependent-row removal.
- `SCI-FLT-FIXED-PRED-010`: rejection of every deferred edge and replacement
  method, including support-conditioned renormalization.
- `SCI-FLT-FIXED-PRED-011`: missing, unavailable, non-admitted, and non-finite
  required parent locations.
- `SCI-FLT-FIXED-PRED-012`: exact complete covariance on ordered `S_out`.
- `SCI-FLT-FIXED-PRED-013`: induced cross-row covariance from diagonal input.
- `SCI-FLT-FIXED-PRED-014`: unavailable parent response or covariance and
  rejection of surrogates.
- `SCI-FLT-FIXED-PRED-015`: exact WCS and grid mismatch failure.
- `SCI-FLT-FIXED-PRED-016`: distinct observation, coadd, and JINC identities
  with no FLT coadd product.
- `SCI-FLT-FIXED-PRED-017`: exact fixed-state NOI member parity.
- `SCI-FLT-FIXED-PRED-018`: per-member re-resolution rejection.
- `SCI-FLT-FIXED-PRED-019`: disabled, identity, zero, and failure lifecycle
  distinctions.
- `SCI-FLT-FIXED-PRED-020`: upstream unavailable MAP and JINC route.
- `SCI-FLT-FIXED-PRED-021`: low-pass qualification completeness.

## 10. Traceability, identity, and reproducibility record

`TRACEABILITY.json` maps every requirement and prediction identifier to its
normative-core section, scientist-rationale section, engineering-conformance
section, and exact admitted Stage A object sections. The durable verifier must
reject missing, duplicate, extra, malformed, or unadmitted trace entries.

`BUILD_BINDING.json` records the packet manifest binding, all 17 admitted
object hashes, every Stage B source and build-tool hash, deterministic build
environment, output PDF identities, page counts, byte sizes, and SHA-256
digests. A clean rebuild in a separate temporary directory must reproduce the
three PDF digests exactly.

PDF text must expose the document identity, exact source digest, exact shared
core digest, exact packet-manifest digest, and exact builder digest. PDF page
rendering and visual QA are required in addition to content and hash checks.

## 11. Nonclaims

This conformance specification does not identify, inspect, modify, or judge an
implementation. It records no implementation conformity, algorithm change,
validation result, calibration state, achieved response or covariance,
numerical adequacy, performance, readiness, scientific freeze, production
status, or Unity activity.
