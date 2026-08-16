# SCI-MAP — Ordinary Mapmaking And Observation Coaddition Scope Brief

Status: `draft`; scientific-owner review required

Scientific owner: Grant Wilson

Version/date: `v0.1 draft`, `2026-08-16`

Approved source identifier: pending owner approval

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: Codex manager, `2026-08-16`; owner review pending
- Existing material proposed for adoption: the bounded owner decisions for
  atomic map-bundle admission, ordinary nonprecision normalization, centered
  integer coaddition, distinct support/validity facts, and immutable raw-parent
  identity
- Existing material proposed for citation: frozen implementation-independent
  `SCI-MAP-001_INDEPENDENT_CORE.tex`, content SHA-256
  `13dd5922bd492e381afcc3b015284216dde1ccc2199ece3d070ee577c7324381`
- Existing material abstracted: stable identity, frame, unit, validity,
  lifecycle, and package-ownership conventions
- Existing material deferred or excluded: MAP-001 audits/repairs/evidence;
  the separate MAP-002 JINC and MAP-003 OOF-transfer packages; all
  implementation, tests, reductions, validation, Unity, conformity, and
  production status
- Genuinely new scientific work: consolidate and explain the already-derived
  ordinary estimator under the later owner specializations; define its
  conditional response/covariance/output boundary; expose unresolved
  dependency and policy questions without inventing answers
- Proposed author references: this brief; the frozen independent core with
  [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md); and
  [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)
- Author-packet exclusions: [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md),
  this full recovery record, current code/interfaces, audits, repairs, tests,
  validation, MAP-002/MAP-003 evidence, and current production state

Confirm that this opening was reviewed before launching scientific authorship:
`no — owner approval is the next gate`.

## 1. Package Name And Scientific Purpose

**SCI-MAP — Ordinary Mapmaking and Observation Coaddition** defines how an
admitted collection of calibrated, conditioned Stokes-I detector samples is
transformed into a scientifically interpretable observation map and how
compatible observation-map bundles are combined on one common grid.

The package must preserve the estimand, units, response, conditional
uncertainty, identity, WCS, support, validity, and provenance needed by noise,
filtering, fitting, Beammap, mode, and feedback consumers. It must not allow a
normalization coefficient, a finite number, or a compatibility alias to acquire
a stronger scientific meaning than its producer established.

## 2. Scientific Boundary

The proposed v0.1 operation begins with:

- calibrated ordinary-`xs` detector samples supplied under SCI-CAL, arranged
  on an externally supplied common sample axis;
- explicit observation, array, network, map-group, Stokes, and input-column
  identity;
- upstream sample/detector eligibility and non-finite state;
- an externally supplied sample-to-map coordinate relation, declared frame,
  WCS, pixel basis, and map extent;
- a finite analysis/gridding coefficient with declared unit, normalization
  scope, lifecycle, applied factors, and statistical status;
- a sample-duration or other explicitly named exposure quantity;
- an admitted response/kernel tracer or an explicit unavailable response
  state; and
- an immutable request for an ordinary supported map and, when applicable,
  observation coaddition.

It ends with:

- a complete observation-map bundle; and
- when requested and scientifically admissible, a complete coadd-map bundle
  formed from atomically admitted compatible observation bundles by centered
  integer common-grid placement.

V0.1 covers the ordinary positive-coefficient normalized estimator. It does
not cover the signed JINC estimator, maximum-likelihood mapmaking, general
reprojection, map filtering, empirical noise estimation, fitting, Beammap
inference, OOF residual-transfer estimation, or iterative feedback.

The ordinary mathematical operator may be reused by Pointing/OOF or another
mode only when its input bundle satisfies this contract. Mode-specific
scientific interpretation, fitting, and consumer validity remain outside
SCI-MAP. The v0.1 coaddition authority is the compatible common-grid
observation-coadd operation; it does not authorize a mosaic or resampling
system.

## 3. Legitimate Inputs

Legitimate inputs may include:

- a finite sample-by-detector signal matrix in one declared unit, with exact
  sample/column membership and no mixed undeclared signal meaning;
- one externally supplied eligibility state per candidate contribution,
  including flag and non-finite policy;
- array, network, observation, group, and Stokes identity that is explicit
  rather than inferred from container position;
- detector/acquisition correspondence sufficient for the upstream producers
  to bind each signal, coefficient, and response input without claiming a
  persistent universal detector ID;
- per-sample coordinates or an equivalent forward projection in a declared
  frame and pixel basis, including the full-precision WCS authority;
- finite positive ordinary gridding coefficients, with zero/invalid states and
  statistical meaning declared by their producer;
- a nonnegative finite exposure contribution in a named unit and accounting
  convention;
- a sample-domain response or template tracer with source, normalization,
  unit, and processing-parent identity, or a typed unavailable state;
- requested/effective/observation-resolved method, grouping, geometry,
  support-policy, unit, and coadd state; and
- for coaddition, complete immutable observation-map bundles with compatible
  signal, estimator, response, unit, frame/WCS, shape, support, validity,
  parentage, and policy identity.

The active validated map signal boundary is `mJy/beam`. Acceptance of other
configuration tokens does not authorize their conversion or downstream
meaning. Other map units require separately approved CAL/beam/pixel/spectral
authority and an explicit SCI-MAP version decision.

## 4. Required Outputs

The contract must define, without merely copying the current file layout:

1. the normalized map signal in its exact declared unit and estimand;
2. the pre-normalized numerator and normalization/coefficient identity needed
   to interpret or reconstruct the ordinary result;
3. the response/kernel companion produced with the same admitted membership,
   coefficients, projection, normalization, boundary, and coadd operator, or
   an explicit unavailable response state;
4. conditional covariance or formal-weight information with every
   independence, coefficient-calibration, and omitted-correlation assumption
   stated, and no automatic precision or significance claim;
5. distinct hit, exposure, numerical-support, science-policy-support, and
   final-validity meanings;
6. one authoritative raw science-validity state that also requires admitted
   bundle identity and finite required companions;
7. exact observation/coadd role, array/group/Stokes identity, signal unit,
   frame, WCS, pixel shape/indexing, estimator, response, parentage, and
   product identity;
8. requested, effective, observation-resolved, and realized state sufficient
   to identify the operator actually applied;
9. atomic failure when a required input, compatibility check, aggregate,
   coordinate/index conversion, companion, or required publication is
   invalid; and
10. raw-parent identity that downstream filters and consumers cannot rewrite
    or use to promote a raw-invalid pixel.

The owner-approved logical v0.1 map facts are:

- geometric projected incidences;
- estimator-admitted contributions;
- admitted observation count for coadds;
- upstream-eligible exposure;
- retained exposure;
- numerical normalization support;
- separate science-policy support; and
- authoritative final raw science validity.

The science rationale must explain why these facts differ. The engineering
view may bind them to canonical product identities after owner approval.

## 5. Upstream And Downstream Responsibilities

- **SCI-CAL** produces calibrated ordinary-`xs` samples, their unit,
  calibration quality/validity, conditional uncertainty transfer, response
  basis, and lineage. SCI-MAP consumes those meanings without reinterpreting
  `flxscale`, atmosphere, passband, beam, or target/source association.
- **ALIGN/AST** produces the common sample axis, eligible coordinate relation,
  frame/WCS, and astrometric uncertainty. SCI-MAP applies the admitted
  projection and may not choose pointing, interpolate missing coordinates, or
  assert absolute astrometric performance.
- **PTC** produces conditioned samples and the analysis coefficient/covariance
  identity. SCI-MAP may propagate a declared covariance but may not call a
  coefficient precision without the required PTC and correlation evidence.
- **VAL** produces sample/detector eligibility, flag precedence, and
  non-finite policy. SCI-MAP owns its estimator-specific support and final map
  validity but may not redefine upstream validity.
- **SCI-MAP** owns the ordinary sample-to-map transformation, its response and
  conditional covariance statement, complete raw map bundle, compatibility
  admission, and ordinary observation coaddition.
- **NOI** owns noise-realization construction, empirical covariance and weight,
  and statistical-significance calibration. SCI-MAP may apply its fixed
  operator to an admitted realization but does not define the realization
  distribution.
- **FLT** owns map filtering and its local support, response, covariance, and
  output validity. It must preserve the immutable SCI-MAP raw validity and
  parent identity.
- **SRC/MODE/BEAM** own detection, fitting, Pointing/OOF interpretation,
  Beammap inference, fit uncertainty, and consumer-specific validity.
- **FRUIT** owns feedback projection, iteration identity, recurrence,
  convergence, and restart. SCI-MAP v0.1 is a one-pass operator and does not
  define iterative response.
- **MAP-002/JINC** owns signed deposition, subpixel JINC response,
  conditioning, coverage, and JINC-specific validity/product availability.
- **MAP-003/OOF transfer** owns the residual transfer estimator, exact
  tracer/parent identity, frequency-domain validity, and LMTOOF consumer
  boundary.

## 6. Externally Imposed Conventions

The following recovered decisions are proposed as binding unless Grant reopens
them:

1. **Ordinary normalized estimator.** For admitted finite positive
   coefficients `u_i`, accumulate `Q=sum(u_i)`,
   `N=sum(u_i x_i)`, and the corresponding
   `K=sum(u_i k_i)`; where the support rule permits and `Q` is finite and
   positive, publish `x_hat=N/Q` and `k_hat=K/Q`.
2. **Nonprecision default.** `Q` and the published ordinary coefficient are
   normalization facts. They equal inverse variance only when the producer has
   established the necessary coefficient-calibration and covariance
   conditions.
3. **Constant preservation.** With fixed membership and coefficients, the
   ordinary positive-coefficient estimator is a convex weighted mean and
   preserves a constant input on valid support. This does not establish
   compact-source, extended-source, integrated, or absolute-offset response.
4. **Same operator for companions.** Signal, response/kernel tracer, declared
   linear realizations, retained exposure, and coadd observation count use the
   same admitted membership and geometric placement where their definitions
   require it.
5. **Atomic coadd admission.** An observation enters a coadd as one complete
   immutable ordered bundle. Any incompatible identity, unit, response,
   frame/WCS, shape, policy, or required companion rejects the observation
   before coadd-owned state changes.
6. **Centered integer common-grid placement.** Observation and coadd shapes
   must differ by nonnegative even row/column counts; their reference pixels
   identify the same world coordinate after the corresponding integer offset.
   No fractional shift, reprojection, interpolation, wrap, or implicit
   recentering is authorized.
7. **No signal centering.** The signal-centering operator is `L=I`.
   Coaddition does not subtract a mean, remove a null mode, or recenter a
   source.
8. **Validity is explicit.** A finite value, positive coefficient, exposure,
   support plane, or compatibility alias cannot independently establish raw
   science validity. An invalid contribution is excluded before its numerical
   payload is evaluated.
9. **Distinct v0.1 facts.** The eight hit/exposure/support/validity facts in
   section 4 retain distinct meanings. A compatibility alias does not become
   a validity or precision authority.
10. **Adopted support policy.** The recovered ordinary v0.1 contract uses
    separate normalization and science-policy support thresholds selected from
    finite strictly positive coefficients. For `N` sorted values, the
    zero-based index is
    `k=floor((floor(0.75*N)+N)/2)`; empty input has threshold zero.
    Normalization uses the selected value times `coverage_cut/10`;
    science-policy support uses the selected value times `coverage_cut`;
    both require an explicit finite-positive coefficient at or above the
    relevant threshold.
11. **WCS and persistence.** Full-precision typed/sidecar WCS is the lossless
    admission/provenance authority. A physical FITS WCS may differ by at most
    0.1 arcsec maximum sky separation while preserving exact axis sign,
    handedness, orientation, and centered-integer shape/reference-pixel
    relations. This tolerance does not authorize a different map.
12. **Failure and parentage.** Unrepresentable finite aggregates or projected
    indices fail before live state changes. Required observation products
    remain required when coaddition is enabled. Downstream processing
    preserves raw validity and parent identity without promotion or mutation.
13. **Capability.** V0.1 is Stokes I only. Map, array, network, group, detector,
    and container identities are distinct. FITS coordinates are one-based;
    in-memory pixel indices are zero-based.
14. **Frames and units.** Science maps use their declared equatorial J2000 TAN
    WCS with degree-valued spatial axes. Point/OOF maps use declared AltAz
    tangent-plane offsets in arcseconds. Signal and response/kernel units
    follow the admitted calibrated map quantity; the active boundary is
    `mJy/beam`.

The support algorithm in item 10 is an adopted v0.1 policy. Its physical
rationale and the authority/evidence required to change it remain explicit
owner questions; the author must not invent them.

## 7. Questions The Contract Must Answer

### A. Estimand, Projection, And Response

- What exact sky quantity does the ordinary map estimate, conditional on the
  admitted CAL and PTC input meanings?
- Which sample-to-pixel coefficients define the ordinary positive gridder, and
  when is the result a one-hot weighted mean versus fractional projection?
- What is the forward response operator for constant, delta/basis-pixel,
  point-source template, resolved, gradient/Fourier, variable-coverage, and
  edge inputs?
- When may a stored kernel/tracer be called the realized map response, and what
  must remain unavailable when its sample-domain parent is incomplete?
- Which map quantities retain an absolute offset, and what downstream claims
  fail when an upstream stage has removed or altered that mode?

### B. Coefficient, Covariance, And Uncertainty

- What are the exact conditional covariance and cross-pixel covariance of the
  ordinary estimator for general admitted sample covariance?
- Under precisely which projection, coefficient, independence, and covariance
  conditions does the normalization equal marginal inverse variance?
- How should MAP represent missing full covariance, imperfect coefficient
  calibration, cross-observation covariance, calibration/response nuisance
  terms, and data-derived membership?
- What coadd estimator is realized by ordinary positive observation
  coefficients, and what stronger conditions would be required for
  inverse-variance or correlated-GLS interpretation?
- Which uncertainty products are MAP-owned transformations, which are merely
  carried from upstream, and which remain NOI-owned empirical results?

### C. Identity, Support, Validity, And Failure

- What is the minimum scientific identity that makes an observation map and a
  coadd bundle unambiguous without relying on container order or a universal
  detector UID?
- What exact contribution, exposure, normalization-support,
  science-policy-support, and final-validity predicates apply, and which owner
  supplies each input fact?
- What is the scientific rationale for the adopted v0.1 threshold rule, who
  may change it, and what evidence would justify a successor?
- How are zero coefficient, zero support, flagged, non-finite, out-of-bounds,
  incompatible-WCS, missing-companion, overflow, and unrepresentable-index
  states distinguished?
- Which failures invalidate one contribution, one pixel, one observation
  admission, or the complete required product?

### D. Units, WCS, Lifecycle, And Consumers

- Which CAL/beam/pixel assumptions are required for `mJy/beam`, and which
  other accepted configuration tokens remain scientifically unavailable?
- What response and covariance meaning survives centered-integer placement at
  a finite coadd boundary?
- What requested, effective, observation-resolved, and realized fields are
  minimally sufficient to reconstruct the operator and its parentage?
- Which raw MAP facts must persist or be resolvable as typed interfaces, and
  which current compatibility aliases are unnecessary to the abstract
  contract?
- What exact bundle may NOI, FLT, SRC, MODE, BEAM, and FRUIT consume, and which
  amplitude, astrometric, precision, significance, or production claims remain
  fail-closed?

### E. Falsifiable Predictions

The contract must derive predictions for at least:

- a constant input with uniform and unequal positive coefficients;
- one-pixel and fractional-projection hand calculations;
- delta/basis-pixel, point-source-template, extended, gradient/Fourier, and
  finite-edge inputs;
- zero/low/negative/non-finite coefficients and exact/near-zero normalization;
- flagged, non-finite, out-of-bounds, missing-companion, and incompatible-
  identity inputs with no partial mutation;
- one observation, unequal observation coefficients, a missing/invalid
  observation, and compatible centered integer placement;
- cross-observation covariance showing when summed coefficients are not
  precision;
- same fixed operator applied to an admitted noise realization;
- sequential/parallel execution under a preregistered floating-point policy,
  with exact integer facts; and
- deliberate attempts to relabel formal standardized signal as significance,
  promote a raw-invalid pixel downstream, or apply an ordinary predicate to
  JINC.

## 8. Non-Goals

SCI-MAP v0.1 does not:

- audit or repair Citlali, inspect a candidate for conformity, optimize
  mapmaking, run tests/reductions, request Unity work, or change production;
- derive the ordinary weighted normalization a second time;
- establish CAL, ALIGN/AST, PTC, VAL, or NOI science on their behalf;
- authorize JINC, maximum-likelihood, reprojection/mosaicking, or a signed
  deposition rule;
- define OOF residual transfer, LMTOOF consumption, source fitting, Pointing
  or Beammap inference, filtering, empirical noise, significance, or feedback;
- authorize Stokes Q/U, measured-R mapmaking, or a non-`mJy/beam` unit;
- claim that the normalization coefficient is inverse variance or that formal
  standardized signal is empirical significance;
- claim observational photometric, astrometric, response, noise, or production
  performance from algebraic correctness; or
- retroactively relabel historical products with the new package authority.

## 9. Allowed References

Proposed for Grant's approval:

1. this Scope Brief after owner approval and content hashing;
2. `SCI-MAP-001_INDEPENDENT_CORE.tex` at
   `c28f18ed089657dae278caba2d6d6d65c7ec72f4`, content SHA-256
   `13dd5922bd492e381afcc3b015284216dde1ccc2199ece3d070ee577c7324381`,
   accompanied by the owner-approved
   [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md); and
3. the content-hashed, owner-approved
   [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)
   containing only stable identity, frame, unit, validity/lifecycle, and
   producer/transformer/consumer boundaries.

The independent core's associated audit is not admitted. The later method
note is not admitted because it duplicates the core. The raw ADR, owner
amendment, integration records, executable contracts, implementation source,
tests, validation, MAP-002/MAP-003 records, and internal dossier are excluded.

No other source may enter authorship without Grant's approval and a revised,
re-frozen brief or an explicit owner answer.

## 10. Owner Decisions And Remaining Ambiguities

### Recovered decisions proposed for confirmation

Grant is asked to confirm:

1. **MAP-SCOPE-D001 — included estimator.** V0.1 covers ordinary
   positive-coefficient normalized mapmaking plus compatible centered-integer
   observation coaddition and the shared raw map bundle.
2. **MAP-SCOPE-D002 — separate methods.** JINC/MAP-002, maximum-likelihood
   mapmaking, and MAP-003 OOF residual transfer remain separate; none inherits
   an ordinary positive-coefficient predicate.
3. **MAP-SCOPE-D003 — reuse.** The frozen SCI-MAP-001 independent core is the
   reusable mathematical basis. The author consolidates and explains it under
   the later decisions rather than rederiving it.
4. **MAP-SCOPE-D004 — coefficient and coadd meaning.** The coefficient is
   nonprecision by default; atomic whole-bundle admission, centered integer
   placement, and `L=I` are binding v0.1 choices.
5. **MAP-SCOPE-D005 — facts and validity.** The eight logical map facts,
   separate normalization/science-support policies, explicit final validity,
   immutable raw parent, and fail-before-mutation rules are adopted. The
   threshold formula is policy authority; its scientific rationale and change
   authority remain open rather than invented.
6. **MAP-SCOPE-D006 — author packet.** Approve the three-part packet in
   section 9 and the information-firewall exclusions.

### Remaining scientific-owner questions

- Record the authority for the physical rationale and future change control of
  the adopted support thresholds.
- Decide whether v0.1 should require a response/kernel product or permit a
  scientifically usable map with a typed unavailable response state for
  restricted consumers.
- Decide the minimum covariance representation that must persist versus be
  resolvable through lineage.
- Decide whether the abstract contract requires observation-map publication
  whenever coaddition is requested, beyond the already adopted current
  required-output rule.
- Confirm whether Point/OOF ordinary-map arithmetic is a registered use of the
  shared operator while mode-specific meaning remains outside v0.1.
- Resolve upstream CAL, ALIGN/AST, PTC, VAL, and NOI facts only when a stronger
  MAP claim first depends on them; do not infer them from current code.

Approval of D001--D006 authorizes scope and packet preparation only. It does
not approve the resulting scientific contract, implementation, validation, or
production use.

## 11. Independence Statement

This brief defines the scientific problem, recovered authority, package
boundary, conditional dependencies, questions, and predicted cases without
prescribing current Citlali behavior as the answer. It was sanitized from the
separate internal dossier.

The scientific author will receive only the owner-approved brief, the exact
references in section 9, and later owner answers. The author will not receive
Citlali source, executable product/config contracts, audits, findings, repairs,
tests, reductions, validation results, Unity material, current conformity
status, or the separate MAP-002/MAP-003 evidence.
