# SCI-CAL v0.1 Author Draft Decisions

Status: independent author proposals, not owner approval

Date: 2026-08-16

These decisions were derived only from the approved four-item author packet.
They make the contract coherent without importing implementation behavior or
inventing missing TolTEC-specific facts.

## Derived draft decisions

### SCI-CAL-AUTH-D001 - Endpoint and typed scope

The v0.1 estimator begins with admitted ordinary `xs` detector occurrences and
ends with calibrated detector samples, conditional uncertainty where available,
a validity/quality tuple, nuisance-completeness state, response basis, and
canonical lineage. Its only calibrated target is top-of-atmosphere
point-source-peak `mJy/beam`. Maps and photometric estimators are downstream.

### SCI-CAL-AUTH-D002 - Selected APT as the absolute-factor boundary

The selected immutable measured APT row's oriented `flxscale` is the sole
absolute detector factor. The realized signal multiplier contains selected
`flxscale`, one target-observation atmosphere correction, and an explicit unity
target-unit factor. `responsivity`, `sens`, parent factors, and opaque `fcf` do
not enter as additional absolute signal factors.

### SCI-CAL-AUTH-D003 - Pointing-derived correction is ancestry, not a repeat factor

A selected lineage classifies a TolProj pointing-derived correction as
`not_applicable`, `embodied_once`, or `required_but_unresolved`. When embodied,
the selected factor is the corrected child value and the parent/correction are
lineage only. When required but unresolved, calibrated output is unavailable.

### SCI-CAL-AUTH-D004 - Structural atmosphere representation

The retained `am12_fixed_djf25_piecewise_linear_los_tau_v1` authority is
represented as a per-array, content-bound piecewise-linear function of
`ell = tau225 * X` at `X_ref = 0`. The authoritative node ordinate must declare
whether it is transmission or correction; interpolation occurs before taking
any reciprocal. The analytic zero anchor is unity, exact nodes are recovered,
seams are continuous, positive-opacity transmission/correction are positive
and strictly directed, and no extrapolation or unity plateau is admitted.

This is a structural adoption only. It does not claim atmosphere truth,
representation accuracy, repeatability, absolute calibration performance, or
production readiness.

### SCI-CAL-AUTH-D005 - Support is an intersection, not a guessed elevation limit

Science-policy support is the intersection of `0 <= tau225 <= 0.15`, the
upstream airmass model's declared domain, the retained operator's numeric
line-of-sight node support, and the opacity source's time validity. CAL does not
invent a fixed minimum elevation or maximum airmass. A time-resolved value must
be exact or covered by a source-authorized interpolation between valid
bracketing states; endpoint extrapolation is not admitted.

### SCI-CAL-AUTH-D006 - Coherent quality and truthful no-output states

One predeclared observation or segment receives one opacity quality class. A
wholly supported low-opacity segment that passes every structural admission
condition may receive only `science-qualification-eligible`; that state is not
an achieved `science-qualified` or `calibrated-science` claim. A segment
entering `0.15 < tau225 <= 0.25` is `engineering-only unavailable` and has no
calibrated v0.1 output. Larger, absent, malformed, or out-of-domain states are
outside supported calibration. Intentional disablement, invalid input,
unavailable uncertainty, engineering-only state, eligibility, and achieved
qualification remain distinct.

### SCI-CAL-AUTH-D007 - Occurrence-scoped admission predicate

Calibration requires a unique acquisition occurrence, a keyed or proven
ordered binding, one immutable selected APT row occurrence, and one admitted
target-to-source association edge naming both endpoints and matcher evidence.
Design identity is independent and is not required for ordinary measured
Beammap quantities. Row, UID, path, semantic-content, and byte identities are
not conflated.

### SCI-CAL-AUTH-D008 - Conditional and nuisance uncertainty are separate

At fixed calibration state, full covariance transforms as `A C_x A^T`; scalar
variance scales by `M^2` and inverse variance by `M^-2`. A required nuisance
ledger covers detector `flxscale`, common calibrator scale, any pointing
correction, WVR/atmosphere model, `sens` when used for approximate weights, and
beam/template response with correlation scope. Total uncertainty is
unavailable unless all required terms and material cross-covariances are
quantified or justified not applicable. Common terms do not average down with
sample count.

### SCI-CAL-AUTH-D009 - Response condition for the unit label

The originating per-detector Beammap beam/template and realized downstream
map/kernel/filter response remain separate. An unresolved source of flux `S`
mJy has ideal peak `S mJy/beam` only when the realized response has unit peak
for the declared template or an explicit renormalization establishes it.
Available elliptical parameters are retained; circularization is labeled.

### SCI-CAL-AUTH-D010 - Claim-layer separation

Structural calibration correctness, atmosphere representation fidelity,
relative repeatability, and absolute flux performance have independent evidence
and status. A structural pass is at most
`science-qualification-eligible`. Unless the owner later defines a different
explicit threshold, an achieved `science-qualified` or `calibrated-science`
claim requires separately accepted atmosphere-representation-fidelity and
observational-performance evidence over the same declared identity and
support. The prior approximately one-percent, five-percent, and
five-to-ten-percent goals are falsification targets, not guarantees. No
scientific validation is executed as part of authorship.

## One unresolved owner decision

### SCI-CAL-OWNER-Q001 - Bind the retained numeric atmosphere operator

Will the owner approve one immutable numeric authority record for
`am12_fixed_djf25_piecewise_linear_los_tau_v1` that supplies, for every
supported TolTEC array:

1. the record's stable identity and content digest;
2. the exact ordered line-of-sight optical-depth nodes and dimensionless
   ordinates, including units and array assignment;
3. whether the ordinate that is piecewise-linearly interpolated is
   transmission or correction;
4. the exact closed numeric support and the rule at every node/seam; and
5. the generating atmosphere-model provenance and exact passband/response
   weighting convention, binding the approved TolTECA v1 passband-set identity
   where applicable?

Minimum sufficient answer: approve and provide one content-bound record with
all five fields. A name, legacy selector, plot, approximate range, or node list
without orientation and provenance is not sufficient.

Consequence while unanswered: the structural operator equations and rejection
rules are authoritative draft content, but numeric atmosphere evaluation,
calibrated numeric output, numeric `science-qualification-eligible`
disposition, atmosphere-representation fidelity, and any total uncertainty
depending on the operator remain unavailable. Resolving this question does not
by itself establish an achieved `science-qualified` or `calibrated-science`
claim; the separate evidence threshold in SCI-CAL-AUTH-D010 still applies. No
nodes, orientation, support, passband weighting, or fallback may be inferred.

## Known limitation that is not silently converted into a decision

The selected TolTECA v1 passband set is a content-bound modeled array reference.
Its detector/network aggregation, telescope-measured uncertainty/covariance,
normalization semantics, generator recipe, and photon-versus-energy physical
convention remain unestablished. Even after SCI-CAL-OWNER-Q001 is answered,
these unknowns must remain explicit limitations or nuisance-unavailable states
until separately authorized evidence resolves them.
