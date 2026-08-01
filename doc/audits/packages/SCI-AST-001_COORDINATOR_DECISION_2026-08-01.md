# SCI-AST-001 coordinator and scientific-owner decision — 2026-08-01

Status: approved; `SCI-AST-001-D001`--`D008` resolved for contract design

Package: `SCI-AST-001`

Governing audit report:
`SCI-AST-001_SCIENTIFIC_CONTRACT_AUDIT.tex`, SHA-256
`0be6771bbe5653bd42e90bc9a8cec1cd69ad84af971e6e7bca3d2fc21ed4bd98`

## SCI-AST-001-D001 — Sign, basis, detector association, and composition

Decision: approved with end-to-end astrometric self-consistency and layered
detector-association compatibility guards.

### Owner authority and evidence classification

The project owner states that the existing end-to-end astrometric convention
is known to be correct: large maps of approximately one square degree place
point sources at the correct sky locations, including at map corners. The
scientific requirement is therefore not to impose a newly named sign
convention, but to preserve and make explicit the self-consistent convention
already realized across pointing corrections, focal-plane offsets, projection,
WCS, and products.

This is a project-owner scientific and operational evidence statement. It is
authoritative for the compatibility policy selected here, but no exact map
artifact or digest was supplied with this decision. The later repair and
re-audit must still bind the claim to exact fixtures and successor-SHA results.

The owner further states that the existing row association is believed to be
correct, although with less evidence, and that no tone-frequency-to-design-ID
mapping is 100 percent reliable. The dated
[`CAL-D002` identity amendment](SCI-CAL-001_APT_IDENTITY_DECISION_AMENDMENT_2026-08-01.md)
records the supporting read-only TolProj and tone-match evidence and governs
the identity portion of this decision. It supersedes the earlier coordinator
recommendation that a universal stable detector UID is mandatory and that all
row-position binding is prohibited.

### Approved contract

- Preserve the established end-to-end astrometric transform, including its
  effective signs, tangent basis, focal-plane rotation, handedness, and
  composition order, for ordinary supported paths that reproduce the known
  correct source locations.
- Do not introduce a global sign flip, axis swap, handedness change, rotation
  change, or composition-order change merely to force internal names to match
  an abstract east/north convention.
- Document each stage's realized input and output basis and sign. Public labels
  and metadata must describe the realized transform; they must not redefine it.
- Where an upstream residual or coordinate increment uses a different sign or
  basis, use one explicit, tested boundary adapter that preserves the approved
  end-to-end result. Do not distribute compensating sign changes across stages.
- Attach focal-plane offsets through an admitted one-to-one detector binding.
  A proven observation-local or target-row-order contract is an allowed
  compatibility mode; an explicit keyed mapping is also allowed. Bare,
  unvalidated table coincidence is not identity.
- Keep observation-local acquisition identity, the selected measured Beammap
  row and its cross-observation matcher edge, and optional design identity as
  distinct, provenance-bearing objects. A matched APT can have the correct
  target row order while carrying an imperfect source-row or design match.
- Do not require a design ID for AST use of admitted measured Beammap geometry.
  If design-derived geometry is selected, its match validity and admission
  rule become part of the coordinate contract.
- Treat current numerical behavior as the compatibility reference only over
  its demonstrated ordinary domain. This decision does not retain the audited
  TAN invalid-domain alias, asymmetric wrap handling, false metadata, or
  unproved detector-row admission.

### Mandatory compatibility and falsification gates

- Trace and test the sign, basis, rotation, handedness, and composition at
  every stage rather than testing only the final map center.
- Preserve point-source sky locations in representative approximately
  one-square-degree maps at the center, edges, and corners, with no mirror,
  axis-swap, rotation, or handedness regression.
- Preserve established Point and Beammap source-crossing directions,
  centroids, and detector-coordinate behavior across the supported arrays.
- Prove the selected detector-binding mode at admission. In verified-row mode,
  exact observation/artifact provenance, network order, per-network count and
  tone order, and unique acquisition keys are mandatory; an unkeyed reorder
  fails closed. Explicit-key mode must be permutation invariant. Missing,
  duplicate, conflicting, or non-finite records fail closed.
- Changing or omitting a design ID must not change coordinates when only
  admitted measured Beammap geometry is used. Forced unmatched or ambiguous
  source/design associations must retain that state rather than being silently
  coerced to an exact identity.
- Any changed ordinary-source location must be attributed to a separately
  approved defect repair and remain within preregistered astrometric
  compatibility tolerances. The candidate result may not define its own gate.

### Effect

`SCI-AST-001-D001` is resolved for contract design. It preserves demonstrated
astrometric behavior while requiring explicit stage semantics and a validated,
layered detector binding without claiming perfect design identity. It changes
the closure contract for `SCI-AST-001-F013` but does not close that finding or
`F006`, approve the governing implementation, authorize repair, supply exact
validation evidence, or decide any other AST owner question.

## SCI-AST-001-D002 — TAN operational domain and invalidity

Decision: approved with an open-forward-hemisphere domain, explicit validity,
and fail-closed consumer policy.

### Issue

The assessed forward TAN implementation maps `abs(D) < epsilon` to `(0,0)` and
accepts `D < 0`. A singular or back-hemisphere direction can therefore become
indistinguishable from a valid source at map center. Finite but enormous
coordinates can also reach geometry or integer-pixel consumers without one
authoritative coordinate-validity operator.

### Approved contract

- Require finite projection center and direction inputs. Compute the declared
  TAN denominator `D` without replacing it by a compatibility sentinel.
- The mathematical forward-projection domain requires finite `D > 0`.
  Non-finite `D` or `D <= 0` is out of domain. Do not use a near-zero epsilon
  branch to map either side of the singularity to center, and do not clamp an
  invalid direction to a map edge.
- Do not introduce an additional fixed minimum denominator or maximum angular
  radius in this decision. A tighter operational radius may be added only when
  derived from a supported map footprint, preregistered independently of the
  candidate, and recorded in requested/effective/resolved/realized state. It
  must not redefine the ordinary demonstrated astrometry.
- Require finite continuous tangent coordinates. Non-finite input,
  out-of-domain `D`, or non-finite projection output produces an explicit
  coordinate-invalid state with a reason; `(0,0)`, NaN alone, an edge pixel,
  or an inherited signal flag is not the validity state.
- Keep projection validity distinct from ALIGN eligibility, signal flags, and
  product support. An already-ineligible input remains excluded. A valid TAN
  coordinate outside a declared map/WCS footprint is explicitly outside that
  product's support and is excluded before integer conversion; it is not
  relabeled as a projection failure or clamped into the map.
- If an otherwise eligible sample lacks a valid required TAN coordinate, the
  required coordinate-dependent product and reduction fail rather than
  silently losing the sample. A globally invalid projection center or
  singular/inconsistent required WCS fails setup.
- Every coordinate consumer must admit the explicit validity and continuous
  support state before geometry, rounding, mapmaking, fitting, feedback, or
  persistence.

### Mandatory compatibility and falsification gates

- The projection center maps exactly to `(0,0)` and existing representative
  approximately one-square-degree center, edge, and corner source locations
  remain unchanged within preregistered compatibility tolerances.
- Test finite inputs on both sides of `D = 0`, positive and negative zero, the
  exact boundary, adjacent representable values, quarter-turn and antipodal
  cases, and every non-finite input or output. No invalid case may become
  center, reuse a prior coordinate, or become a finite valid-looking pixel.
- As `D` approaches zero from the positive side, verify the expected response
  growth until either finite projection or declared product support ends;
  never define correctness by the candidate result.
- Pass forward/inverse TAN round trips over the admitted domain, including
  near-boundary and longitude-wrap fixtures, to preregistered angular and WCS
  tolerances.
- Test the separation among projection-invalid, valid-but-outside-product,
  and preexisting-ineligible samples through every named consumer. No
  non-finite or unsupported coordinate reaches integer conversion.
- Sequential and supported parallel execution must produce identical
  coordinates, validity reasons, exclusion counts, and required-failure state.

### Effect

`SCI-AST-001-D002` is resolved for contract design. It supplies the domain,
boundary, invalid representation, and consumer failure policy needed by
`SCI-AST-001-F001` and `F012`. Both findings remain open until implementation,
exact repair-SHA local and operational evidence, and fresh re-audit succeed.
This decision does not select a repair base or authorize repair, Unity work,
application integration, production expansion, or a tighter approximation
radius.

## SCI-AST-001-D003 — Frames, epochs, transforms, and longitude topology

Decision: approved with the established product-family frame split, explicit
frame/epoch authority, canonical circular longitude topology, and no implicit
new transformation.

### Issue

Point, OOF, Beammap, and Science do not use the same coordinate role. The
existing numerical paths can be operationally self-consistent while persisted
metadata is incomplete or nonstandard, a missing epoch silently defaults to
2000 without a transform, and AltAz corrects only one wrap direction. This can
mislabel otherwise correct coordinates or turn a boundary crossing into a
multi-radian displacement.

### Approved contract

- Preserve native AltAz tangent coordinates for Point, OOF, and Beammap.
  Citlali consumes the admitted telescope azimuth/elevation coordinates as
  supplied and does not independently reapply refraction, EOP, precession, or
  another sky-frame transformation on these ordinary paths.
- Preserve equatorial J2000 TAN for Science. Where admitted headers establish
  FK5 J2000, publish standard `RADESYS=FK5` and `EQUINOX=2000.0`. Where inputs
  explicitly establish ICRS, preserve ICRS or apply one named, versioned
  transformation; never relabel ICRS as FK5 or vice versa.
- Any other requested conversion, including apparent/of-date or AltAz to
  equatorial, must name its source and target frames, epoch/time scale, site,
  transformation implementation/version, and required EOP/refraction inputs.
  Missing required authority or inputs fails before numerical application.
- Do not silently default a missing or invalid epoch to 2000. Existing
  compatibility products without sufficient frame/epoch authority may be
  retained only as explicitly `legacy_unverified` and may not support a new
  precision, frame, or transformation claim. New precise coordinate products
  fail admission when frame or epoch identity is ambiguous.
- Normalize persisted RA and azimuth longitudes to `[0, 2*pi)` internally and
  `[0, 360)` where written in degrees. Compute longitude differences through
  one canonical shortest-signed operator in `[-pi, pi)`, in both wrap
  directions. At the exact antipodal tie the `-pi` convention is deterministic
  and the forward TAN remains invalid under `SCI-AST-001-D002`.
- Apply the same circular topology after inverse TAN and at every coordinate
  adapter. Independent preprocessing of two longitude series is not a
  substitute for taking their shortest signed difference.
- Record the admitted source frame, epoch, native-coordinate source,
  transformation or explicit no-transform policy, EOP/refraction authority or
  non-applicability, and realized output frame through the four-stage state.
- Preserve the demonstrated end-to-end source locations and handedness from
  `SCI-AST-001-D001`; metadata must describe the realized operator rather than
  change it.

### Mandatory compatibility and falsification gates

- Preserve representative Point, OOF, and Beammap source crossings in native
  AltAz and Science source locations at the center, edges, and corners of
  approximately one-square-degree J2000 maps.
- Test constant and varying longitude series across `0/2*pi` in both
  directions, adjacent boundary values, the exact `pi` tie, horizon and polar
  cases, inverse-TAN normalization, and sequential/parallel equivalence.
- Prove that explicit FK5/J2000 and ICRS fixtures retain their identities and
  round-trip through standard FITS metadata without relabeling. A named
  transform must pass an independent reference fixture and record its inputs
  and version.
- Exercise missing, invalid, contradictory, and legacy frame/epoch headers.
  New precision paths must fail; retained compatibility paths must be visibly
  `legacy_unverified`, never silently defaulted.
- Verify that ordinary native-coordinate paths request no unavailable EOP or
  refraction input and apply no extra transform. Any enabled conversion path
  must fail when a required authority or input is absent.

### Effect

`SCI-AST-001-D003` is resolved for contract design. It supplies the longitude
topology and frame/epoch policy needed by `SCI-AST-001-F002`, and the relevant
parts of `F006` and `F010`. Those findings remain open pending implementation,
full-precision product decisions, exact repair-SHA validation, and fresh
re-audit. This decision does not settle nondefault WCS controls, persisted
numeric precision, simulation parity, or uncertainty scope, and it does not
authorize repair, Unity work, application integration, or production
expansion.

## SCI-AST-001-D004 — Pointing support modes and time adequacy

Decision: approved with explicit support modes, bracket-only interpolation,
and a scientifically proportionate time-precision gate.

### Owner authority and issue

The pointing-offset interpolation corrects slow drift in measured pointing
offsets. The project owner judges that a subsecond timing error will not
produce a pointing error that is a meaningful fraction of an arcsecond. The
contract must therefore remove ambiguous mode selection, stale/clamped
support, and untraceable interpolation without requiring a timing refactor for
precision that has no material astrometric consequence.

The assessed implementation silently selects observation-span interpolation
when either of two MJD values is nonpositive, accepts mixed sentinel states,
and converts MJD to integer Unix seconds. The audit probe demonstrated a
0.666569-second loss for one value, but did not demonstrate a meaningful
pointing displacement from that loss.

### Approved contract

- One pointing-correction pair means one constant correction over its admitted
  observation support. It is not represented as two synthetic endpoints.
- Two explicitly present, finite MJD supports select time interpolation. The
  supports must be strictly increasing, use a declared scale compatible with
  ALIGN, and bracket every otherwise eligible sample. Do not extrapolate,
  clamp, select the nearest endpoint, or reuse a prior observation's support.
- Legacy observation-span interpolation is a separate explicit mode and is
  selected only when both support times are deliberately absent. Mixed
  present/absent, equal, reversed, non-finite, or otherwise ambiguous support
  fails before applying a correction.
- The legacy span uses the exact first and last admitted aligned sample
  identities and times, requires a finite positive span, and interpolates only
  within that span. It never masquerades as MJD-supported interpolation.
- Time representation is governed by astrometric adequacy, not an arbitrary
  one-microsecond target or a requirement to preserve unused digits. The
  existing integer-second representation may remain if a preregistered bound
  using the actual time-quantization error and pointing-correction drift rate
  shows the resulting pointing error is negligible relative to established
  Point/Beammap centroid, repeatability, and PSF-width tolerances.
- If that bound fails, improve time precision only enough to meet the
  preregistered astrometric tolerance and remain compatible with ALIGN's
  admitted clock/cadence model. Do not wholesale retime ordinary data.
- Record the support mode, exact source support records, admitted sample span,
  time representation and quantization bound, interpolation weights, exclusion
  or failure counts, and realized correction identity through the four-stage
  state.

### Mandatory compatibility and falsification gates

- Test constant support; explicit-MJD endpoints and midpoint; samples just
  inside and outside both endpoints; both-absent legacy span; and mixed,
  equal, reversed, non-finite, zero-span, and unbracketed cases. No failed case
  may equal a clamped endpoint or reused prior correction.
- For representative pointing solutions, calculate the correction drift rate,
  actual time-quantization bound, and corresponding maximum angular error.
  Compare that bound to preregistered existing centroid/repeatability and
  PSF-width tolerances rather than to an arbitrary clock precision.
- If integer-second time is retained, prove support order, bracketing, and
  interpolation behavior remain valid after its declared conversion. If not,
  the minimum adequate higher-precision representation is required.
- Preserve ordinary Point and Beammap source-crossing times, centroids, and
  recovered PSF widths. Stop for owner review if ordinary valid samples move
  materially or established astrometric performance degrades.
- Sequential and supported parallel execution must produce identical support
  selection, interpolation weights, correction values, and failure state.

### Effect

`SCI-AST-001-D004` is resolved for contract design. It supplies the support
mode, sentinel, bracketing, extrapolation, span, and time-adequacy policy for
`SCI-AST-001-F005`. The ambiguous-mode and support-validation defects remain
open. Integer-second conversion is not by itself a required repair: it becomes
accepted compatibility behavior if the preregistered angular-error bound
passes, and otherwise must be improved only to the demonstrated need. `F005`
remains open pending that evidence, implementation of the remaining contract,
exact repair-SHA validation, and fresh re-audit. This decision does not
authorize repair, Unity work, application integration, production expansion,
or a broader ALIGN timing change.

## SCI-AST-001-D005 — Accepted nondefault WCS controls

Decision: approved as a narrow truthful-boundary contract: retain the legacy
automatic-zero path and reject unsupported explicit values.

### Issue

The configuration accepts `crpix1`, `crpix2`, `crval1_J2000`,
`crval2_J2000`, `tan_ra`, and `tan_dec`, but the assessed application does not
reliably realize nondefault requests. Geometry replaces CRPIX with the centered
map value, observation setup obtains CRVAL from the telescope source header,
and the configured TAN fields have no complete realization path. Silent
success therefore permits requested and realized WCS identities to disagree.

### Approved contract

- Preserve exact numeric zero in each of the six legacy scalar fields as the
  current `automatic` compatibility sentinel. In this bounded successor,
  legacy zero is not interpreted as an explicit pixel or sky coordinate.
- Under `automatic`, preserve the established centered-map reference pixel,
  telescope/source-derived sky reference, product-family frame split, and FITS
  zero-based-to-one-based CRPIX conversion. This decision does not change
  ordinary generated WCS geometry.
- Reject every nonzero value for any of the six controls during configuration
  admission, before observation setup or numerical processing. The error must
  name the field and state that explicit WCS control is unsupported; warning,
  silent overwrite, serialization-only acceptance, or later fallback is not
  allowed.
- Reject non-finite values independently. Requested state retains the supplied
  bytes/value and rejection reason; an admitted automatic request resolves to
  the actual reference pixel, sky center, frame, units, and source authority
  in effective/observation-resolved/realized provenance.
- If explicit control is implemented later, introduce a typed `automatic`
  versus `explicit` representation with versioned field semantics, units,
  frame, index base, interaction rules, and round-trip tests. Do not continue
  overloading zero, because zero degrees and an internal zero pixel can be
  legitimate explicit values.
- No accepted field may be ignored. The configuration surface and generated
  templates must describe only the admitted automatic behavior until a
  separately approved explicit implementation exists.

### Mandatory compatibility and falsification gates

- All-zero legacy configurations must reproduce the existing centered-map and
  source-header WCS values, handedness, source locations, and FITS CRPIX
  indexing within preregistered compatibility tolerances.
- Exercise each of the six fields independently and in combinations with
  positive, negative, smallest representable nonzero, NaN, and infinity
  values. Every unsupported or non-finite request must fail at admission and
  produce no partial reduction or misleading realized WCS.
- Prove that automatic request, effective mode, resolved reference values, and
  realized FITS/product metadata round-trip without requested-to-realized
  backflow or loss of the derivation source.
- Config serialization, generated profiles, and the full config preflight must
  preserve legacy zero while identifying it as automatic rather than an
  explicit scientific coordinate.

### Effect

`SCI-AST-001-D005` is resolved for contract design and supplies the initial
closure policy for `SCI-AST-001-F003`. `F003` remains open until admission,
state/provenance, negative tests, exact repair-SHA validation, and fresh
re-audit pass. This decision deliberately avoids implementing unused WCS
flexibility and does not settle persisted numeric precision, authorize repair
or Unity work, expand production use, or approve a future explicit-control
contract.

## SCI-AST-001-D006 — Map-center response and uncertainty

Decision: approved only for a single map-center response calculation; broader
per-sample, per-detector, per-pixel, or dense covariance work is rejected as
unnecessary and wasteful.

### Owner authority and proportionality judgment

The project owner states that offset-pointing sources are observed at the same
or very similar declination as the science target. The projection response at
the realized map center is therefore an adequate operational representation
for the relevant pointing and source-position uncertainty. Calculating a
response across the map, timestream, detector set, or observation is not
approved.

This is a scientific and operational scope decision. It replaces the broader
coordinator recommendation for a reconstructible detailed AST response and
conditional composition of all available correction, APT, ALIGN, frame, and
inverse-projection covariance terms.

### Approved contract

- When a product reports positional uncertainty in equatorial sky
  coordinates, evaluate the local inverse-TAN 2x2 response once at the
  realized map/WCS tangent center and use it to transform the available
  map-center positional covariance, including its cross term.
- If the product reports only native tangent-plane offsets and uncertainties,
  preserve those values and units; do not perform an unnecessary sky-frame
  propagation merely to create an additional product.
- Treat the center response as a product-level quantity. It may be stored
  directly or reconstructed from the realized center, frame, projection, and
  convention. Do not calculate or store response matrices per sample, time,
  detector, source-map pixel, or science-map pixel.
- Do not construct a dense observation covariance, a response grid, or a
  composed covariance spanning pointing correction, ALIGN, APT/focal-plane,
  frame/model, interpolation, selection, and systematic terms. Such work is
  outside the approved scope.
- Preserve explicit availability for the map-center uncertainty and its input
  terms. A missing or unmodeled term is `unavailable`, not zero. Do not publish
  a synthetic total or a new precision claim from incomplete inputs.
- Preserve the existing mean-coordinate and mapmaking calculations. This
  decision adds no sample eligibility, weighting, or mapmaking operation and
  must not move a source or change an ordinary science pixel value.
- A future request for spatially varying, per-sample, per-detector, or composed
  covariance requires a new owner decision supported by a demonstrated
  scientific need and performance budget; it is not latent repair scope.

### Mandatory compatibility and falsification gates

- Verify the center response against an independent analytic or finite-
  difference inverse-TAN reference at the equator and representative target
  declinations, including the audited high-declination cases and longitude
  wrap. This is a small fixture suite, not an observation-wide calculation.
- Demonstrate that the transformed center covariance has the expected units,
  symmetry, finite values, and cross term when the required input covariance
  is available. Missing input must produce explicit unavailability rather
  than a zero matrix or fabricated total.
- Confirm that an offset-pointing source at the map center retains its
  established fitted position and tangent-plane errors, and that any reported
  equatorial uncertainty uses the center response rather than scalar unit
  conversion alone.
- Verify that the implementation creates no per-sample, per-detector,
  per-pixel, response-grid, or dense-covariance allocation or hot-path work.
  Ordinary runtime and numerical map products must remain unchanged within
  preregistered compatibility tolerances.

### Effect

`SCI-AST-001-D006` is resolved for contract design with a deliberately narrow
map-center uncertainty product. It revises the closure scope for
`SCI-AST-001-F007` and `F008`: the scalar inverse-TAN error conversion must be
corrected at map center, and unavailable terms must be represented truthfully,
but no broader covariance propagation or response materialization is
required. Both findings remain open until the bounded implementation,
fixtures, performance confirmation, exact repair-SHA evidence, and fresh
re-audit succeed. This decision does not authorize repair, Unity work,
application integration, production expansion, or any off-center response or
covariance calculation.

## SCI-AST-001-D007 — Bounded approximation and simulation parity

Decision: approved with preservation of the established small-angle production
path, an offline astrometric-adequacy bound, and fail-closed simulation scope.

### Issue and owner proportionality judgment

Citlali uses a fast small-angle approximation when combining pointing
corrections and detector focal-plane offsets. The approximation is
operationally useful, and the project owner's large-map evidence establishes
that ordinary approximately one-square-degree products place sources correctly
at their centers, edges, and corners. The contract nevertheless does not state
the supported geometry envelope or bound its difference from exact spherical
composition.

Separately, the assessed simulation path bypasses parts of real-data time,
longitude-wrap, and frame preparation, forces epoch 2000, and exposes a
Galactic mode without constructing Galactic tangent coordinates. It therefore
cannot presently establish parity with the production coordinate operator.

The owner approves bounding the current approximation without replacing it or
adding exact spherical work to the production hot path. Simulation work is
limited to truthful parity for already supported AltAz and RA/Dec modes; a new
Galactic implementation is not approved.

### Approved contract

- Preserve the current small-angle calculation and its established signs,
  basis, rotation, composition order, and runtime behavior over the validated
  operational envelope. Do not replace it with per-sample exact spherical
  composition.
- Define the envelope from preregistered supported map footprints,
  focal-plane/detector offsets, and pointing-correction magnitudes. Compare the
  approximation with an independent exact spherical reference offline across
  that envelope.
- Judge adequacy by the maximum angular discrepancy relative to established
  Point/Beammap centroid, repeatability, and PSF-width tolerances. Do not impose
  an arbitrary numerical precision threshold or let the candidate result set
  its own gate.
- Admit a requested/resolved geometry only within the validated envelope.
  Configuration or setup outside it fails closed. An unexpected runtime input
  outside the admitted envelope also fails rather than silently extrapolating.
- No automatic exact-spherical fallback is authorized. Adding one later
  requires a separate owner decision with demonstrated scientific need and a
  performance budget.
- For simulated AltAz and RA/Dec data, reuse the same applicable coordinate
  preparation, topology, frame identity, support state, and AST operator as
  the real-data path. Simulation-specific data generation may remain separate,
  but it may not bypass the admitted coordinate contract.
- Reject Galactic simulation during configuration admission until its source
  conversion, tangent coordinates, frame metadata, and parity are explicitly
  implemented and approved. Do not advertise or partially execute that mode.
- Simulation parity is validation behavior. It adds no calculation or
  allocation to an ordinary real-data production reduction.

### Mandatory compatibility and falsification gates

- Use a compact offline grid covering zero, ordinary, and envelope-boundary
  combinations of supported footprint, focal-plane offset, and pointing
  correction. Compare the existing approximation against an independent exact
  spherical oracle and record the maximum angular discrepancy and its gate.
- Preserve representative Point, OOF, Beammap, and approximately one-square-
  degree Science source positions, source-crossing directions, centroids, and
  PSF widths within preregistered compatibility tolerances.
- Exercise inputs just inside, at, and outside the approved envelope. Outside
  requests must fail without partial products; the production hot path must
  contain no newly introduced exact spherical calculation.
- For each supported simulated AltAz and RA/Dec fixture, compare real-path and
  simulation-path prepared coordinates from the same admitted inputs,
  including zero correction, wrap, and representative support cases.
- Verify that Galactic simulation and every other unimplemented frame fail at
  configuration admission with a specific unsupported-mode error.
- Measure or inspect the changed execution path sufficiently to confirm no
  material regression in ordinary coordinate-processing timing or allocation.

### Effect

`SCI-AST-001-D007` is resolved for contract design. It completes the
small-angle scope of `SCI-AST-001-F006` and supplies the parity/fail-closed
policy for `F014`. Both findings remain open pending the bounded implementation,
offline reference evidence, compatibility and performance gates, exact
repair-SHA validation, and fresh re-audit. This decision does not authorize
repair, Unity work, an exact-spherical production path, Galactic simulation,
application integration, or production expansion.

## SCI-AST-001-D008 — Persisted precision, metadata, validity, and products

Decision: approved with high precision for compact coordinate authorities,
measured adequacy for large timestream arrays, factorized validity, truthful
full/mini availability, and atomic four-stage provenance.

### Issue and owner proportionality judgment

The assessed products do not share one coherent coordinate contract. Key WCS
values are held as `float`; frame and epoch metadata can be incomplete; the
generic TOD writer labels time, counters, and state fields as radians; explicit
coordinate validity is absent; mini TOD omits detector coordinates without a
complete availability declaration; and AST provenance records modes and
counts rather than its resolved and realized scientific identity.

These are principally precision-at-authority, schema, validity, and
reproducibility defects. They do not justify doubling every large coordinate
timestream, adding detector-by-sample identity arrays, expanding mini TOD, or
changing map calculations.

### Approved contract

- Store and persist product-level WCS authorities and fitted/catalog sky
  coordinates in double precision. This includes continuous reference values,
  pixel scale and orientation/handedness terms needed for an exact WCS
  round-trip.
- Do not automatically convert full detector-coordinate timestream arrays to
  double precision. Retain their existing representation if an offline
  quantization bound shows negligible angular effect relative to established
  Point/Beammap centroid, repeatability, and PSF-width tolerances. If that gate
  fails, stop for owner review before changing the large-product type.
- Apply the approved frame and epoch policy from `SCI-AST-001-D003`, including
  truthful standard FITS metadata. Do not default, relabel, or transform an
  ambiguous coordinate identity.
- Give each persisted telescope and coordinate field its declared units,
  topology, frame where applicable, validity role, and indexing semantics from
  the admitted ALIGN/AST registry. Remove the generic all-radians attribute.
- Represent coordinate validity with a dedicated compact AST status, distinct
  from signal flags and coordinate values. Factor it whenever the admitted
  operator permits: one packed status on the aligned telescope-sample grid,
  one detector-admission state per detector, and product-level exclusion and
  failure counts.
- Do not create a routine detector-by-sample identity array or validity matrix.
  If an enabled path produces nonfactorable detector-by-sample coordinate
  invalidity, fail that coordinate product and return for a separately scoped
  owner decision rather than silently losing state or creating a large product.
- Full TOD retains its coordinate arrays together with the compact state needed
  to interpret their validity and scientific identity. Mini TOD does not gain
  coordinate arrays; it declares those fields unavailable and retains only the
  compact product identity, availability, and summary counts appropriate to
  its role.
- Reuse the admitted ALIGN grid identity and AST detector binding. Do not
  duplicate per-sample IDs where the ordered grid identity reconstructs the
  association.
- Persist requested, effective, observation-resolved, and realized AST state
  atomically and one-way. Include support mode and sources, frame/WCS identity,
  approximation envelope, detector binding, validity/uncertainty availability,
  product links, counts, algorithm/contract versions, and artifact digests.
- Preserve existing map values, source positions, ordinary data cardinality,
  and runtime behavior. Metadata and validity state may describe a result more
  truthfully but must not alter the coordinate operator or science weighting.

### Mandatory compatibility and falsification gates

- Round-trip double-precision WCS and catalog coordinates through the actual
  FITS/table writers and readers, including CRPIX index conversion, scale sign,
  handedness, frame, epoch, longitude wrap, and representative center/corner
  positions.
- Measure the current full-TOD coordinate quantization error offline over the
  supported domain and compare it with preregistered existing astrometric
  performance. A passing representation remains unchanged; a failing one
  triggers owner review and is not silently widened.
- Validate the field registry and generated attributes for time, coordinate,
  pointing-offset, counter, state, and other representative telescope fields.
  No field may inherit radians merely from the generic writer.
- Exercise factorized sample validity, detector admission, projection-invalid,
  valid-but-outside-product, and preexisting-ineligible states through all
  consumers and the full-TOD round-trip. No coordinate value or signal flag may
  substitute for AST validity.
- Verify that mini TOD contains no new coordinate or validity arrays and
  explicitly reports their unavailability. Full and mini readers must
  distinguish omitted, unavailable, and present-valid fields.
- Verify exact ordered-grid and detector-binding reconstruction without a
  repeated per-sample identity array. Mismatch, reorder, duplicate, and stale
  identity fail closed.
- Test atomic provenance success and injected write failure. The four stages,
  product links, cardinalities, versions, and digests must serialize exactly,
  with no realized-to-requested backflow.
- Confirm no material increase in ordinary mini-TOD size or production
  coordinate-processing runtime, and no change in ordinary map pixels,
  centroids, source locations, or PSF widths beyond preregistered compatibility
  tolerances.

### Effect

`SCI-AST-001-D008` is resolved for contract design. It supplies the bounded
closure policy for `SCI-AST-001-F009`, the remaining product-precision and
metadata portion of `F010`, the per-variable schema for `F011`, and the
persisted/factorized validity portion of `F012`. Those findings remain open
pending implementation, adequacy and schema evidence, exact repair-SHA local
and operational validation, and fresh re-audit. The full set D001--D008 is now
owner-approved, but this decision does not itself authorize repair, select a
repair base, request Unity evidence, launch re-audit, integrate application
code, or expand production use.

## Decision completion

All `SCI-AST-001` scientific-owner decisions are resolved for contract design.
The findings and governing implementation remain open. No repair, Unity
request, re-audit, application integration, production expansion, or
composition-framework action is authorized by this decision record.
