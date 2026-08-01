# SCI-AST-001 coordinator and scientific-owner decision — 2026-08-01

Status: in progress; `SCI-AST-001-D001` and `D002` approved; `D003`--`D008`
pending

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

## Pending decisions

- `SCI-AST-001-D003`: frames, epoch, transforms, and longitude topology.
- `SCI-AST-001-D004`: support modes and time precision.
- `SCI-AST-001-D005`: accepted nondefault WCS controls.
- `SCI-AST-001-D006`: response, covariance, and unavailable semantics.
- `SCI-AST-001-D007`: approximation bounds and simulation parity.
- `SCI-AST-001-D008`: persisted precision, metadata, validity, and products.

No repair, Unity request, re-audit, application integration, production
expansion, or composition-framework action is authorized by this partial
decision record.
