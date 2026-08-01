# SCI-AST-001 Coordinator Review And Owner Decision Brief

Date: 2026-08-01
Package: `SCI-AST-001`
Coordinator disposition: audit integrated; owner decisions required
Repair authorization: none

The project owner explicitly authorized this ordinary audit/handoff and
phase-zero return integration on 2026-08-01. That authorization does not
approve either held composition-framework decision, a framework amendment, or
the closure pilot.

## Reviewed identity

- Governing application SHA: `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Audit branch: `codex/audit-sci-ast-001`.
- Independent-core commit: `17d683ada3856ecb5f0a5c42eed744cb219a3586`.
- Independent-core SHA-256: `ed1fe3bf68ed53974b8c910bd3824717eb0cf5ff11d0b27c0fdf27aa6e606276`.
- Report-bearing audit commit: `429e1b5361683ba15c8d897ba22bdc4c4d03bf91`.
- Final identity-binding package commit: `e3553bc0fcaa158ed4d986f59e9f25e5e2eeac7a`.
- Final report SHA-256: `0be6771bbe5653bd42e90bc9a8cec1cd69ad84af971e6e7bca3d2fc21ed4bd98`.
- Machine-readable ledger proposal SHA-256: `a9b31ad958f8a91f2725fd2be351fd19e74319f87e1b643cf46e839606fb7ca7`.
- Local-evidence SHA-256: `e391757e63477fd568a415e8c6b9c4bca6dc98603cd83088c350cbc03a51f986`.

The worktree was clean at the final package commit. The audit changed only
audit packages, evidence, proposals, and an unsupplied Unity request. It did
not modify application code, tests, build files, or production configuration.

## Canonical result

The audit verdict is `amend`. The contract is `proposed`, the governing
implementation is `nonconformant`, validation is `in_progress`, production
remains `existing_use_only`, and a fresh exact-successor-SHA re-audit is
required.

The five P0 findings are:

1. invalid or back-hemisphere TAN directions can alias to a finite map center;
2. AltAz uses a one-sided azimuth wrap correction;
3. accepted nondefault WCS controls can be ignored or overwritten;
4. ALIGN does not yet supply AST's required typed interface; and
5. APT rows are used without proving an exact one-to-one raw-detector join.

Ten P1 findings cover support-time precision, sign/basis/frame authority,
inverse-TAN covariance, uncertainty availability, provenance, WCS precision,
false generic TOD units, coordinate-validity propagation, simulation parity,
and missing exact-SHA numerical/operational evidence.

`SCI-AST-001-F013` is linked to the already-open CAL detector-identity defect
`SCI-CAL-001-F004` and owner decision `CAL-D002`; no duplicate CAL finding is
created.

The governing-SHA Unity request is retained as an unsupplied audit artifact.
It must not be run or treated as repair evidence: known analytic P0 defects
already establish nonconformance. Its proposed one-microsecond timing
tolerance is not approved, and its bundle-generation procedure needs a
successor rewrite. A future request follows successful repair-local gates; its
complete human-run return is then evidence for the fresh re-audit.

## Coordinator normalizations

The immutable audit proposal remains unchanged. Canonical coordination state
distinguishes the report-bearing audit commit `429e1b536...` from the final
identity-binding package commit `e3553bc0...`.

Seven proposed handoffs are integrated as canonical recipient records. The
submission payload of `SCI-VAL-001-XAUD-005` is preserved, including its
incorrect `T01-T23` label; only canonical lifecycle and recipient-disposition
metadata change. Corrected successor `SCI-VAL-001-XAUD-006` records the actual
suite `A01-A23` without changing a scientific claim or requested action. This
preserves immutable submission fields and the registry supersession rule.

The incoming ALIGN handoff `SCI-AST-001-XAUD-001` is acknowledged and its
dependency, findings, interface-test requirement, and consumer restriction
are incorporated. It does not close the ALIGN dependency.

## Decisions for the project owner

### SCI-AST-001-D001 — correction sign, basis, and composition order

Issue: the current implementation lacks one authoritative statement of the
signs and basis for telescope pointing corrections and detector focal-plane
offsets, their rotation, their order of application, and the detector identity
used to join those values. An apparently small sign or ordering ambiguity can
move every detector in a common direction or mirror the focal plane.

Recommendation: define positive tangent longitude as east and positive tangent
latitude as north; apply the pointing correction before detector displacement;
join detector offsets by stable UID rather than row position; and require
explicit adapters for any upstream residual whose sign or coordinate-increment
semantics differ. Validate the convention against Point and Beammap source
crossings.

### SCI-AST-001-D002 — TAN domain and invalidity

Issue: a tangent projection is defined only on the forward hemisphere. The
current near-zero-denominator branch can turn a singular or back-hemisphere
direction into a plausible coordinate at map center.

Recommendation: require finite inputs and strictly positive TAN denominator
`D`; never map a small or negative `D` to center. Require finite continuous
coordinates and declared WCS support before integer conversion. Any tighter
operational radius should be derived from supported map footprints and
preregistered rather than guessed.

### SCI-AST-001-D003 — frames, epochs, and longitude topology

Issue: Point/OOF/Beammap and Science do not use the same sky-coordinate role,
while some current metadata defaults an epoch or uses incomplete frame labels.
Longitude differences also require circular, shortest-path handling.

Recommendation: retain AltAz tangent coordinates for Point, OOF, and Beammap;
use explicit J2000 equatorial TAN for Science. Use `FK5` with
`EQUINOX=2000` only where headers establish that identity; otherwise mark the
legacy coordinates restricted. Normalize longitude and use the shortest
wrapped difference. Missing frame or epoch authority fails any new precision
claim.

### SCI-AST-001-D004 — pointing support and time precision

Issue: the implementation mixes support sentinels, truncates MJD information,
and can leave bracketing or extrapolation behavior ambiguous. This seam must
compose with ALIGN's eventual clock and support contract.

Recommendation: one support pair means a constant correction. Two finite,
strictly increasing supports use lossless times on the same scale as ALIGN,
require bracketing, and never extrapolate. Only two deliberately absent
supports select an explicit legacy observation-span mode. Mixed, reversed, or
nonfinite supports fail closed. Do not adopt the audit request's one-
microsecond tolerance; derive precision from approved ALIGN cadence/jitter and
source-crossing evidence.

### SCI-AST-001-D005 — accepted WCS controls

Issue: current configuration accepts `CRPIX`, `CRVAL_J2000`, `tan_ra`, and
`tan_dec`, but nondefault values are not reliably realized. Silent acceptance
creates false requested-to-realized provenance.

Recommendation: retain exact zero as an explicit `automatic` compatibility
sentinel because generated configurations already use it. Reject every nonzero
value for those controls until its exact semantics are separately implemented
and tested. No accepted field may be silently ignored.

### SCI-AST-001-D006 — response and uncertainty

Issue: AST products do not expose enough response and uncertainty state to
propagate pointing, alignment, focal-plane, frame, and inverse-projection
effects, but routine dense observation covariance would be impractical.

Recommendation: publish a versioned or reconstructible compact AST Jacobian,
explicit term-availability states, and conditional 2x2 propagation of the
available correction, APT, ALIGN, and frame terms, including inverse-TAN
cross-covariance. Unknown terms are unavailable, never zero. Detailed mappings
may be compact or supplied `as_requested`; composition uses `J_AST J_ALIGN`.

### SCI-AST-001-D007 — approximation and simulation parity

Issue: the established small-angle path is operationally useful, but its
valid radius/error is not stated and simulation can bypass real-path setup or
advertise unsupported Galactic coordinates.

Recommendation: preserve the measured small-angle hot path inside an approved
radius/error bound, use exact spherical equations as the oracle, and either
fall back to the exact operator or fail closed outside the domain. Simulation
must use the same state and operator; unsupported Galactic simulation fails
closed.

### SCI-AST-001-D008 — persisted products and provenance

Issue: coordinate precision, frame metadata, per-field units, validity,
requested-to-realized provenance, and full-versus-mini TOD content are not one
coherent product contract.

Recommendation: use double precision for coordinates and WCS; standard frame
metadata; ALIGN's per-variable units; explicit coordinate validity; and atomic
four-stage requested/effective/observation-resolved/realized provenance. Reuse
compact ALIGN grid identity plus detector UID rather than routine per-sample ID
arrays. Full TOD carries coordinates and validity; mini TOD declares them
unavailable.

## Decision and repair sequence

Review D001-D005 first, then D007. Review D006 and D008 after the ALIGN seam
representation and timing facts are sufficiently fixed. Only after all eight
decisions are recorded may the coordinator select an exact AST repair base and
prepare a bounded repair task. No repair, Unity request, re-audit, integration,
or production expansion is authorized by this brief.
