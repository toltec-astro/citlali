# SCI-POINT Scientific-Owner Decision Ledger

Status: Stage A owner-review ledger; no open recommendation is authoritative

Scientific owner: Grant Wilson

## Decisions Already Governing

| ID | State | Decision |
| --- | --- | --- |
| `SCI-POINT-SCOPE-D001` | decided `2026-09-02` | Launch a separate SCI-POINT Stage A package for bright, approximately central Pointing sources. |
| `SCI-POINT-SCOPE-D002` | decided `2026-09-02` | Recover and preserve working prior art before deriving new science; do not reinvent a working wheel. |
| `SCI-POINT-SCOPE-D003` | decided `2026-09-02` | Per-detector Beammap fitting is already SCI-BEAM authority and is outside SCI-POINT. |
| `SCI-POINT-SCOPE-D004` | decided `2026-09-02` | Blank-field faint distributed-source detection/fitting is not part of this package. |
| `SCI-POINT-SCOPE-D005` | inherited | OOF optical inference remains separate even where it reuses Pointing maps or fitting code. |
| `SCI-POINT-ODQ-001` | decided `2026-09-02` | SCI-POINT v0.1 ends at authoritative per-array measurements; any cross-array aggregate belongs to the named pointing-support producer with complete member/weight/covariance/failure/provenance identity. |
| `SCI-POINT-ODQ-002` | decided `2026-09-02` | POINT publishes measured displacement only; the pointing-support producer owns aggregation/sign/telescope-offset composition/selection/correction publication and AST owns application. |
| `SCI-POINT-ODQ-003A` | decided `2026-09-02` | FRUIT is not a separate POINT parent type; a terminal FRUIT result retains its exact MAP/JINC/FLT map type plus complete FRUIT lineage. |
| `SCI-POINT-ODQ-003B` | decided `2026-09-02` | Coadd parents are outside SCI-POINT base v0.1. |
| `SCI-POINT-ODQ-003` | decided `2026-09-02` | Observation-local MAP, JINC, FLT-FIXED, and FLT-MATCHED are eligible as distinct explicit routes with no automatic selection, substitution, equivalence, or fallback. |
| `SCI-POINT-ODQ-004` | decided `2026-09-02` | Adopt the established six-parameter elliptical-Gaussian fit as `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`; no additional profile family enters base v0.1. |
| `SCI-POINT-ODQ-005` | decided `2026-09-02` | Preserve the established configurable center/search, weighted-peak initialization, global fallback, bounded fit domain, and parameter constraints with explicit requested/effective/realized state. |
| `SCI-POINT-ODQ-006` | decided `2026-09-02` | Each requested array fit is independently atomic and reports complete, diagnostic-only, or unavailable; one array failure does not erase siblings or create a POINT-owned whole-observation result. |
| `SCI-POINT-ODQ-007` | decided `2026-09-02` | Require available marginal formal parameter errors with honest method/limitation labels; joint covariance may be unavailable and must not be treated as zero, diagonal, or independence. |
| `SCI-POINT-ODQ-008` | decided `2026-09-02` | Fitted amplitude, widths, and angle are required fit-result components and, with centroid and fit state, telescope/observing-condition quality-control metrics under exact processed-map meanings and limitations. |
| `SCI-POINT-ODQ-009` | decided `2026-09-02` | Keep fit completeness, pointing-support displacement, telescope/observing QC, and CAL/TolProj amplitude policies separately owned; VAL only registers/evaluates; exact collision-free mechanics are assigned to Stage B for later owner approval. |

## Ordered Decision Walkthrough

### `SCI-POINT-ODQ-001` — Does POINT own a cross-array aggregate? — **decided**

**Question.** Does base v0.1 end with the per-array displacement
measurements, or also define one observation-level aggregate displacement?

**Recovered working behavior.** Citlali publishes per-array rows. Current
TolTECA later forms an arithmetic mean of the rows while constructing a
correction record.

**Approved decision.** Keep the per-array measurements as the terminal
authoritative SCI-POINT products and leave cross-array aggregation in the
named pointing-support producer. This preserves the working boundary instead
of relocating or redesigning it. A future POINT aggregate may be added as a
separate method if scientific use requires it.

**Consequence.** `POINT-AGGREGATE` is outside base v0.1, while the
downstream producer must state its exact member, weighting, covariance,
failure, and provenance policy.

Approval record:
`SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-09-02.md`.

### `SCI-POINT-ODQ-002` — Measurement versus correction construction — **decided**

**Question.** Does POINT ever change sign or compose telescope user/paddle
offsets to publish a correction candidate?

**Approved decision.** No. POINT publishes measured source displacement in the
declared tangent basis. The selected pointing-support producer owns
aggregation, measurement-to-correction sign, telescope-offset composition,
record selection/native support, and correction-record publication. AST owns
conforming application. The record must retain exact POINT ancestry.

Approval record:
`SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-09-02.md`.

### `SCI-POINT-ODQ-003` — Admitted observation-local parent routes — **decided**

**Question.** Which exact observation-local parent families are admitted in
v0.1: ordinary MAP, JINC, FLT-FIXED, and FLT-MATCHED?

**Approved decision.** All four families are scientifically eligible. POINT
is route-parameterized but never route-agnostic: each family is a distinct
explicit method route with complete parent identity, and POINT may not select,
substitute, equate, or fall back among them automatically. Scientific
eligibility does not establish numerical availability; every exact predecessor
and compatibility boundary must still be present and bound.

**Preserved subdecisions.** FRUIT is lineage on an exact terminal map type, not
a separate parent family. Coadds and intermediate FRUIT iterations are outside
base v0.1.

Approval records:
`SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-09-02.md` and
`SCIENTIFIC_OWNER_ODQ_003A_003B_DIRECTION_2026-09-02.md`.

### `SCI-POINT-ODQ-004` — Compatibility estimator — **decided**

**Question.** Is the established six-parameter elliptical Gaussian the base
v0.1 normative estimator?

**Approved decision.** Yes. Adopt it as
`POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`. Do not introduce another
profile family in base v0.1. Stage B must state its existing zero-background
signal model, parameter meaning, weighting, support, initialization,
constraints, degeneracies, response interpretation, failure behavior, and
formal uncertainty meaning without copying implementation or silently
changing the estimand.

Approval record:
`SCIENTIFIC_OWNER_ODQ_004_APPROVAL_2026-09-02.md`.

### `SCI-POINT-ODQ-005` — Center, search, support, and constraints — **decided**

**Question.** Which parts of the established central search, weighted-peak
seed, bounded fit domain, global fallback, and amplitude/FWHM/angle bounds are
scientific method identity rather than implementation detail?

**Approved decision.** Preserve the established configurable expected center,
central search, weighted-peak initialization, global-search fallback, bounded
fit domain, and amplitude/width/angle constraints. Every requested, effective,
and realized value or named state is explicit method identity. Report any
realized global fallback, resolve numeric sentinels to named effective states,
and do not freeze one universal numerical configuration or introduce a new
search algorithm.

Approval record:
`SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-09-02.md`.

### `SCI-POINT-ODQ-006` — Per-array acceptance and partial success — **decided**

**Question.** What makes one array result scientifically usable, and what
happens when only a subset succeeds?

**Approved decision.** Treat each requested array fit independently as
complete for the stated use, diagnostic-only with the excluded use/claim and
reason, or unavailable with reason. A failed or unavailable array does not
erase sibling results, and POINT does not synthesize missing results or create
a whole-observation success result. A constraint-, support-, or uncertainty-
limited numerical fit may remain diagnostic-only under the later exact
named-use policy. Any downstream aggregate owns and declares whether and how
a partial set is admitted.

Approval record:
`SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-09-02.md`.

### `SCI-POINT-ODQ-007` — Formal covariance baseline — **decided**

**Question.** Must base v0.1 publish full joint parameter covariance, or may it
publish the established marginal formal errors with covariance explicitly
unavailable?

**Approved decision.** Preserve the established marginal formal parameter
errors as the required compatibility representation when available, with
honest method, assumption, domain, conditioning, and limitation labels. Joint
covariance may be unavailable; absence is not zero, diagonal covariance, or
independence and does not invalidate the fit. A use requiring joint covariance
is unavailable unless its owner authorizes another treatment. Later joint-
covariance, astrometric, empirical-repeatability, or NOI companions are
separately versioned products and do not rewrite the original fit claims.

Approval record:
`SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-09-02.md`.

### `SCI-POINT-ODQ-008` — Amplitude and effective shape — **decided**

**Question.** Are fitted amplitude and shape required POINT products or merely
incidental diagnostics?

**Approved decision.** Retain fitted amplitude, two widths, and angle as
required numerical fit-result components. Together with centroid and honest
fit state, they are authorized quality-control metrics for telescope
performance and observing conditions. Amplitude remains conditional on exact
parent unit/calibration/normalization/response and is not universal flux;
shape remains effective under the exact processed-map response and is not an
intrinsic beam or SCI-BEAM result. Qualified values retain their constraint,
support, and uncertainty state. CAL/TolProj and exact QC interpretation remain
separately owned named uses; the metrics alone do not prove a unique physical
cause.

Approval record:
`SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-09-02.md`.

### `SCI-POINT-ODQ-009` — Named-use VAL profiles — **decided**

**Question.** Which POINT-owned profiles distinguish a complete per-array fit,
a displacement usable by a pointing-support producer, a CAL amplitude use,
and a diagnostic-only result?

**Approved decision.** Use separate named-use policies: POINT owns per-array
fit completeness; the pointing-support producer owns displacement admission
for correction construction; the named telescope/observing QC process owns
parameter-QC admission, references, thresholds, aggregation, and actions; and
CAL/TolProj owns amplitude admission for photometric transfer. One immutable
result may have different outcomes for different uses, and diagnostic-only is
an explicit use-specific outcome rather than a universal bad flag. VAL only
registers/evaluates. The Stage B author may define exact collision-free
identifiers, facts, actions, and mechanics subject to final owner approval and
may not introduce a base-v0.1 aggregate profile.

Approval record:
`SCIENTIFIC_OWNER_ODQ_009_APPROVAL_2026-09-02.md`.

## Dispatch Gate

All bounded owner questions are decided. Stage B remains blocked until the
candidate documents are repaired into a closed sanitized packet, the exact
author input bytes are content-bound, and the owner explicitly approves that
packet for dispatch.
