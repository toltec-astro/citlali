# SCI-JINC — Internal Stage A Dossier

Status: internal implementation-informed scope evidence; never an author reference

Date: `2026-08-28`

Starting authority:
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`

Nothing in this dossier establishes the scientifically correct analytic JINC
family, parameter values, upstream coefficient, grouping, response,
conditioning bound, covariance approximation, output schema, implementation
conformity, validation result, achieved performance, readiness, or production
policy.

## Anti-Repetition Strategy

Stage A began with the package-neutral registry and the frozen SCI-MAP recovery
record. It recovered the frozen `SCI-MAP-002` independent core, every named
owner decision, the third-successor acceptance, later destination-ownership
work, the March memo-alignment note, the internal draft noise derivation, and
post-registry JINC convergence/validation history. No derivation or numerical
campaign was repeated.

The independent core is the reusable scientific basis. The eight owner
decisions supersede or specialize it. Source/audit/repair/validation material
is used here only to locate scope seams and is withheld from the future author.

## Apparent Current Transformation

The current repository exposes JINC as a selectable mapmaking method and
contains a mature, performance-sensitive coefficient-cache and footprint
accumulator. Apparent inputs include transformed detector samples,
coefficient/weight state, sample coordinates, stable array identity,
array-specific JINC shape parameters, map WCS/geometry, masks/flags, a
processing-filtered kernel-template stream, sample frequency, grouping, and
execution policy.

Apparent accumulation maintains separate signal numerator, signed
normalization denominator, quadratic coefficient accumulator, response
numerator, coefficient-squared time, and conditioning diagnostics. Apparent
finalization and writing expose normalized signal, working/formal weight,
kernel, temporal support, mask, empirical companions when available, and
provenance joins.

That is an implementation scope inventory, not a finding that any current
path conforms to the proposed scientific authority.

## Recovered Scientific/Implementation Separation

| Topic | Scientific authority recovered | Implementation/evidence material quarantined |
| --- | --- | --- |
| Signed estimator | Frozen core plus owner acceptance preserve `N/C` and conditional `C^2/Q` | Current accumulators, finalizers, names, and tests |
| Support | D003 support decision selects square cache and dual-use `r_max` | Cache allocation, radius arithmetic, boundary code, fixtures |
| Subpixel response | D003 subpixel decision selects phase-binned point evaluation | Lookup matrices, rounding/bin code, convergence tests |
| Conditioning | D003 conditioning decision selects dimensionless cancellation logic | Current summation method, fixed constants, diagnostic planes, repairs |
| Admission | D003 admission decision requires stable array identity and finite positive parameters | Current parsers, adapters, exception paths, coefficient checks |
| Temporal support | D003 coverage decision defines `sum(c^2/f_s)` | Current coverage plane, sample-rate source, writer labels |
| Formal support | D003 mask decision defines the authoritative JINC formal-support state | Current mask names, thresholds, empirical replacement paths |
| Response | D003 kernel decision defines processing-filtered template `K/C` | RTC template constructors, processing paths, kernel writers/consumers |
| Provenance | D003 provenance decision defines four one-way stages and joins | Current YAML/header/schema classes and repair history |
| Destination ownership | Owner-directed Stage A recovery retains unique resolved destination ownership | Parallel worker code, repair diff, re-audit, thread-count tests |

## Inherited TolTEC Numerical Realization — Evidence Only

At implementation revision
`fbfdc3479e7e61e2618bdf5ab81f6634df476e4c`, `jinc_mm.h` realizes the
dimensionless radius with nominal array wavelength divided by `45 m`:

| Array | Current `s_a` | Current `(a,b,c)` |
| --- | --- | --- |
| `a1100` | `(1.1 mm)/(45 m) = 5.042028597151 arcsec` | `(1.1,1.67,2.0)` |
| `a1400` | `(1.4 mm)/(45 m) = 6.417127305465 arcsec` | `(1.1,2.17,2.0)` |
| `a2000` | `(2.0 mm)/(45 m) = 9.167324722093 arcsec` | `(1.1,3.17,2.0)` |

Science and Beammap mode material currently carries `r_max=3.0`; Pointing and
OOF material carries `r_max=1.5`. The exact current source/config digests and
development revisions are preserved in `PRIOR_WORK.md`. The trace reaches the
initial `50 m` realization, later per-array representation, array-specific `b`
change, first recovered Pointing/OOF `1.5` occurrence, `50 m` to `45 m` change,
and typed-config adoption.

Classification: inherited implementation defaults with partially recoverable
development history; not current TolTEC scientific authority. Comments such
as `b (beam-size/3)`, “optimal (hopefully),” and “jinc optimization” are clues,
not authorization. No recovered scientific source authorizes the values for
TolTEC, and `45 m` has no approved interpretation as effective aperture,
illumination diameter, beam-derived diameter, or another physical quantity.

A future optimization study may test whether the inherited `b` values and the
`50 m` to `45 m` change attempted to track an effective TolTEC/LMT angular
response. This dossier does not adopt that hypothesis or derive a rule from
the values.

## Historical Memo And Analytic-Identity Disposition

`handoff/DEBUG_NOTES_2026-03-09_JINC_MEMO_ALIGNMENT.md` records a deliberate
move from a matched-amplitude-like normalization to “memo-style” signed
gridding and separates formal quadratic propagation from downstream filtering.
It also recommends later parameter tuning. The named underlying memo was not
found in reachable Git objects or the current workspace.

The subsequently supplied Schloerb memo content-binds the generic analytic
family and provides `s=lambda/D` precedent. ODQ-102B preserves the more general
`r'_a=r/s_a` concept, with explicit array-associated angular `s_a`, while
typing the TolTEC numerical realization unavailable and deferring optimization.
Neither the source audit nor current behavior is sanitized into scientific
authority by fiat.

## Current Frozen-Chain Boundary

Frozen PTC r0.5 owns its transformed product and declaration obligations but
leaves the exact ordinary MAP-facing coefficient family open. Frozen MAP
r0.7.1 consequently leaves its source-closed numerical route unavailable. A
JINC package cannot evade that ownership by calling an existing weight
inverse variance.

The exact ordinary `SCI-PTC_TO_SCI-MAP v0.1/r0.1` boundary demonstrates the
needed shape of a representation-independent handoff—product generation,
signal, retention, coefficient, response, covariance, coordinate, exposure,
policy, failure, and causes—but its MAP-owned profiles and one-hot placement
are not JINC authority. SCI-JINC needs an explicit corresponding boundary.

## Apparent Product And Consumer Families

| Family | Apparent role | Stage A treatment |
| --- | --- | --- |
| Observation signal map | Direct signed JINC estimate | In scope scientifically; schema and current values excluded |
| Working/formal weight | Quadratic-propagation or empirical-working companion | In scope only with exact estimator and assumptions; empirical meaning belongs to NOI |
| Kernel/response plane | Processing-filtered template transformed through JINC | In scope as response identity; achieved fidelity unassessed |
| Coverage/time plane | Coefficient-squared effective integration time | In scope with corrected meaning; not exposure/validity |
| Formal-support mask | JINC numerical/formal validity | In scope; distinct from empirical policy and ordinary MAP aliases |
| Noise realizations/variance/significance | Downstream empirical uncertainty products | SCI-NOI boundary; not JINC authority |
| Filtered products | Wiener/convolution outputs | SCI-FLT boundary; preserve raw JINC parent |
| Beammap and fruit-loop products | Detector-resolved inference and iterative reuse | BEAM/FRUIT consumers; do not enlarge JINC scope |
| Coadd products | Later combination of normalized observations | No recovered independent JINC coadd authority; owner question |

## Destination Ownership Evidence

The post-registry record resolves a bounded current route to unique
per-detector map destinations with sequential scan invocation and eager
pre-mutation checks. A later independent re-audit and reconstruction record
contain tests and execution claims. For Stage A, only the abstract invariant
is retained: the complete destination identity and uniqueness are established
before mutation, or the operation fails atomically.

The raw handoff, repair, re-audit, reconstruction, thread-count results, and
current grouping behavior remain excluded. They are later conformance evidence
and cannot prove general correctness or performance.

## Later Evidence Inventory

Later refs include a targeted working-support repair and owner-retrieved Unity
record, an APT/ALIGN/JINC convergence audit, reconstructed destination
ownership, native-execution changes, fruit-loop compatibility changes, local
Stage-7 completion records, and bounded map comparisons. These are preserved
at exact revisions and digests in `PRIOR_WORK.md`.

They may inform a future authorized conformity or validation plan. They do not
enter the author packet, close an owner question, establish response fidelity,
authorize a parameter choice, or support an achieved-performance/readiness/
production claim in this package.

## Information Firewall

The proposed author receives only:

- the owner-approved Scope Brief;
- the exact frozen implementation-independent core with a sanitized
  supersession cover; and
- the content-bound conventions/ownership extract.

The author does not receive this dossier, the complete recovery record,
current source/config/product schemas, the raw owner-decision documents,
audit/findings/repairs/re-audits/tests, the March memo note, the internal noise
memo, reductions, Unity evidence, comparisons, integration state, branch
history, or production status. If a scientific fact is missing, the author
must ask the owner rather than inspect excluded material.
