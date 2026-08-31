# ADR 0021: AST scan-motion velocity and validity

> Canonical numbering note: this decision first appeared as divergent ADR
> 0018 at `doc/adr/0018-ast-scan-motion-velocity-and-validity.md`, introduced by
> `5f0196d147723d56fcc3d4df6e984fdb0db8cb1c`. It entered canonical application
> ancestry as ADR 0021.
> The historical number remains a provenance locator only.

Status: accepted numerical scientific-owner authority 2026-08-30; supported-
family and physical-membership clauses partially superseded by ADR 0023

Decision owners: Citlali project owner and scientific owner

## Context

The frozen SCI-AST package defines coordinate realization, spherical topology,
role-local validity, and exact ALIGN parentage. It deliberately leaves the
producer field registry and producer-state interpretation open, and it does not
define the scan-motion derivative, telemetry-defect rule, physical-scan
membership, or actual scan maximum required by the first nonidentity RTC
planner in ADR 0019.

Those missing facts cannot be inferred from similar field names, a configured
scan-rate header, a processing chunk, a direct finite difference, or detector
data. They need a bounded scientific-owner assignment before AST can publish a
motion product and RTC can consume it.

## Decision

The controlling bounded authority is
[`wp7-ast-scan-motion-v1`](../WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md).
For real TolTEC `Science`/`Lissajous`, 50 Hz, J2000 inputs in its exact supported
family:

- physical science-scan identity is `(observation, subobservation, scan)`, not
  a processing chunk or constant producer position field;
- `Data.TelescopeBackend.SourceRaAct` and `SourceDecAct` are the realized
  boresight direction in J2000 radians at the exact producer time;
- native continuity requires strictly increasing finite time and adjacent
  intervals `0 < dt <= 0.030 s`, with equality admitted;
- an eleven-record robust component-wise pair-slope/intercept check marks a
  record defective only when its radial intercept residual is strictly greater
  than `2.0 arcsec`; equality remains valid;
- an eleven-record unweighted quadratic least-squares fit in the canonical
  local J2000 east/north tangent basis supplies the two velocity components,
  and their Euclidean norm is the scalar speed in arcseconds per second;
- symmetric support, gaps, topology, invalid direction, defects, rank failure,
  and nonfinite results have exact typed validity and causes; v1 introduces no
  one-sided endpoint derivative and never crosses invalid support;
- AST publishes the uninflated actual maximum over valid physical-scan records
  with speed `v >= 1 arcsec/s`, including its maximizing record identity, while
  RTC separately owns occurrence admission and its slow-motion cause; and
- ALIGN maps the immutable raw AST motion role to each network's independent
  occurrence/time axis without constructing a common analysis grid. The mapped
  view retains source records, times, weights, network occurrence, validity,
  cause, and support.

The product is a compact immutable role. It owns genuinely derived defect,
derivative, scalar-speed, compact-support, validity/cause, and scan-summary
facts and references immutable producer and ALIGN facts through bounded typed
handles. It does not duplicate full axes or pointing planes, create per-cell
identity objects, or add generalized provenance machinery.

The exact field registry, membership rule, continuity boundary, defect
operator and threshold, derivative, cause vocabulary, scan maximum, mapping,
and identity binding are versioned together as `wp7-ast-scan-motion-v1`.
Changing them requires a named successor authority and new evidence.

ADR 0023 supplies that named successor for the supported-family and physical-
membership clauses only. Its `wp7-ast-scan-motion-v2` authority preserves this
ADR's numerical operator and all unaffected semantics.

## Consequences

- The bounded AST scientific-authority prerequisite in ADRs 0016 and 0017 is
  closed. The raw product and network-specific mapped views pass local and
  representative-data gates, and exact repair SHA
  `abb33fdb9e45352190d2e55592cc5eba967993f2` passed fresh independent
  exact-SHA conformance review with no findings.
- The observation `(152390, 0, 2)` value `221.40490828695155 arcsec/s` at
  telescope record `16973` is an authorized truthful AST diagnostic for that
  exact bounded scope. ADR 0022 supersedes using it as a whole-scan RTC
  admission value. It is not a default or substitute for producing
  authoritative AST motion for another scan.
- Independent network timing under ADR 0018 remains unchanged. A synchronous
  cross-network consumer must separately request an ALIGN-owned common-analysis-
  grid relation.
- Certified filter-bank, representative PSD, native-rate map/OOF comparison,
  and line-path prerequisites remain separate and pending.
- Frozen SCI-AST text remains historical authority. This ADR closes only its
  explicitly open facts for this bounded scan-motion role and does not reopen
  ordinary coordinate realization, pointing corrections, or another scan
  program.

## Evidence and authority

- [Approved owner decision packet](../WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
- [Frozen-to-successor authority crosswalk](../WP7_AST_SCAN_MOTION_AUTHORITY_CROSSWALK_2026-08-30.md)
- [Network-specific timing decision](0018-network-specific-timing-and-common-analysis-grid.md)
- [Scan/array RTC planning decision](0019-scan-array-rtc-bandwidth-planning.md)
- [Pre-certified filter-bank decision](0020-precertified-rtc-filter-bank-and-science-error-budgets.md)
- `AstScanMotionProduct` public contract at
  `include/citlali/core/pipeline/ast_scan_motion.h` in preserved divergent
  snapshot `49fe73e757daa1885cd23127e8441cba47e648d2`; canonical replay is pending
- ALIGN network-mapped view contract at
  `include/citlali/core/pipeline/ast_scan_motion_alignment.h` in the same
  preserved divergent snapshot; canonical replay is pending
- [Representative acceptance package](../../handoff/WP7_AST_SCAN_MOTION_ACCEPTANCE_PACKAGE_2026-08-30.md)
