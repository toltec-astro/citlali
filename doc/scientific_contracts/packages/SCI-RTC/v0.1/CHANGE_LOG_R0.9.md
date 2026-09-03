# SCI-RTC v0.1/r0.9 Change Log

Date: 2026-08-20

R0.9 is a bounded scientific-contract cleanup implementing binding owner
Decisions 1--8 in `SCIENTIFIC_OWNER_DECISIONS_R0.9.md`. It does not add a
numerical default, alter the established raw $x/r$ or CAL boundary, reopen the
r0.8 level-shift model, or renumber a normative identifier.

## Changes

- Replaced role-partitioned operation availability with an explicit RTC
  application context. All v0.1 operation classes remain admitted across
  contexts until evidence or policy disqualifies them; admission is distinct
  from selection, numerical resolution, qualification, and execution.
- Defined the one-way `RTCApplicationContext` → `RTCResolvedPlan` →
  `RTCRealizedRecord` lifecycle and prohibited later-state rewriting.
- Made the atomic RTC bundle consumer-neutral; named consumers bind required
  members and optional diagnostic detail without changing schema or signal
  meaning.
- Reaffirmed distinct upstream IQ-to-$x/r$ and ALIGN authority, exact paired
  identity, independent $x/r$ validity, and fail-closed coordinate authority.
- Strengthened typed, cause-preserving non-finite handling.
- Simplified covariance/uncertainty obligations: every claim discloses included
  and excluded components and correlations; qualified partial claims are
  allowed, unknowns are unavailable rather than zero, and only complete
  declared coverage is total.
- Required selected despiking to modify accepted target $x$ cells or record an
  explicit selected failure/no-correction disposition. Normal production uses
  compact treatment state and spike-population counts/characteristics; full
  event/donor manifests are optional, inert diagnostic detail.
- Added owner-ledger entries `SCI-RTC-OWNER-076`--`083` and decision-log entries
  `RTC-SCI-D010`--`D017` plus `RTC-DRAFT-D013`.
- Updated the rationale, shared normative core, engineering guidance,
  falsifiers, exact crosswalk, verifier, and canonical PDFs.

## Preserved boundaries

- Normative inventory remains 38 definitions, 37 equations, 12 assumptions,
  108 requirements, and 71 predictions.
- Decision 9 remains unchanged: additive level shifts, finite physical-time
  transition support, unmodeled transition cells, optional valid additive
  plateau correction, and no gain-change model in RTC v0.1.
- Implementation conformity, observational performance, science-impact
  qualification, and production readiness are not claimed.

## Freeze disposition

On `2026-08-20`, after the fresh r0.9 consistency review passed, the scientific
owner stated exactly, “Freeze SCI-RTC v0.1/r0.9.” The status-only freeze changes
no normative content or ledger state. The canonical PDFs were republished to
carry the frozen status; implementation conformity remains unassessed.
