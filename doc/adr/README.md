# Citlali Architecture Decision Records

This directory contains durable decisions that are consequential, easy to
misunderstand, and expensive to reconstruct from Git history. The current
software map remains [`../ARCHITECTURE.md`](../ARCHITECTURE.md), scientific
semantics remain [`../SCIENTIFIC_CONVENTIONS.md`](../SCIENTIFIC_CONVENTIONS.md),
and phase sequencing remains [`../REFACTOR_STATUS.md`](../REFACTOR_STATUS.md).

An ADR records why a decision exists and what would supersede it. It does not
duplicate changing implementation inventories or validation snapshots.

## Records

| ADR | Status | Decision |
| --- | --- | --- |
| [0001](0001-config-state-transitions.md) | Accepted | Immutable request to effective plan to observation-resolved and realized state, with a one-way legacy adapter |
| [0002](0002-reduction-result-and-required-output-failures.md) | Accepted | Structured reduction result, required-output failure propagation, and CLI-only process exit |
| [0003](0003-session-lifecycle-and-engine-compatibility.md) | Accepted | Sequential session lifecycle and `Engine` as a frozen compatibility boundary |
| [0004](0004-compiled-boundary-and-header-policy.md) | Accepted | Evidence-driven compiled boundaries and header/hot-path policy |
| [0005](0005-defer-measured-r-channel-execution.md) | Accepted | Preserve measured R-channel structure while deferring execution until its contract is approved |
| [0006](0006-fruit-loop-restart-checkpoint.md) | Accepted | Resume fruit loops from a required state-complete iteration checkpoint with absolute iteration identity and fail-closed compatibility checks |
| [0007](0007-observe-only-coherent-raw-iq-event-sidecar.md) | Accepted | Score versioned network-specific coherent raw-I/Q modes for every present network at shared RTC-seeded epochs without changing science data |
| [0008](0008-application-mainline-and-build-adaptation-lanes.md) | Superseded in part by 0014 | Keep application development authoritative on one mainline while isolating successor-build adaptation |
| [0009](0009-science-map-bundle-admission-and-validity.md) | Accepted | Admit complete full-precision science-map bundles atomically, retain centered-integer `L = I` coaddition with nonprecision coefficients, and persist distinct support and validity facts |
| [0010](0010-canonical-baseline-apt-v1.md) | Accepted; candidate implementation unactivated | Define a typed Citlali-produced baseline APT with artifact-local UID, embedded raw-channel relation, distinct semantic/envelope/transport identities, and receipt-last publication |
| [0011](0011-canonical-observation-apt-contract.md) | Accepted; candidate implementation unactivated | Replace row-position correspondence with occurrence-scoped target/seed relations and publish one observation-specific canonical ECSV plus receipt, with complete target/relation records embedded |
| [0012](0012-canonical-apt-v2-compact-normalization.md) | Accepted owner-directed repair; Citlali verified locally, cross-repository gates pending | Supersede new v1 issuance with compact normalized ECSV bundles, occurrence-scoped TolAPT matching, v2-only guardians, and one root receipt |
| [0013](0013-bounded-native-scientific-provenance.md) | Superseded in part by 0016; bounded publication remains authoritative | Separate runtime sample state, bounded canonical native provenance, and opt-in bounded debug tracing; regenerate deterministic per-sample consequences instead of serializing them |
| [0014](0014-spack-build-foundation.md) | Accepted | Use Spack as the successor dependency/environment authority while preserving the full refactored application and fallback build |
| [0015](0015-release-bundle-contract.md) | Accepted | Bind immutable first-party sources to one host-path-free Spack lock per supported platform profile, with explicit evidence and build-cache trust |
| [0016](0016-stage-boundary-native-runtime-state.md) | Accepted owner-directed repair; local implementation complete, Unity science acceptance pending | Retain exact numerical state and compact masks at stage boundaries without a per-detector-sample operation narrative |
| [0017](0017-wp7-timestream-successor.md) | Accepted; program active, application implementation held for governance reconciliation | Implement the scientifically closed WP-7.1 timestream contract by bounded replay on canonical application ancestry |
| [0018](0018-network-specific-timing-and-common-analysis-grid.md) | Accepted; bounded divergent implementation and exact-SHA review passed | Preserve network-specific native timing and make common analysis grids explicit derived relations |
| [0019](0019-scan-array-rtc-bandwidth-planning.md) | Accepted in part; superseded in part by 0020 and 0022 | Plan RTC bandwidth and decimation at scan/array scope without treating one historical factor as universal |
| [0020](0020-precertified-rtc-filter-bank-and-science-error-budgets.md) | Accepted scientific budgets; certification artifacts pending | Select only pre-certified RTC filters under mapped-response and broadband-alias budgets |
| [0021](0021-ast-scan-motion-velocity-and-validity.md) | Accepted numerical authority; family/membership clauses superseded in part by 0023 | Define AST scan-motion velocity, validity, and maximum-speed diagnostics |
| [0022](0022-occurrence-level-rtc-upper-speed-admission.md) | Accepted; census evidence partial and production implementation pending | Apply upper-speed admission to network occurrences rather than selecting a whole-scan mode from a sparse maximum |
| [0023](0023-ast-route-family-motion-membership.md) | Accepted bounded authority; divergent conformance review passed | Extend AST scan-motion membership to the approved Pointing, OOF, Science, and Beammap route families |

Numbers are never reused. A materially different decision adds a new ADR and
marks the old record superseded; do not rewrite the historical rationale.

The WP-7 records above first appeared with colliding numbers 0014--0020 on the
preserved divergent implementation branch. Their canonical numbers are
0017--0023. Each record identifies its original path and source commit; the
historical numbers remain provenance locators and are not canonical aliases.
