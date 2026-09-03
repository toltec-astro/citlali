# SCI-POINT v0.1 Scientific-Owner Decisions For Authorship

Identity: `SCI-POINT_OWNER_DECISIONS v0.1/r0.3`

Scientific owner: Grant Wilson

Status: ODQ-001 through ODQ-009 decided; packet-byte approval pending

## Closed Decisions

| ID | Binding scientific decision |
| --- | --- |
| `SCI-POINT-ODQ-001` | Base v0.1 ends with authoritative per-array measurements. A named pointing-support producer owns any cross-array aggregate and its members, weights, covariance/dependence, partial-set, failure, method, and provenance policy. |
| `SCI-POINT-ODQ-002` | POINT publishes measured displacement only. The pointing-support producer owns aggregation, measurement-to-correction sign, telescope user/paddle-offset composition, selection, native support, and correction-record publication; AST owns application. |
| `SCI-POINT-ODQ-003A` | FRUIT is lineage on an exact terminal MAP, JINC, FLT-FIXED, or FLT-MATCHED product, not another POINT parent type. Intermediate FRUIT iterations are excluded. |
| `SCI-POINT-ODQ-003B` | Coadd parents are outside base v0.1. |
| `SCI-POINT-ODQ-003` | Observation-local MAP, JINC, FLT-FIXED, and FLT-MATCHED are eligible as distinct explicit routes. POINT shall not automatically select, substitute, equate, or fall back among them. Scientific eligibility does not establish numerical availability. |
| `SCI-POINT-ODQ-004` | Adopt the established zero-background six-parameter elliptical-Gaussian fit as `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`. No additional source-model family enters base v0.1. |
| `SCI-POINT-ODQ-005` | Preserve configurable expected-center/central-search, weighted-peak initialization, global fallback, bounded fit-domain, and amplitude/width/angle constraints. Requested, effective, and realized state is explicit; fallback and sentinel resolution are never hidden. |
| `SCI-POINT-ODQ-006` | Each requested array fit remains scientifically atomic. Producer realization, component identifiability, and named-use evaluation are separate. Named-use evaluation preserves request, applicability, eligibility, and realization axes; `diagnostic_only` is neither a producer state nor eligibility value and is represented as the prescribed consumer action `diagnostic_display_only`. One array failure does not erase siblings; POINT does not synthesize missing arrays or publish whole-observation success. Partial-set aggregate admission is downstream-owned. |
| `SCI-POINT-ODQ-007` | Publish established marginal formal parameter errors when available with honest method and limitation labels. Joint covariance may be unavailable; absence is not zero, diagonal covariance, or independence. Later uncertainty estimates are separate versioned companions. |
| `SCI-POINT-ODQ-008` | Fitted amplitude, widths, and angle are required fit-result components and, together with centroid and fit state, telescope/observing-condition QC metrics. Preserve exact processed-map meanings; do not promote them to universal flux, intrinsic beam, SCI-BEAM authority, or unique causal diagnosis. |
| `SCI-POINT-ODQ-009` | POINT owns fit completeness; the pointing-support producer owns displacement admission; the named QC process owns parameter-QC policy; CAL/TolProj owns photometric-transfer amplitude admission. VAL only registers/evaluates. Exact collision-free profile mechanics are author-assigned for final owner review; no aggregate profile enters base v0.1. |

## Method-Authority Dispositions

| Identity | Binding state and consequence |
| --- | --- |
| `POINT-COMPATIBILITY-METHOD v0.1` | `unavailable_pending_separate_owner_approval`. No width convention, objective, fit-weight meaning, search/fallback procedure, or solution rule may be inferred. Every numerical fit and fit-derived product remains unavailable. |
| `POINT-FORMAL-ERROR-METHOD v0.1` | `unavailable_pending_separate_owner_approval`. When compatibility fitting later becomes available, fit values may exist while marginal formal errors, joint covariance unless independently authorized, formal standardization, and uses requiring formal uncertainty remain unavailable. |
| `POINT-FULL-MAP-RMS-METHOD v0.1` | `unavailable_pending_separate_owner_approval`. Its absence blocks only `fitted_amplitude_over_full_map_rms` and legacy alias `sig2noise`; it does not block an otherwise authorized fit, displacement, effective-shape role, or formal error. |

These three authorities are scientifically and lifecycle-distinct. A separate
quarantined recovery may produce sanitized records for exact owner review; it
does not authorize the Stage B author to inspect implementation or infer a
method.

## Author Freedom

The author may choose clear notation and canonical parameter names,
typed structures, requirement/prediction partitioning, and collision-free
profile identifiers and mechanics when those choices faithfully express the
closed decisions. The author may use a positive-definite shape matrix
symbolically, but may not choose the legacy width convention, numerical angle
gauge, objective, weighting, search/fallback/association procedure,
formal-error method, or full-map-RMS method.
If two other scientifically different choices remain possible, the author
must return a precise question rather than select one silently.

## Non-Effects

These decisions do not establish numerical route availability,
implementation conformity, validation, uncertainty coverage, achieved
pointing performance, readiness, production state, or Stage B authorization.
