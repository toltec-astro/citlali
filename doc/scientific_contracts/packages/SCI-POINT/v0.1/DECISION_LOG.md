# SCI-POINT v0.1 Decision Log

Status: Stage A process and disposition log

| ID | Date | Decision |
| --- | --- | --- |
| `SCI-POINT-STAGEA-D001` | 2026-09-02 | Create SCI-POINT as the narrow bright-source Pointing package rather than a generic source-fitting or combined Pointing/OOF package. |
| `SCI-POINT-STAGEA-D002` | 2026-09-02 | Treat prior-work recovery and the working-wheel adoption register as mandatory anti-repetition controls. |
| `SCI-POINT-STAGEA-D003` | 2026-09-02 | Preserve SCI-BEAM ownership of all per-detector Beammap fitting and associated PSF/sensitivity/APT products. |
| `SCI-POINT-STAGEA-D004` | 2026-09-02 | Defer historical SCI-SRC blank-field detection/catalog work and future faint distributed-source fitting. |
| `SCI-POINT-STAGEA-D005` | 2026-09-02 | Supersede the unlaunched SCI-MODE grouping; preserve its CAL/AST handoffs as evidence and defer OOF-specific material. |
| `SCI-POINT-STAGEA-D006` | 2026-09-02 | Quarantine all implementation, configuration, schema, audit, validation, TolTECA, and TolProj observations from future Stage B authorship. |
| `SCI-POINT-STAGEA-D007` | 2026-09-02 | Treat MAP, JINC, FLT-FIXED, FLT-MATCHED, and terminal FRUIT as non-equivalent candidate parent families; admit none during Stage A. |
| `SCI-POINT-STAGEA-D008` | 2026-09-02 | Hold cross-array aggregation and correction construction for explicit owner decisions rather than silently ratifying current coupling. |
| `SCI-POINT-STAGEA-D009` | 2026-09-02 | Record owner-approved ODQ-001: SCI-POINT v0.1 ends with per-array measurements; the named pointing-support producer owns any cross-array aggregate and must preserve exact member/policy/provenance identity. |
| `SCI-POINT-STAGEA-D010` | 2026-09-02 | Record owner-approved ODQ-002: POINT publishes measured displacement only; aggregation/sign/telescope-offset composition/selection/correction publication remain with the pointing-support producer and application remains with AST. |
| `SCI-POINT-STAGEA-D011` | 2026-09-02 | Record owner ODQ-003A: FRUIT is terminal lineage on an exact MAP/JINC/FLT map type, not a separate POINT parent family. |
| `SCI-POINT-STAGEA-D012` | 2026-09-02 | Record owner ODQ-003B: coadd parents are deferred beyond SCI-POINT base v0.1. |
| `SCI-POINT-STAGEA-D013` | 2026-09-02 | Record owner-approved ODQ-003: MAP, JINC, FLT-FIXED, and FLT-MATCHED are eligible as distinct observation-local routes with no automatic selection, substitution, equivalence, or fallback; exact numerical availability and binding remain separate gates. |
| `SCI-POINT-STAGEA-D014` | 2026-09-02 | Record owner-approved ODQ-004: adopt the established six-parameter elliptical-Gaussian Pointing fit as `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1` without adding another base-v0.1 profile family or authorizing numerical redesign. |
| `SCI-POINT-STAGEA-D015` | 2026-09-02 | Record owner-approved ODQ-005: preserve the established configurable center/search, weighted-peak initialization, global fallback, bounded fit domain, and parameter constraints while making requested/effective/realized state and every fallback or sentinel resolution explicit. |
| `SCI-POINT-STAGEA-D016` | 2026-09-02 | Record owner-approved ODQ-006: each requested array fit is independently complete, diagnostic-only, or unavailable; sibling results survive a failure, POINT does not impute or publish whole-observation success, and partial-set aggregate policy remains downstream. |
| `SCI-POINT-STAGEA-D017` | 2026-09-02 | Record owner-approved ODQ-007: require available marginal formal parameter errors with honest method and limitation labels; permit joint covariance to be unavailable without implying zero, diagonal covariance, or independence; attach later uncertainty products as versioned companions. |
| `SCI-POINT-STAGEA-D018` | 2026-09-02 | Record owner-approved ODQ-008: fitted amplitude, widths, and angle are required fit-result components and, together with centroid and fit state, authorized telescope/observing-condition QC metrics; preserve exact processed-map meanings and prohibit automatic flux, intrinsic-beam, or unique-cause claims. |
| `SCI-POINT-STAGEA-D019` | 2026-09-02 | Record owner-approved ODQ-009: assign fit completeness to POINT, displacement admission to the pointing-support producer, telescope/observing QC policy to the named QC process, and photometric-transfer amplitude admission to CAL/TolProj; VAL only registers/evaluates and Stage B defines exact collision-free mechanics for final owner review. |
| `SCI-POINT-STAGEA-D020` | 2026-09-02 | Assemble the closed decisions and sanitized scientific boundaries as the exclusive nine-object `SCI-POINT_AUTHOR_PACKET_MANIFEST v0.1/r0.1` review candidate, bind every admitted object and the manifest by SHA-256, and produce a deterministic eleven-file `.tar.gz`; preparation does not approve the bytes or authorize Stage B dispatch. |
| `SCI-POINT-STAGEA-D021` | 2026-09-02 | Record the final lifecycle repair: preserve independent per-array atomicity while separating producer lifecycle, component identifiability, and named-use disposition; `diagnostic_only` is not a producer state. |
| `SCI-POINT-STAGEA-D022` | 2026-09-02 | Record the owner disposition that `POINT-COMPATIBILITY-METHOD v0.1` and `POINT-FORMAL-ERROR-METHOD v0.1` are scientifically/lifecycle-distinct and `unavailable_pending_separate_owner_approval`; the first blocks numerical fits and fit-derived products, while the second alone blocks formal uncertainty and dependent uses. |
| `SCI-POINT-STAGEA-D023` | 2026-09-02 | Create separate quarantined recovery briefs for the two missing authorities without launching recovery or admitting recovery material to implementation-blind authorship. |
| `SCI-POINT-STAGEA-D024` | 2026-09-02 | Supersede the r0.1 author candidate with the exclusive 33-object `SCI-POINT_AUTHOR_PACKET_MANIFEST v0.1/r0.2`, adding exact conditional boundary schemas, state tables, downstream envelopes, and prediction obligations while preserving the immutable r0.1 archive; r0.2 remains unapproved and Stage B remains unlaunched. |
| `SCI-POINT-STAGEA-D025` | 2026-09-02 | Make named-use evaluation SCI-VAL-congruent: preserve request/applicability/eligibility/realization, restrict eligibility to eligible/ineligible/decision_unavailable, and represent diagnostic limitation as prescribed action `diagnostic_display_only` that cannot rescue another use. |
| `SCI-POINT-STAGEA-D026` | 2026-09-02 | Add `POINT-FULL-MAP-RMS-METHOD v0.1` as a third scientifically/lifecycle-distinct `unavailable_pending_separate_owner_approval` authority whose absence blocks only the canonical dynamic-range diagnostic and legacy alias. |
| `SCI-POINT-STAGEA-D027` | 2026-09-02 | Bind asymmetric product/claim dependencies, branch-independent source association, typed known/isolated/bright/approximately-centered applicability facts, and split association/fixed-response/full-response/observational-bias roles. |
| `SCI-POINT-STAGEA-D028` | 2026-09-02 | Define exact MAP/JINC/FLT signal-role names while marking every parent file as draft boundary requirements with unbound source authority/version/digest and unavailable numerical route. |
| `SCI-POINT-STAGEA-D029` | 2026-09-02 | Supersede r0.2 with the exclusive 37-object `SCI-POINT_AUTHOR_PACKET_MANIFEST v0.1/r0.3` and deterministic 39-file archive; retain earlier archives, require exact size/hash/safety/link/parity checks, and leave the exact bytes unapproved and Stage B unlaunched. |

No decision in this log freezes scientific authority, changes numerical
behavior, or establishes conformity, validation, performance, readiness, or
production state.
