# SCI-VAL v0.1 — Adjacent Scientific Source-Binding Register

Status: continuing r0.3 source-binding authority; availability limits
preserved

Date: `2026-08-21`

## Purpose

This register binds every adjacent meaning used by SCI-VAL to the exact
approved package or sanitized boundary authority available to this revision.
It does not import a full adjacent contract, upgrade an open package to
frozen authority, or create a missing policy.

The source tables rendered in the r0.3 rationale and engineering companion
are snapshots of this register. This file remains the continuing authority;
an adjacent source update is recorded here and in affected immutable profile
bindings without requiring a rewrite of SCI-VAL Core narrative.

| Producer or use owner | Exact source/version binding | Imported meaning | Compatibility and change consequence |
| --- | --- | --- | --- |
| SCI-RTC | SCI-RTC v0.1/r0.9 frozen authority as recorded in `CROSS_PACKAGE_FOLLOWUP.md`; sanitized clauses in `AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`, section “RTC Facts Supplied To VAL” | Representative source, original/synthesized/replaced origin, typed causes, operator controls, support, influence precision, response/uncertainty availability, immutable lifecycle | Any change to representative-source or origin semantics requires a new source binding and compatibility review of `SCI-VAL:independent_exposure@1`; VAL cannot reinterpret the producer fact |
| SCI-CAL | SCI-CAL v0.1/r0.3 architecture-frozen rationale, scientific authority not frozen, as recorded in the program `INDEX.md`; sanitized clauses in `AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`, section “CAL Facts Supplied To VAL” | Calibration factor/domain, detector binding, atmosphere role, response, uncertainty, and availability meanings only | Conditional input. A changed or unresolved CAL meaning leaves dependent profiles unavailable; no numerical CAL policy or identity response is inferred |
| SCI-PTC | SCI-PTC v0.1/r0.4 frozen scientific authority as recorded in the program `INDEX.md`; imported semantics restricted to `DECISION_LOG.md` inherited fact and `AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`, section “PTC Facts Supplied To VAL” | Distinct basis/loading fit, application, output, coefficient/QC, response, empirical/simulation, and staged support roles; only fit-support change requires refit/invalidation | Package-qualified profiles remain unbound until exact PTC-owned policy records are registered. Any source revision requires a new binding and cannot rewrite earlier VAL decisions |
| SCI-MAP | SCI-MAP v0.1/r0.3 house rationale, scientific authority not frozen, as recorded in the program `INDEX.md`; accepted F010/ADR boundary summarized in `DECISION_LOG.md`; sanitized clauses in `AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`, section “MAP Facts And Authority Retained By MAP” | Upstream admission is distinct from projection, numerical contribution, exposure, normalization support, science-policy support, response/covariance, coaddition, and final raw validity | Boundary meaning may be used; an exact MAP-owned `map_upstream_admission` policy is not available in this package. Requested evaluation remains unavailable until MAP supplies a complete versioned profile. A later MAP boundary or profile requires a new binding |
| ALIGN/AST/TEL input | Owner-approved sanitized boundary in `SCOPE_BRIEF.md` and `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`; no standalone frozen package/version supplied | Time, coordinate, association, origin, synthesis/recomputation, detector binding, frame, and producer-local validity fact roles | Meaning-only dependency. No profile may claim exact source compatibility until a versioned owning authority is supplied |

## Binding Rule

An exact source binding is part of policy/profile identity and replay. If the
authoritative source changes, VAL must either resolve a newly registered
compatible binding or return the owner-declared unavailable result. It must
not silently substitute “current” adjacent meaning. In particular, this
register contains no MAP predicate, threshold, or exception and does not make
`SCI-MAP:map_upstream_admission` evaluable.
