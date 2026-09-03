# SCI-POINT — Bright-Source Pointing Inference

Status: final targeted Stage B document revision `r0.3` complete; proposed
conditional scientific-owner freeze pending

Version: `v0.1`

Launch base: `0b977a90a0bae6a68dadcf7824c9b7a0c80a7f45`

Launch branch: `codex/sci-point-v0.1-stage-a`

## Program Adherence And Non-Repetition Rule

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
The owner's exact package direction is preserved in
[`SCIENTIFIC_OWNER_STAGE_A_DIRECTION_2026-09-02.md`](SCIENTIFIC_OWNER_STAGE_A_DIRECTION_2026-09-02.md).

The package began with [`PRIOR_WORK.md`](PRIOR_WORK.md) and the quarantined
[`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md). Recovered work is disposed once
in the [`WORKING_WHEEL_ADOPTION_REGISTER.md`](WORKING_WHEEL_ADOPTION_REGISTER.md).
The recovered mature fitter is treated as the working wheel. No fresh
derivation is proposed where a frozen contract or mature pointing
path already supplies the needed meaning.

## Stage A Result

Recovery supports a narrow package:

- `POINT-FIT` fits one known, isolated, bright pointing source on one exact
  observation-local array-map parent;
- `POINT-MEASUREMENT` publishes the per-array fitted source displacement and
  accompanying fitted amplitude, shape, formal uncertainty, support, method,
  parent, and diagnostic identity; and
- by owner decision `SCI-POINT-ODQ-001`, cross-array aggregation remains a
  separately owned pointing-support-producer operation rather than a
  SCI-POINT v0.1 product; and
- by owner decision `SCI-POINT-ODQ-002`, POINT stops at measured displacement;
  correction sign, telescope-offset composition, selection, publication, and
  application remain downstream responsibilities.

Owner-approved ODQ-004 adopts the existing six-parameter
elliptical-Gaussian path as
`POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`. The per-array result table,
AltAz tangent-plane convention, and explicit distinction among legacy dynamic
range, formal-fit standardization, and statistical significance are recovered
as the compatibility baseline. ODQ-005 preserves the established configurable
central search, weighted-peak initialization, global fallback, bounded fit
domain, and parameter constraints while requiring explicit requested,
effective, and realized state. These decisions do not silently freeze one set
of numerical values. ODQ-006 makes each requested array result independently
atomic. The final Stage A repair separates producer lifecycle, component
identifiability, and named-use disposition; `diagnostic_only` is use-specific,
not a producer state. ODQ-007 requires the established marginal formal errors
only under an exact separately approved method, permits joint covariance to be
honestly unavailable, and prohibits interpreting missing covariance as zero,
diagonal, or independent. ODQ-008 makes fitted amplitude,
widths, and angle required fit-result components and, together with centroid
and fit state, authorizes them as quality-control metrics for telescope
performance and observing conditions while preserving their exact processed-
map response limitations. ODQ-009 separates fit completeness, pointing-
support displacement, telescope/observing QC, and CAL/TolProj amplitude uses
under their respective policy owners; VAL only registers and evaluates.

The scientific owner confirmed that the packet cannot select or reconstruct
the legacy width convention, numerical objective/weights, complete search and
fallback procedure, marginal formal-error calculation, or exact full-map-RMS
definition. Therefore `POINT-COMPATIBILITY-METHOD v0.1`,
`POINT-FORMAL-ERROR-METHOD v0.1`, and
`POINT-FULL-MAP-RMS-METHOD v0.1` are separately
`unavailable_pending_separate_owner_approval`. The first blocks every
numerical fit and fit-derived product. Once it is approved, absence of the
second blocks formal uncertainty and dependent uses without erasing otherwise
authorized fitted values. Absence of the third blocks only the descriptive
dynamic-range diagnostic.

## Stage B r0.3 result

The exact owner-approved Stage A r0.3 packet and final targeted owner directive
now bind a shared normative common core and two rendered views:

- [`pdf/SCI-POINT-SCIENTIFIC-RATIONALE-v0.1.pdf`](pdf/SCI-POINT-SCIENTIFIC-RATIONALE-v0.1.pdf)
  explains the conditional scientific contract;
- [`pdf/SCI-POINT-ENGINEERING-CONFORMANCE-v0.1.pdf`](pdf/SCI-POINT-ENGINEERING-CONFORMANCE-v0.1.pdf)
  specifies prospective conformance evidence without independent science;
- [`STAGE_B_R0_3_RECORDS.json`](STAGE_B_R0_3_RECORDS.json) preserves exact
  SCI-VAL, response, diagnostic, dependency, lifecycle, weight, lineage, and
  profile records;
- [`STAGE_B_R0_3_TARGETED_AMENDMENTS.md`](STAGE_B_R0_3_TARGETED_AMENDMENTS.md)
  provides a non-normative navigation crosswalk for the eight targeted repairs;
- [`STAGE_B_R0_3_PARITY_REPORT.json`](STAGE_B_R0_3_PARITY_REPORT.json),
  [`STAGE_B_R0_3_SEMANTIC_CHANGE_REPORT.md`](STAGE_B_R0_3_SEMANTIC_CHANGE_REPORT.md),
  and the clean-build/PDF-QA reports close traceability; and
- [`PROPOSED_SCIENTIFIC_OWNER_FREEZE_R0_3.md`](PROPOSED_SCIENTIFIC_OWNER_FREEZE_R0_3.md)
  states the bounded conditional freeze proposed to the scientific owner.

This Stage B result adds no numerical route or implementation, validation,
accuracy, performance, readiness, production, authorization, or Unity claim.

## Explicit Boundary

SCI-POINT does not own:

- per-detector Beammap fitting, effective PSF, sensitivity, or APT production,
  which remain SCI-BEAM responsibilities;
- blank-field source detection, deblending, catalog construction, or faint
  distributed-source fitting;
- OOF optical inference or OOF observation association;
- mapmaking, filtering, FRUIT recurrence, calibration, coordinate realization,
  support-policy authorship, or empirical uncertainty inference;
- selection, interpolation, or application of pointing-correction records to
  another observation; or
- implementation audit, repair, validation, achieved performance, readiness,
  or production authorization.

Code reuse between Pointing and Beammap fitting does not merge their scientific
authority. Use of a POINT result by TolProj, TolTECA, AST, or CAL does not make
that downstream operation part of POINT.

## Parent-State Rule

Owner-approved ODQ-003 makes ordinary MAP, JINC, FLT-FIXED, and FLT-MATCHED
eligible as distinct observation-local parent families and therefore distinct
POINT method routes. FRUIT is not a separate parent type: a terminal FRUIT
result retains its exact terminal map type and carries complete FRUIT
terminal/generation lineage. Coadd parents are outside base v0.1. The current
filenames `raw` and `filtered` do not by themselves identify any of those
scientific methods. POINT may not automatically select, substitute, equate,
or fall back among routes. Eligibility does not establish numerical
availability; each exact boundary must still be present and bound as recorded
in [`PARENT_ROUTE_AND_CLAIM_MATRIX.md`](PARENT_ROUTE_AND_CLAIM_MATRIX.md).

## Stage A Packet

- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): scientist-readable candidate scope
- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact recovery and classification
- [`WORKING_WHEEL_ADOPTION_REGISTER.md`](WORKING_WHEEL_ADOPTION_REGISTER.md):
  adopt/abstract/defer/supersede/exclude decisions
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined implementation and
  evidence inventory
- [`OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md`](OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md):
  package and operational ownership
- [`OPERATOR_AND_PRODUCT_TAXONOMY.md`](OPERATOR_AND_PRODUCT_TAXONOMY.md):
  proposed role and product identities
- [`PARENT_ROUTE_AND_CLAIM_MATRIX.md`](PARENT_ROUTE_AND_CLAIM_MATRIX.md):
  non-equivalent candidate parents and allowed claims
- [`CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md`](CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md):
  genuine gaps and conflicts
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  sanitized predecessor extract admitted to the author packet
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): exclusive,
  SHA-bound 37-object `r0.3` author-packet review candidate
- [`AUTHOR_PACKET_MANIFEST.sha256`](AUTHOR_PACKET_MANIFEST.sha256): manifest
  digest sidecar
- [`author_packet/README.md`](author_packet/README.md): deterministic archive,
  rebuild, and verification instructions
- [`author_packet/SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz`](author_packet/SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz):
  portable copy of the exact proposed Stage B author inputs
- [`SCIENTIFIC_OWNER_METHOD_AUTHORITY_RESPONSE_2026-09-02.md`](SCIENTIFIC_OWNER_METHOD_AUTHORITY_RESPONSE_2026-09-02.md):
  binding owner disposition for the compatibility and formal-error method
  authorities
- [`POINT_COMPATIBILITY_METHOD_RECOVERY_BRIEF.md`](POINT_COMPATIBILITY_METHOD_RECOVERY_BRIEF.md):
  quarantined numerical-method recovery assignment; not launched
- [`POINT_FORMAL_ERROR_METHOD_RECOVERY_BRIEF.md`](POINT_FORMAL_ERROR_METHOD_RECOVERY_BRIEF.md):
  quarantined formal-error recovery assignment; not launched
- [`POINT_FULL_MAP_RMS_METHOD_RECOVERY_BRIEF.md`](POINT_FULL_MAP_RMS_METHOD_RECOVERY_BRIEF.md):
  quarantined full-map-RMS recovery assignment; not launched
- [`SCIENTIFIC_OWNER_R0_3_CLOSURE_DIRECTIVE_2026-09-02.md`](SCIENTIFIC_OWNER_R0_3_CLOSURE_DIRECTIVE_2026-09-02.md):
  binding final policy-type, method-gate, and boundary-closure disposition
- [`PROPOSED_SANITIZED_AUTHOR_INPUTS.md`](PROPOSED_SANITIZED_AUTHOR_INPUTS.md):
  realized Stage B firewall and dispatch-state record
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  bounded owner questions
- [`SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-09-02.md):
  per-array terminal-product and downstream-aggregation decision
- [`SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-09-02.md):
  measurement-versus-correction-construction decision
- [`SCIENTIFIC_OWNER_ODQ_003A_003B_DIRECTION_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_003A_003B_DIRECTION_2026-09-02.md):
  terminal-FRUIT lineage and coadd-exclusion subdecisions
- [`SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-09-02.md):
  eligible observation-local parent-family decision
- [`SCIENTIFIC_OWNER_ODQ_004_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_004_APPROVAL_2026-09-02.md):
  compatibility-estimator decision
- [`SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-09-02.md):
  search, support, and constraint-state decision
- [`SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-09-02.md):
  per-array atomicity and partial-success decision
- [`SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-09-02.md):
  formal-uncertainty and joint-covariance decision
- [`SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-09-02.md):
  amplitude, effective-shape, and quality-control role decision
- [`SCIENTIFIC_OWNER_ODQ_009_APPROVAL_2026-09-02.md`](SCIENTIFIC_OWNER_ODQ_009_APPROVAL_2026-09-02.md):
  named-use policy ownership and author-assignment decision
- [`SOURCE_IDENTITY_MANIFEST.md`](SOURCE_IDENTITY_MANIFEST.md): exact recovery
  source identities
- [`DECISION_LOG.md`](DECISION_LOG.md): Stage A process decisions
- [`CROSSWALK.md`](CROSSWALK.md): Stage A deliverable crosswalk
- [`STAGE_A_CHANGE_LOG.md`](STAGE_A_CHANGE_LOG.md): package creation record

The canonical `src/` and `pdf/` paths now contain document revision r0.3 and
its two verified rendered PDFs. Stage A materials remain immutable scientific
authority and historical recovery evidence; the shared Stage B common core is
the sole normative scientific source for the two views.

## Current Gate

Recovery, the ODQ-001 through ODQ-009 decisions, exact Stage A r0.3 author
packet approval, implementation-blind Stage B authorship, final targeted r0.3
repair, clean rebuild, parity verification, and all-page PDF QA are complete.
The current decision is the proposed conditional r0.3 scientific-owner freeze.
The three numerical method authorities, exact route instances, four named-use
profile registrations, and response/bias authorities remain separately gated;
none is approved by the document freeze.

No algorithm, frozen authority, implementation, configuration, validation,
performance, readiness, production, or Unity state is changed or claimed.
