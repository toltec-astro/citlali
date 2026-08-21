# SCI-RTC — Raw-Timestream Conditioning And Temporal Response

Status: Candidate v0.1/r0.10 formally reopened from the unchanged frozen r0.9
scientific authority; owner review pending. Implementation conformity not
assessed under this contract.

Version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[CAL/MAP pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md).
Work began with the package's [`PRIOR_WORK.md`](PRIOR_WORK.md) recovery record.
That record reuses the frozen implementation-independent RTC core, later owner
decisions, the phase-zero downsampling amendment, and the learned-sampling
design instead of repeating their derivations.

Implementation, audits, repairs, re-audits, tests, validation, Unity evidence,
and production status remain quarantined in
[`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md). They are not proposed author
inputs.

## Current Gate

- Package selection: approved by Grant Wilson on `2026-08-17`.
- Stage A prior-work recovery: complete and owner-reviewed.
- Sanitized Scope Brief: approved `2026-08-17`, including the owner-modified
  `flxscale` donor-scaling rule in `RTC-SCOPE-D004`.
- Exact content-bound author packet: approved `2026-08-17`.
- Implementation-blind Stage B authorship: complete from that packet only.
- Manager review: r0.1 complete; r0.2, r0.3, and r0.4 scientific-owner
  revisions implemented and closing-reviewed; r0.5 paired-coordinate revision
  complete and independently closing-reviewed; r0.6 owner-confirmed bounded
  correction complete and independently closing-reviewed; r0.7 owner-directed
  surgical correction complete and manager-reviewed; r0.8 binding owner
  Decision 9 applied; r0.9 owner Decisions 1--8 applied and freshly
  consistency-reviewed.
- Scientific-owner freeze: complete for v0.1/r0.9 on `2026-08-20`.
- Bounded conditioned-$r$ reopening: owner-authorized on `2026-08-21`; the six
  decisions are incorporated in candidate v0.1/r0.10 without changing r0.9
  conditioned-$x$ numerical behavior or any unrelated owner-ledger state.
- Implementation conformity, validation, and production promotion: not
  established.

The next gate is scientific-owner review of the exact candidate r0.10 source,
crosswalk, consistency report, and rendered PDFs. Candidate review is not an
RTC implementation-conformity audit and cannot silently resolve PTC, SCI-VAL,
or any unrelated open RTC policy.

## Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact recovery and disposition record
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined
  implementation-informed scope evidence
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): owner-approved sanitized author input
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md): approved
  binding cover for the reusable RTC core
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  sanitized conventions and package boundaries
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): approved exact
  allowed and prohibited inputs
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): 24 bounded author
  presentation and consolidation decisions
- [`MANAGER_REVIEW_R0.1.md`](MANAGER_REVIEW_R0.1.md): independence,
  structural, scientific, build, and visual-QA review
- [`SCIENTIFIC_OWNER_DIRECTIVE_R0.2.md`](SCIENTIFIC_OWNER_DIRECTIVE_R0.2.md):
  binding learn--resolve--apply revision direction
- [`CHANGE_LOG_R0.2.md`](CHANGE_LOG_R0.2.md): exact r0.1-to-r0.2 changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.2.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.2.md):
  section-to-formal-authority routing
- [`CONSISTENCY_REPORT_R0.2.md`](CONSISTENCY_REPORT_R0.2.md): required-topic
  and claim-boundary consistency review
- [`CROSS_PACKAGE_FOLLOWUP_R0.2.md`](CROSS_PACKAGE_FOLLOWUP_R0.2.md): routed
  CAL, BEAM, AST, PTC, VAL, MAP, and filtering questions
- [`SCIENTIFIC_OWNER_DIRECTIVE_R0.3.md`](SCIENTIFIC_OWNER_DIRECTIVE_R0.3.md):
  binding bounded-iteration revision direction
- [`CHANGE_LOG_R0.3.md`](CHANGE_LOG_R0.3.md): exact r0.2-to-r0.3 changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.3.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.3.md):
  bounded-iteration rationale-to-authority routing
- [`CONSISTENCY_REPORT_R0.3.md`](CONSISTENCY_REPORT_R0.3.md): iterative-plan
  invariant and claim-boundary consistency review
- [`MANAGER_REVIEW_R0.3.md`](MANAGER_REVIEW_R0.3.md): independence,
  structural, scientific, build, and visual-QA closing review
- [`SCIENTIFIC_OWNER_REVIEW_R0.4.md`](SCIENTIFIC_OWNER_REVIEW_R0.4.md):
  approved calibration-order and replacement-influence decisions plus bounded
  correction request
- [`CHANGE_LOG_R0.4.md`](CHANGE_LOG_R0.4.md): exact r0.3-to-r0.4 changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.4.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.4.md):
  bounded-correction rationale-to-authority routing
- [`CONSISTENCY_REPORT_R0.4.md`](CONSISTENCY_REPORT_R0.4.md): signal,
  attempt/plan, sampling, and influence consistency review
- [`MANAGER_REVIEW_R0.4.md`](MANAGER_REVIEW_R0.4.md): independence,
  structural, scientific, build, and visual-QA closing review
- [`SCIENTIFIC_OWNER_DIRECTIVE_R0.5.md`](SCIENTIFIC_OWNER_DIRECTIVE_R0.5.md):
  binding paired-coordinate, leakage, level-shift, and composition authority
- [`CHANGE_LOG_R0.5.md`](CHANGE_LOG_R0.5.md): exact r0.4-to-r0.5 changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.5.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.5.md):
  paired-coordinate rationale-to-authority routing
- [`CONSISTENCY_REPORT_R0.5.md`](CONSISTENCY_REPORT_R0.5.md): paired boundary,
  operator, state, and claim consistency review
- [`CROSS_PACKAGE_FOLLOWUP_R0.5.md`](CROSS_PACKAGE_FOLLOWUP_R0.5.md): routed
  Tune/readout, CAL, BEAM, AST/ALIGN, downstream, and validation follow-up
- [`MANAGER_REVIEW_R0.5.md`](MANAGER_REVIEW_R0.5.md): independence,
  structural, scientific, build, metadata, and all-page visual-QA closing review
- [`SCIENTIFIC_OWNER_REVIEW_R0.6.md`](SCIENTIFIC_OWNER_REVIEW_R0.6.md): binding
  atmospheric-template, shift-learning/replacement, and output-boundary
  decisions plus bounded scientific corrections
- [`CHANGE_LOG_R0.6.md`](CHANGE_LOG_R0.6.md): exact r0.5-to-r0.6 changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.6.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.6.md):
  bounded-correction rationale-to-authority routing
- [`CONSISTENCY_REPORT_R0.6.md`](CONSISTENCY_REPORT_R0.6.md): mapping, output,
  atmosphere, shift, and falsifier consistency review
- [`CROSS_PACKAGE_FOLLOWUP_R0.6.md`](CROSS_PACKAGE_FOLLOWUP_R0.6.md): routed
  Tune, PTC, CAL, and successor-authority follow-up
- [`MANAGER_REVIEW_R0.6.md`](MANAGER_REVIEW_R0.6.md): independence,
  structural, scientific, build, metadata, and all-page visual-QA closing review
- [`SCIENTIFIC_OWNER_REVIEW_R0.7.md`](SCIENTIFIC_OWNER_REVIEW_R0.7.md): r0.6
  acceptance, two formal blockers, five bounded clarifications, and stopping rule
- [`CHANGE_LOG_R0.7.md`](CHANGE_LOG_R0.7.md): exact r0.6-to-r0.7 surgical changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.7.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.7.md):
  seven-correction rationale-to-authority routing
- [`CONSISTENCY_REPORT_R0.7.md`](CONSISTENCY_REPORT_R0.7.md): bounded author
  self-check pending fresh consistency review
- [`MANAGER_REVIEW_R0.7.md`](MANAGER_REVIEW_R0.7.md): independence,
  seven-correction, output-split, build, metadata, and all-page QA review
- [`SCIENTIFIC_OWNER_DECISION_R0.8.md`](SCIENTIFIC_OWNER_DECISION_R0.8.md):
  binding additive-only, finite-transition, unmodeled-support, and optional
  stable-plateau correction authority
- [`CHANGE_LOG_R0.8.md`](CHANGE_LOG_R0.8.md): exact r0.7-to-r0.8 Decision 9 changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.8.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.8.md):
  Decision 9 rationale-to-authority routing
- [`CONSISTENCY_REPORT_R0.8.md`](CONSISTENCY_REPORT_R0.8.md): additive-model,
  timing/support, correction, state, build, metadata, and all-page visual check
- [`SCIENTIFIC_OWNER_DECISIONS_R0.9.md`](SCIENTIFIC_OWNER_DECISIONS_R0.9.md):
  binding application-context, lifecycle, bundle, mapping/coordinate,
  non-finite, covariance-disclosure, and despike-reporting authority
- [`CHANGE_LOG_R0.9.md`](CHANGE_LOG_R0.9.md): exact bounded r0.8-to-r0.9 changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.9.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.9.md):
  owner Decisions 1--8 rationale-to-authority routing
- [`CONSISTENCY_REPORT_R0.9.md`](CONSISTENCY_REPORT_R0.9.md): fresh
  implementation-blind decision, build, metadata, and all-page consistency review
- [`SCIENTIFIC_OWNER_FREEZE_R0.9.md`](SCIENTIFIC_OWNER_FREEZE_R0.9.md): exact
  owner freeze, claim boundary, retained open states, and change-control rule
- [`SCIENTIFIC_OWNER_REOPENING_DIRECTIVE_R0.10.md`](SCIENTIFIC_OWNER_REOPENING_DIRECTIVE_R0.10.md):
  exact bounded reopening scope and owner decisions D01--D06
- [`CHANGE_LOG_R0.10.md`](CHANGE_LOG_R0.10.md): exact bounded r0.9-to-r0.10
  candidate changes
- [`RATIONALE_TO_CONTRACT_CROSSWALK_R0.10.md`](RATIONALE_TO_CONTRACT_CROSSWALK_R0.10.md):
  six-decision rationale-to-authority routing
- [`CONSISTENCY_REPORT_R0.10.md`](CONSISTENCY_REPORT_R0.10.md): implementation-blind
  source, build, metadata, and all-page visual verification record
- [`SCIENTIFIC_OWNER_CANDIDATE_R0.10.md`](SCIENTIFIC_OWNER_CANDIDATE_R0.10.md):
  exact content-bound candidate review snapshot; not an authority freeze
- [`DECISION_LOG.md`](DECISION_LOG.md): package-selection and approved scope
  decisions
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  63 open, one conditional, 20 resolved, and five deferred decisions
- [`CROSSWALK.md`](CROSSWALK.md): exact shared-core and packet traceability
- `src/`: one six-file shared core and the two audience views
- `pdf/`: canonical candidate r0.10 science-team rationale and complete formal
  engineering view, with page counts, hashes, and all-page inspection recorded
  after the fresh build

## Protected Boundary

SCI-RTC begins with an admitted exact aligned paired raw $x/r$ detector stream
and retains the distinct upstream IQ-to-$x/r$ and ALIGN mapping identities,
time grid, independent validity, and support required by the resolved
application-context plan.
Its local operator does not reapply ALIGN. It owns the scientific use of those
coordinates, spike-aware level-shift segmentation on the original pair,
post-segmentation conditioned-$x$ donor replacement, diagnostic-only
atmospheric-template evidence, temporal conditioning, response, support,
influence, covariance, phase-zero sampling, and its consumer-neutral
conditioned-$x$/optional-same-grid-conditioned-$r$/raw-$r$ atomic bundle.
Artifact detection and affected support are pair-coherent; coordinate-specific
correction remains diagonal, and unrepairable $r$ support is honestly invalid
or unavailable without affecting valid conditioned $x$.

It does not derive ALIGN timing, AST coordinates, BEAM calibration factors,
the CAL atmosphere operator, PTC correlated-mode cleaning or weights, VAL
eligibility policy, MAP estimation, FLT map filtering, or FRUIT recurrence.
It also does not silently equate the Beammap raw `Delta f/f` signal boundary
with a calibrated `mJy/beam` path.

## Authority And Status

This package contains an owner-approved Stage A scope/packet, preserved
content-bound r0.4 inputs, an implementation-blind Stage B r0.5
owner-directed supersession, the binding r0.6 scientific-owner correction,
the binding r0.7 surgical-correction request, binding r0.8 Decision 9, and
binding r0.9 owner Decisions 1--8. Grant Wilson froze this exact v0.1/r0.9
package as scientific authority on `2026-08-20`. The freeze retains every
recorded open, conditional, resolved, and deferred owner-ledger state; it does
not invent missing decisions or make blocked claims available.

On `2026-08-21`, Grant Wilson explicitly authorized a bounded reopening of the
existing conditioned-$r$ extension point. Candidate r0.10 records six resolved
decisions covering role/optionality, pair-coherent artifacts, exact paired
grid/failure isolation, coordinate-diagonal response/covariance, optical
leakage/source protection, and downstream handoff. The frozen r0.9 baseline
remains unchanged until this candidate receives separate owner freeze action.

Current application and production behavior retain their existing repository
status until a later, separately authorized conformity and validation program
assesses them. Any substantive change beyond this bounded reopening requires
explicit owner action and another versioned successor or formally reopened
revision.
