# SCI-VAL — Sample And Detector Validity, Flags, And Map Eligibility

Producer facts and use-specific eligibility; not final map validity.

Status: owner-approved targeted Stage B r0.3 revision implemented and
manager-reviewed; scientific-owner review pending; scientific authority not
frozen

Proposed scientific contract version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package is governed by the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[accepted pilot process](../../../PILOT_PROCESS_REVIEW_2026-08-16.md). Work
began with [`PRIOR_WORK.md`](PRIOR_WORK.md), not a new derivation.

Recovery found no frozen or approved implementation-independent SCI-VAL core.
It did find approved cross-package validity decisions, eleven historical
incoming handoffs, current product and convention authorities, and extensive
implementation/audit material. The approved scientific distinctions are
abstracted into [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md).
Implementation details, findings, repairs, tests, validation, Unity evidence,
and production status remain quarantined in
[`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md) and are excluded from scientific
authorship.

The genuinely new work is to define a reusable, cause-preserving interchange
and deterministic evaluation contract without taking ownership of producer
facts/local composites or scientific-use policies from RTC, CAL, PTC, MAP,
and later consumers. The first scope review is recorded in
[`SCOPE_REVIEW_R0.1.md`](SCOPE_REVIEW_R0.1.md); its seven bounded corrections
are incorporated. The subsequent ownership correction is approved in
[`SCOPE_OWNERSHIP_DECISION_R0.2.md`](SCOPE_OWNERSHIP_DECISION_R0.2.md).

## Proposed Boundary

SCI-VAL begins with typed facts supplied by upstream producers for exact
sample and detector occurrences: identity, origin, direct validity, numerical
state, availability, causes, operator support, transitive influence, response
status, and producer-local decisions. It accepts an exact owner-approved,
versioned policy supplied by the named-use owner and evaluates it without
inventing its predicates.

It ends with a cause-preserving evaluated disposition for that exact use,
including independent request, applicability, eligibility, and realization
axes; the input facts considered; policy/use identity; decision stage;
exact/conservative influence; aggregation if any; lifecycle; and reasons. It
does not modify signal, replace data, estimate a model, choose a map support
threshold, or infer final product validity.

The central statement is:

> VAL does not decide whether a number is good and does not invent a
> scientific-use policy. It determines reproducibly whether exact
> producer-owned facts/supports satisfy one exact use-owner-supplied policy,
> without erasing causes or transferring scientific ownership.

Producer and consumer ownership remains explicit:

- RTC owns conditioning, replacement, temporal support, typed causes, and
  causal influence.
- CAL owns calibration-domain, detector-binding, atmosphere, factor, and
  response validity.
- PTC owns fit/application/output/coefficient supports and its staged
  decisions.
- Producers own their causes and Boolean composition into producer-local
  supports.
- Each named-use owner owns its admission policy.
- VAL owns shared types, knowledge-state logic, immutable provenance, cause
  preservation, and deterministic supplied-policy evaluation mechanics.
- MAP and later operators own their estimator support and output validity.

## Package Contents

- [`PRIOR_WORK.md`](PRIOR_WORK.md): exact recovery and disposition record
- [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md): quarantined implementation and
  historical evidence
- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md): sanitized owner-approved scientific
  boundary
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md):
  approved author-facing stable inputs
- [`AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`](AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md):
  sanitized exact RTC/CAL/PTC/MAP meanings approved for the author packet
- [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md): approved packet and
  firewall; exact content hashes
- [`DECISION_LOG.md`](DECISION_LOG.md): owner-approved scope decisions and
  inherited approved facts
- [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md):
  owner questions that materially affect Stage B
- [`CROSS_PACKAGE_FOLLOWUP.md`](CROSS_PACKAGE_FOLLOWUP.md): adjacent authority
  routing without amendment
- [`SCOPE_REVIEW_R0.1.md`](SCOPE_REVIEW_R0.1.md): exact first-review hash and
  disposition of every bounded correction
- [`SCOPE_OWNERSHIP_DECISION_R0.2.md`](SCOPE_OWNERSHIP_DECISION_R0.2.md):
  owner-approved producer/use-owner/VAL authority split and no-rescue rule
- [`REVISION_DIRECTIVE_R0.2.md`](REVISION_DIRECTIVE_R0.2.md): exact targeted
  revision authority and decisions `VAL-R02-D001--D009`
- [`REVISION_DIRECTIVE_R0.3.md`](REVISION_DIRECTIVE_R0.3.md): exact surgical
  revision authority, bound feedback digest, and decisions
  `VAL-R03-D001--D010`
- [`PROFILE_REGISTRY.md`](PROFILE_REGISTRY.md): immutable owner/source-bound
  profile records and reserved package-qualified profile names
- [`SOURCE_BINDING_REGISTER.md`](SOURCE_BINDING_REGISTER.md): exact admitted
  adjacent scientific source/version bindings and change consequences
- [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md): eight first-pass
  derivation decisions, r0.2/r0.3 dispositions, and layered resolution of the
  six bounded questions
- [`CROSSWALK.md`](CROSSWALK.md): exact requirement and prediction traceability
- [`MANAGER_REVIEW_R0.1.md`](MANAGER_REVIEW_R0.1.md): content and artifact QA
  record for the historical first pass
- [`MANAGER_REVIEW_R0.2.md`](MANAGER_REVIEW_R0.2.md): targeted revision content
  and artifact QA record
- [`MANAGER_REVIEW_R0.3.md`](MANAGER_REVIEW_R0.3.md): surgical revision
  content, consistency, and artifact QA record
- [`src/`](src/): six shared canonical modules and the two view wrappers
- [`pdf/`](pdf/): the 8-page scientist-facing rationale and 20-page formal
  engineering conformance view

## Stage B Surgical r0.3 Result

The original approved four-file packet and the owner-approved r0.2/r0.3
directives are content-bound and verified. The active canonical profile is
`SCI-VAL:independent_exposure@1`, with no compatibility alias from the former
draft key; the namespace does not confer policy ownership. Every aggregate is
now a distinct registered proposition binding its exact homogeneous atomic
source profile. Structural and non-gating conflicts, exception conflicts, and
the four owner-supplied response/uncertainty roles have deterministic
semantics. Exact adjacent source/version bindings remain in the continuing
source register without inventing unavailable PTC or MAP policy.

The scientist-facing rationale is a standalone 8-page narrative with the
worked occurrence table on physical page 2. The 20-page engineering view
imports the six canonical formal modules exactly once and retains the complete
tuples, knowledge logic, truth tables, equations, requirements, predictions,
and replay/conformance obligations. All existing IDs
`SCI-VAL-REQ-001--049` and `SCI-VAL-PRED-001--024` are preserved; r0.3 appends
no normative ID. The totals remain `49` requirements, `24` predictions, and
an exact `73`-row crosswalk.

The durable verifier passes original packet and revision-directive hashes,
registry/source bindings, sequential ID coverage, exact crosswalk coverage,
the dual-view genre split, complete PDF text coverage, successful builds, and
stable PDF checks. All `28` rendered PDF pages were inspected with Poppler.
The indivisible normative-item blocks remain effective; the rebuilt pages
have no clipping, overlap, broken table, bad glyph, or unreadable content.

No general SCI-VAL scientific question remains open from
`SCI-VAL-OWNER-QB001--QB006`. QB001 and QB003 retain exact serialization work
as engineering-deferred; QB006 retains sufficient-summary and associative
combine details as profile-local. Those dispositions do not select an
implementation representation or package-owned profile.

No implementation conformity, validation, scientific freeze, production
readiness, or adjacent-package readiness claim is made.
