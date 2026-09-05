# Program charter and approved scope

## Source cover

These are process obligations, not scientific weighting rules. The charter
is the Citlali Scientific Contract Library Program at the exact source G03.
Only the excerpts below are admitted. References to other repository material
remain locators and confer no access.

### Roadmap binding

The applicable approved program roadmap is
`doc/scientific_contracts/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md`
at `00b974c9039d4c3025dcce18f26bca69d36af9c3`, SHA-256 `01fba487ef6cc023712c2b005d1ba0e1b19005f07a43e1d0b0a4af32423c74b2`.
Its full content is excluded from this author channel. The bounded scientific
sequence supplied by the owner's subsequent first-scope approval is: prepare
the common PTC registry-entry contract and a first uniform/constant family for
MAP/JINC consideration; defer sensitivity and residual-scatter definitions;
select no mode default. The manager approval record has SHA-256
`364880bec1e1e889075e7c0b1acc4b23968d56c493c1ccb52f4e25d71b2434ea`. That scope approval does not release this packet or adopt a
numerical definition. The local [Scope Brief](../SCOPE_BRIEF.md) is the complete
candidate scientific task; scientific substance remains for owner review.

## Governing principles

Source: G03; whole-source SHA-256 `351e9b7775b0bf78cba01bf4cd2fafd9591c4b43931b0dc23d82d97f0dfe82d2`.

<!-- EXCERPT PROCESS_AND_SCOPE-Governing principles -->
## Governing Principles

1. Citlali is a scientific instrument implemented in software. Scientific
   meaning is the authority being constructed; repository provenance is
   supporting bookkeeping.
2. Current implementation can identify the problem a package appears to own,
   but it cannot establish the scientifically correct estimator, weighting,
   normalization, threshold, fit, flag, or numerical method.
3. Implementation-informed scope investigation and implementation-blind
   scientific derivation are separate intellectual stages.
4. One versioned scientific authority is rendered as two conformant views:
   the scientist-facing *Scientific Rationale and Contract* and the
   engineering-facing *Engineering Conformance Specification*.
5. Shared notation, definitions, equations, assumptions, requirements, and
   edge-case classifications have one canonical LaTeX source.
6. An engineering requirement must trace to the scientist-facing authority.
   The engineering view may not introduce new science.
7. Owner approval is required for the scientific boundary and scientific
   substance. A substantive change after approval creates a versioned
   successor; frozen authority is never silently edited.
8. Documentation should remain proportional to the science. This program does
   not create a lifecycle database, generalized audit engine, or elaborate
   document generator.
9. Once a scientific-owner decision is incorporated into a scientific
   contract, subsequent audit and implementation work tests conformance to
   that decision rather than reopening it because a stronger architecture is
   possible. Reopening requires an actual contradiction, inability to perform
   the intended science, or new scientific evidence that the approved
   decision is wrong. An unimplemented stronger capability, an intentionally
   unsupported readiness tier, or an implementation discrepancy is not by
   itself grounds to reopen scientific scope.

<!-- END EXCERPT PROCESS_AND_SCOPE-Governing principles -->

## Required opening

Source: G03; whole-source SHA-256 `351e9b7775b0bf78cba01bf4cd2fafd9591c4b43931b0dc23d82d97f0dfe82d2`.

<!-- EXCERPT PROCESS_AND_SCOPE-Required opening -->
## Required Package Opening

The first substantive section of every package `README.md` and Scope Brief is
**Program adherence and prior-work recovery**. It must:

1. link to this charter;
2. link to the applicable approved program roadmap and predecessor authority;
3. identify the package's completed `PRIOR_WORK.md` record;
4. state which earlier materials are adopted, cited, abstracted, superseded,
   or excluded;
5. state what genuinely new scientific work remains;
6. identify exactly which sanitized references may enter the author packet;
   and
7. confirm that implementation-derived material remains outside the
   implementation-blind author channel.

A package may not commission a new scientific-contract author until that
opening has been reviewed as part of the Scope Brief. The manager must return a
package that omits or weakens the opening instead of allowing work to proceed.

<!-- END EXCERPT PROCESS_AND_SCOPE-Required opening -->

## Owner scope and author firewall

Source: G03; whole-source SHA-256 `351e9b7775b0bf78cba01bf4cd2fafd9591c4b43931b0dc23d82d97f0dfe82d2`.

<!-- EXCERPT PROCESS_AND_SCOPE-Owner scope and author firewall -->
### Owner scope approval

Grant reviews the sanitized Scope Brief, its prior-work recovery synthesis,
proposed boundary, allowed references, and unresolved decisions. Approval
freezes the brief as the author task's input. Substantive scope changes require
renewed approval.

### Stage B: implementation-blind scientific derivation

Each contract author is a fresh GPT-5.6 Ultra task with no inherited Citlali
implementation context. It works from an isolated author packet containing
only:

- the owner-approved Scope Brief;
- owner-approved scientific or instrument references; and
- subsequent owner answers to scientific questions.

The author must not inspect Citlali implementation, tests, audits, repairs,
re-audits, source-specific explanations, or validation findings. Prior work
may prevent repetition only when Stage A classifies it as reusable scientific
material and Grant approves it for the author packet.

### Contract review and freeze

The manager reviews scope compliance, firewall integrity, shared-core use, and
crosswalk completeness. Grant reviews scientific substance. A fresh,
implementation-blind consistency reviewer then checks that the two rendered
views agree. Any material discrepancy blocks freezing.

<!-- END EXCERPT PROCESS_AND_SCOPE-Owner scope and author firewall -->

## Rationale and version discipline

Source: G03; whole-source SHA-256 `351e9b7775b0bf78cba01bf4cd2fafd9591c4b43931b0dc23d82d97f0dfe82d2`.

<!-- EXCERPT PROCESS_AND_SCOPE-Rationale and version discipline -->
## Science-Team Rationale House Standard

Every new package begins by referring to this section and the prior-work
recovery above. The package must preserve three mutually consistent artifacts:

1. the normative engineering conformance contract;
2. the science-team rationale; and
3. the crosswalk plus scientific-owner decision ledger.

The science rationale opens with a compact input/output/equation/source/status
block. Its main narrative should ordinarily fit in eight to twelve pages,
explain the physical model before formal machinery, and omit hashes, internal
identifiers, and code-level detail. Formal states, exact requirements,
identity mechanics, and audit-oriented detail belong in appendices or the
engineering contract.

Version axes are never conflated:

- v identifies the scientific contract or package version;
- r identifies the revision of one document representing that contract.

Titles, running headers, metadata, filenames, package indexes, change logs,
and crosswalks must carry both axes when ambiguity is possible. A prose edit
advances r, not v.

Every scientific quantity follows the producer--transformer--consumer rule:
the producer owns its meaning; a transformer owns only an explicitly approved
transformation and its lineage; a consumer applies the selected value without
reinterpretation. Package boundaries and figures must show those roles
directly.

Every missing fact is classified as exactly one of:

- approved;
- known but not supplied;
- unresolved decision;
- implementation unassessed; or
- validation not performed.

The rationale must not infer across those classes. It includes a validity
table, uncertainty budget, an explanation of what each validation layer would
establish, the owner decisions, and the crosswalk. Claims remain separated
into engineering conformance, representation fidelity, observational
performance, and production readiness.

The companion owner-decision ledger records, for every question: stable ID,
owning scientific authority or package, state (open, decided, deferred, or
superseded), evidence or decision required, exact blocked claim or output,
resolution authority, resolution date, and affected documents. An unnamed
owner is recorded as unnamed; authority is never manufactured by inference.
The PDF decision register is generated from or explicitly checked against
this ledger.

Rationale drafting is limited to three substantive review rounds unless the
scientific owner explicitly reopens the process. Freeze is based on scientific
accuracy, explicit uncertainty, and traceability rather than artificial
completeness. After the final cleanup there is no further stylistic round.
A frozen rationale is revised only when an open decision is formally resolved,
the normative engineering contract changes, validation evidence changes an
evidentiary status, or a genuine scientific inconsistency is found.

<!-- END EXCERPT PROCESS_AND_SCOPE-Rationale and version discipline -->
