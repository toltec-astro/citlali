# Citlali Scientific Contract Library Program

Status: governing program charter

Scientific owner: Grant Wilson

Program status phrase for a frozen package:

> **Scientific authority frozen; implementation conformity not yet assessed
> under this contract.**

## Purpose

This program creates durable scientific ground truth for the scientifically
meaningful processes in Citlali. Each package states the intended quantity or
transformation, legitimate inputs, meaningful outputs, mathematical reasoning,
assumptions, conventions, validity behavior, uncertainty, limiting cases, and
scientifically predictive tests.

The program is presently a contract-development program. It does not audit the
application against a new contract, repair code, run closure validation, or
claim implementation conformity.

Every package must begin by linking to this charter and completing the
[prior-work recovery procedure](#prior-work-recovery). This is the principal
anti-drift rule for the program.

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

## Required Package Opening

The first substantive section of every package `README.md` and Scope Brief is
**Program adherence and prior-work recovery**. It must:

1. link to this charter;
2. identify the package's completed `PRIOR_WORK.md` record;
3. state which earlier materials are adopted, cited, abstracted, superseded,
   or excluded;
4. state what genuinely new scientific work remains;
5. identify exactly which sanitized references may enter the author packet;
   and
6. confirm that implementation-derived material remains outside the
   implementation-blind author channel.

A package may not commission a new scientific-contract author until that
opening has been reviewed as part of the Scope Brief. The manager must return a
package that omits or weakens the opening instead of allowing work to proceed.

## Prior-Work Recovery

The purpose of recovery is to preserve earlier scientific thought without
allowing historical implementation behavior to become scientific authority.
It is not an obligation to repeat every old audit.

### Discovery

Before drafting a Scope Brief, the scope investigator searches the repository,
named topic branches, durable handoffs, earlier contracts, method notes,
external instrument references, owner decisions, and relevant adjacent
repositories. The investigator records exact paths or references in the
package's `PRIOR_WORK.md`.

[`PRIOR_WORK_REGISTRY.md`](PRIOR_WORK_REGISTRY.md) provides the initial
cross-package discovery map. It must be extended and re-verified rather than
treated as exhaustive or current by default.

Discovery is implementation-informed Stage A work. Its complete record is not
sent to the scientific-contract author.

### Source classification

Every recovered item receives one of these classifications:

| Class | Meaning | Permitted use |
| --- | --- | --- |
| Governing scientific authority | Current owner-approved convention, contract, or durable decision | Binding input unless the owner explicitly reopens it |
| Approved scientific decision | Scoped owner answer with recoverable provenance | Binding within its recorded scope |
| Reusable scientific reference | Independent derivation, instrument reference, or method note with no unresolved implementation dependence | May enter the author packet after owner approval |
| Implementation-informed scope evidence | Code, product descriptions, interfaces, and architecture showing what problem is currently assigned | May shape the Scope Brief, but may not prescribe the scientific answer |
| Historical audit, repair, or validation evidence | Findings, source traces, candidates, tests, reductions, and conformity claims | Preserved for later audit; excluded from independent derivation |
| Superseded or conflicting material | Earlier authority or reasoning that disagrees with a later source | Retained with the conflict and required review recorded |

### Disposition instead of repetition

For each recovered item, `PRIOR_WORK.md` records one disposition:

- **adopt**: incorporate an already approved scientific statement without
  re-deriving it;
- **cite**: use the source as an approved reference while retaining its own
  authority;
- **abstract**: extract a sanitized scientific question, definition, or
  independently derived result from a mixed implementation/audit document;
- **supersede**: replace it explicitly and record why;
- **defer**: preserve it for later implementation audit or validation; or
- **exclude**: keep it outside the package because it is irrelevant,
  implementation-prescriptive, unreliable, or outside scope.

Fresh derivation is required only for a genuine gap, an unresolved conflict,
an unapproved scientific choice, or reasoning whose independence cannot be
established. Rewording an existing approved derivation is not progress.

Mixed documents require care. For example, the historical Convolve contract
contains valuable independent mathematics alongside implementation audit and
repair material. Stage A may recover and sanitize the former, with assumptions
and provenance intact; the mixed document itself does not enter the author
packet unless Grant explicitly approves it as a scientific reference.

### Recovery deliverable

The investigator produces a short recovery synthesis containing:

- the scientific questions already answered and their authority;
- reusable equations, definitions, conventions, and decisions;
- unresolved or conflicting questions;
- material excluded from the author packet and why;
- the exact new work the package must perform; and
- a proposed allowed-reference list for owner approval.

The synthesis should cite existing material rather than copy it. Shared
methods belong in [`../science/`](../science/README.md) and are referenced by
stable method ID when applicable.

## Information Firewall

### Stage A: implementation-informed scope investigation

The investigator may inspect implementation, repository documentation,
products, architecture, prior audits and repairs, historical discussions, and
approved external references. The output is:

1. an internal dossier, which may contain implementation details; and
2. a separate author-facing Scope Brief sanitized of implementation-derived
   scientific answers.

The internal dossier and historical conformity evidence never enter the
scientific-contract author packet.

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

## Canonical Package Contents

```text
packages/<PACKAGE-ID>/v<MAJOR>.<MINOR>/
  README.md
  PRIOR_WORK.md
  SCOPE_BRIEF.md
  DECISION_LOG.md
  CROSSWALK.md
  src/
    common/
      notation.tex
      definitions.tex
      equations.tex
      assumptions.tex
      requirements.tex
      edge_cases.tex
    scientific-rationale.tex
    engineering-conformance.tex
  pdf/
    <PACKAGE-ID>-SCIENTIFIC-RATIONALE-v<MAJOR>.<MINOR>.pdf
    <PACKAGE-ID>-ENGINEERING-CONFORMANCE-v<MAJOR>.<MINOR>.pdf
```

Use only the common modules the package needs. The layout is a convention, not
a requirement to create empty files.

## Requirement Identity And Crosswalk

Binding requirements use stable identifiers such as `SCI-CAL-REQ-001`. Every
engineering requirement points to a scientist-facing section, equation,
assumption, or statement. `CROSSWALK.md` contains only:

| Requirement | Scientist-facing source | Engineering interpretation |
| --- | --- | --- |

An untraceable engineering requirement must be scientifically justified in
both views or removed.

## Package Lifecycle

1. Recover and classify prior work.
2. Investigate the implementation-informed scientific scope.
3. Produce and obtain owner approval of the sanitized Scope Brief.
4. Dispatch a fresh implementation-blind scientific author.
5. Iterate on scientific substance and owner decisions.
6. Commission an implementation-blind consistency review.
7. Freeze the approved brief, canonical sources, two PDFs, crosswalk, decision
   log, version/date, and exact file identifiers together.

Freezing a contract does not validate Citlali. Implementation audit,
repair, re-audit, validation, and production authorization are later programs
requiring separate authorization.

## Program Index

Current package status and the next owner-facing action live in
[`INDEX.md`](INDEX.md). Internal worker lifecycle and repository mechanics do
not belong in that index.
