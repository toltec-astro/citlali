# Scientific Contract Library Pilot Process Review

Status: accepted program workflow; applied to SCI-BEAM Stage A

Review date: `2026-08-16`

Pilot packages reviewed: SCI-CAL v0.1 and SCI-MAP v0.1

## Outcome

The two pilots established a durable package workflow. New packages shall use
that workflow and shall not repeat already documented derivation, audit, or
validation work merely to recreate context. SCI-BEAM may begin Stage A under
this review, but implementation-blind scientific authorship remains blocked
until the scientific owner approves its package-specific Scope Brief and exact
author packet.

This review authorizes process and sequencing. It does not approve any BEAM
scientific model, threshold, product meaning, or production use.

## Lessons Retained From The Pilots

1. **Begin from the program and recovery record.** Every package README opens
   by linking to the library program and its `PRIOR_WORK.md`. The recovery
   registry seeds the search but never replaces package-specific verification.
2. **Reuse before deriving.** Existing independent mathematics, owner
   decisions, and stable conventions are adopted or content-bound once.
   Duplicative restatement is not a second authority. Fresh derivation is
   limited to a genuine gap, unresolved conflict, or owner-requested change.
3. **Separate Stage A from Stage B.** An implementation-informed internal
   dossier establishes scope and quarantine. A sanitized Scope Brief and
   approved reference packet are the only inputs to implementation-blind
   scientific authorship.
4. **Preserve one normative core.** Notation, definitions, equations,
   assumptions, requirements, and edge cases live in shared modules. The
   scientist-facing rationale and engineering-facing conformance view explain
   or organize that same authority; neither may create independent science.
5. **Separate audiences without splitting authority.** The rationale leads
   with the physical model and ordinarily keeps its main narrative to eight to
   twelve pages. Exact states, identities, failure predicates, and audit detail
   belong in appendices or the engineering view.
6. **Keep version axes distinct.** Contract version (`v`) and document
   revision (`r`) are different. Stable filenames point to the active views;
   revision metadata and changelogs preserve drafting history.
7. **Make owner choices explicit.** Scientific-owner decisions live in a
   decision ledger. Open decisions remain visible and limit claims; prose must
   not silently resolve them. A polished house version is not a frozen
   scientific authority while owner decisions remain open.
8. **Use exact author packets.** Every admitted author reference is named and
   content-bound. Implementation, audits, repair findings, tests, validation,
   and current production state remain excluded unless the owner explicitly
   approves a sanitized scientific extract.
9. **Check formal and visual integrity.** Mechanical coverage checks,
   requirement/prediction traceability, clean LaTeX compilation, Poppler
   rendering, and page-by-page visual inspection are required before a rendered
   package is presented as complete.
10. **Stop stylistic churn.** After the owner accepts the house form, revise
    only for an owner decision, normative change, new scientific evidence, or
    a demonstrated inconsistency between the shared authority and a rendered
    view.

## Anti-Repetition Procedure

At the beginning of each package:

1. resolve the exact references in `PRIOR_WORK_REGISTRY.md`;
2. search living status, relevant topic refs, later owner decisions, sibling
   repositories, and named external scientific references;
3. classify each item as governing authority, reusable independent science,
   approved decision, implementation-informed scope evidence, historical
   audit/validation evidence, conflicting/superseded material, or irrelevant;
4. record exact identities, conflicts, limitations, and disposition in the
   package `PRIOR_WORK.md`;
5. state which questions are already answered and which work is genuinely new;
6. quarantine source-, audit-, repair-, validation-, and production-specific
   details in `INTERNAL_DOSSIER.md`; and
7. give an author only the owner-approved, content-bound packet named by the
   sanitized Scope Brief.

When a later source restates an earlier derivation, the package cites the
earliest adequate authority plus any binding supersession decision rather than
asking an author to derive the result again. When sources conflict, the
conflict is an owner question; convenience does not choose the winner.

## Pilot-Three Boundary Decision

SCI-BEAM is the third pilot package. Its Stage A work is separate from active
ALIGN/AST work:

- BEAM may accept a declared coordinate relation, frame, and validity state as
  conditional upstream inputs;
- BEAM may infer observation-local source centroids in that declared frame;
- BEAM does not define physical event timing, absolute pointing, astrometric
  correction, or detector-coordinate truth; and
- active ALIGN material is not imported. Only frozen records may be recovered,
  and any resulting dependency remains explicit and fail-closed.

This boundary permits recovery and scope drafting. It does not close BEAM's
ALIGN/AST dependency or authorize Stage B.

## Gate Into Stage B

SCI-BEAM Stage B may begin only after the scientific owner:

1. approves the scientific boundary and exclusions in `SCOPE_BRIEF.md`;
2. dispositions every scope-level owner choice that would materially change
   the author's task;
3. approves the exact allowed reference packet and its hashes; and
4. confirms that the author will not receive the internal dossier, source,
   audit findings, repairs, tests, validation evidence, or production status.

The manager then commissions a fresh implementation-blind scientific author.
Agreement to this program review is not advance approval of a later Scope
Brief.
