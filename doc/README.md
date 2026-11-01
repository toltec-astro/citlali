# Citlali Documentation

Citlali documentation serves several audiences and evidence needs. Put new
material in the narrowest applicable layer and link across layers instead of
repeating it.

## Start here

The accepted development branch is `codex/refactor-mainline`. Begin with one
entry below rather than reading every dated report.

| Your question | Start with |
| --- | --- |
| What works now, and what still needs acceptance? | [Current state](CURRENT_STATE.md) |
| Which local folder should I use? | [Workspace map](WORKSPACE_MAP.md); the older folder named `citlali-refactor` is not mainline |
| How do I prepare configured POINT? | [POINT deployment and launcher](../requirements/README.md) |
| How do I build native Citlali? | [Supported Spack workflow](../spack/README.md) |
| Where does a code change belong? | [Architecture and source map](ARCHITECTURE.md) |
| Which scientific specification governs? | [Scientific-contract index](scientific_contracts/README.md), its named acceptance records, and [scientific conventions](SCIENTIFIC_CONVENTIONS.md) |
| What was integrated, with what limits? | [Integration ledger](INTEGRATION_LEDGER.md) and [accepted pointing/handoff scope](POINTING_HANDOFF_INTEGRATION_2026-10-31.md) |
| What are the older scientific files, and can they be cleaned up? | [Scientific workspace inventory and proposed batch C](SCIENTIFIC_WORKSPACE_CLEANUP_2026-11-01.md) |

The pointing/handoff cutoffs were integrated as application `f2ad2913a`; their
scope document preserves the proposal and acceptance conditions. It is no longer
the next integration request. Publication, hosted checks, scientific acceptance
and production use remain separate states in the current-state page.

The user-guide index below is an expanding documentation layer, not a
prerequisite for the existing deployment and build instructions.

## Documentation Layers

| Need | Authority | Content |
| --- | --- | --- |
| Use and interpret Citlali | [`user/`](user/README.md) | What a reduction or product does, why one would use it, relevant configuration, and interpretation limits |
| Understand a reusable scientific method | [`science/`](science/README.md) | Equations, assumptions, estimator properties, limitations, and the products or stages that use the method |
| Define intended scientific behavior | [`scientific_contracts/`](scientific_contracts/README.md) | Owner-approved package boundaries, scientific rationale, binding requirements, edge cases, and implementation-independent conformance obligations |
| Apply project-wide scientific conventions | [`SCIENTIFIC_CONVENTIONS.md`](SCIENTIFIC_CONVENTIONS.md) | Identities, units, coordinate frames, indexing, validity, and provenance semantics |
| Understand software structure | [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`adr/`](adr/README.md) | Ownership, boundaries, lifecycle, and durable design decisions |
| Verify a scientific or engineering claim | [`../validation/`](../validation/README.md) | Executable contracts, profiles, accepted evidence, and frozen campaigns |
| Reconstruct a dated investigation or transfer work | `handoff/`, dated reports, and audit packages | Scope-specific evidence and coordination history; these are not user manuals |

## Current instructions versus historical evidence

`CURRENT_STATE.md` is the short orientation page; `REFACTOR_STATUS.md` is the
dated work history and sequencing record. Read its newest applicable entry,
not an older milestone as though it describes today's baseline. The integration
ledger records owner acceptance and exact source identities.

Scientific authority comes from accepted package manifests and owner decisions.
A draft, generated PDF, copied source file or successful test is not approval by
itself. In particular, historical ALIGN/AST r0.2 output must not be substituted
for the [accepted r0.3 freeze record](scientific_contracts/audits/ALIGN_AST_HORIZONTAL_R0.3/SCI-ALIGN_AST_STAGE_B_R0.3_FREEZE_RECORD.md).

Dated reports, handoffs, failed checks and audit packages preserve what was known
at that time. Their old paths, counts and commands may require the recorded
environment or archive restoration. Keep those records intact and use current
entry points for new work. The old Conan examples in the root README are
historical reproduction material; the supported development route is Spack.

Existing dated reports and plans do not need to be moved merely to match this
organization. The proposed scientific-file cleanup removes no tracked report,
accepted specification or backup.

Every new scientific-contract package begins with the governing
[`scientific_contracts/README.md`](scientific_contracts/README.md) charter and
a prior-work recovery record. Earlier scientific reasoning is adopted, cited,
or explicitly superseded where possible; implementation audits and repair
history are preserved without entering implementation-blind derivation.

## Authoring Rule

User documentation says **what** the pipeline or product does and **why** it
matters. When the explanation requires non-obvious mathematics or statistics,
the user document links to one registered note in [`science/`](science/README.md).
Every other pipeline stage using the same technique links to that note as well.
Do not copy the derivation into each product guide, audit report, or source-file
comment.

Documentation is part of a user-visible scientific change when the change
alters configuration, output membership, product meaning, units, validity,
provenance, defaults, or interpretation. Such a change is not ready for final
acceptance until the applicable user guide, scientific-method reference,
executable contract, and validation evidence agree. A change that affects none
of those surfaces does not need documentation merely to fill a template.
