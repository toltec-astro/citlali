# Citlali Documentation

Citlali documentation serves several audiences and evidence needs. Put new
material in the narrowest applicable layer and link across layers instead of
repeating it.

## Documentation Layers

| Need | Authority | Content |
| --- | --- | --- |
| Use and interpret Citlali | [`user/`](user/README.md) | What a reduction or product does, why one would use it, relevant configuration, and interpretation limits |
| Understand a reusable scientific method | [`science/`](science/README.md) | Equations, assumptions, estimator properties, limitations, and the products or stages that use the method |
| Apply project-wide scientific conventions | [`SCIENTIFIC_CONVENTIONS.md`](SCIENTIFIC_CONVENTIONS.md) | Identities, units, coordinate frames, indexing, validity, and provenance semantics |
| Understand software structure | [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`adr/`](adr/README.md) | Ownership, boundaries, lifecycle, and durable design decisions |
| Verify a scientific or engineering claim | [`../validation/`](../validation/README.md) | Executable contracts, profiles, accepted evidence, and frozen campaigns |
| Reconstruct a dated investigation or transfer work | `handoff/`, dated reports, and audit packages | Scope-specific evidence and coordination history; these are not user manuals |

Existing dated reports and plans remain valid historical records. They do not
need to be moved merely to match this organization.

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
