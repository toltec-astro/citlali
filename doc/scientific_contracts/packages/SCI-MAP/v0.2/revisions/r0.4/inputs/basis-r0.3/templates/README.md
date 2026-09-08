# SCI-MAP v0.2/r0.3 machine-readable prospective templates

Status: **prospective candidate templates**. They provide field names,
enumerations, and structural shapes. Schema acceptance alone cannot establish
scientific correctness, authority, implementation conformity, or any
requirement result.

| File | Purpose |
| --- | --- |
| `REQUIREMENT_RESULT_RECORD.schema.json` | One ECS requirement result plus the independent evidence-artifact realization axis. |
| `REQUIREMENT_RESULTS_INITIAL.json` | Exactly 52 stable requirement placeholders, each initialized to `not_assessed` and artifact `not_produced`. |
| `MAP_LIFECYCLE_AND_PRODUCT_ROLE_RECORD.schema.json` | Selected role/parent/membership, reached stages, separate attempt outcome, product realization, failure evidence, companion status, and empty-support evidence. |
| `COVERAGE_CUT_OVERRIDE_RECORD.schema.json` | Typed `coverage_cut_above_one_override_requested` fact, reached-stage bindings, and explicit optional empty-support nonproduction record. |
| `FULL_PROCEDURE_COMPARISON_RECORD.schema.json` | Complete stable `SCI-MAP-FULL-PROCEDURE-COMPARISON-SPACE` authorization; coadd-procedure and whole-chain types remain unavailable. |
| `WCS_COMPARISON_RECORD.schema.json` | Exact finite WCS population and representation comparison. |
| `VAL_HANDOFF_DECISION_RECORD.schema.json` | Six r0.3 candidate profile-status axes plus the currently unavailable PTC/VAL object-decision record. |

## Required semantic review beyond structural validation

A reviewer must additionally establish all of the following from the shared
authority and exact source/profile bindings:

- lifecycle `stages` is an ordered prefix of
  `requested/effective/observation_resolved/applied/realized`, with exact
  parent links and no duplicate or skipped stage;
- `succeeded` has all five stages and exact product contents;
  pre-application stop has no applied identity, attempt outcome, or product;
  `failed` after application retains the applied attempt and exact failure
  evidence but no product; and
  `not_produced` empty support retains its proof and no product or empty bundle;
- stronger-role ordered membership equals its exact base parent's membership;
  a separately selected stronger-role admission may reject an observation
  before constructing that request's explicitly identified base parent; if
  the parent is realized, its ordered membership equals the stronger role's;
- `blocked` names the exact unavailable authority/dependency/profile/source
  binding/prerequisite, while `not_applicable` includes an exact owner-approved
  rationale;
- `pass` has complete positive evidence for the exact candidate, fixture,
  comparison predicate, role, and shared-authority digest;
- evidence-artifact realization is assessed independently of the requirement
  result;
- the current profile-status record is exactly
  `candidate/pending/pending/pending/unavailable/unavailable`; its VAL Boolean
  projection is false and no object decision is realized;
- a cut above one has matching authority, scope, exact value, and every reached
  lifecycle stage; the number never self-authorizes;
- a numerical procedure difference has every stable comparison-space field and two
  exact maps into one authorized common domain; a state transition alone is
  insufficient; a response-bearing coadd binds one exact same family `R`,
  fixed coadd state, and an exact composition theorem, and no finite difference
  is assumed to be a Jacobian;
- the WCS population was frozen before FITS generation, includes every
  required center and outer-boundary vertex/corner, and drops no invalid
  required point; and
- hashes, identities, causes, row maps, units, frames, gauges, support, and
  provenance refer to immutable exact bytes rather than filenames or equal
  shapes.
