# ADR 0001: Configuration State Transitions

- **Status:** Accepted
- **Recorded:** 2026-07-16
- **Decision owners:** Citlali project owner and engineering

## Context

Historical Citlali configuration mixed user requests, defaults, automatic
choices, calibration-dependent values, mutable processor fields, and output
metadata. Typed values initially mirrored legacy state in both directions,
which made it unclear whether a value described what the user requested, what
Citlali intended to execute, or what one observation actually realized.

TolTECA also owns the user-facing numbered overlay workflow. Citlali receives a
generated low-level YAML file without enough information to reconstruct every
upstream authoring decision.

## Decision

Configuration moves in one direction:

```text
TolTECA overlays
  -> generated low-level Citlali YAML
  -> immutable typed request
  -> effective execution plan
  -> observation-resolved state
  -> realized execution metadata
  -> versioned provenance
```

The generated low-level YAML is Citlali's immutable input boundary. TolTECA
owns overlay discovery, order, merge, and upstream input selection. Citlali
records the source paths and exact merged content it received without
inventing unavailable overlay provenance.

Each migrated fact has one typed authority. Context-free activation,
normalization, defaults, and compatibility rules produce a separate effective
plan without mutating the request. Observation metadata and calibration
produce complete observation-resolved values before processor state changes.
Execution records realized state and completed products separately.

An established numerical processor may receive a legacy-shaped input through
one narrow, one-way adapter. Legacy state never writes back into the request or
effective plan. Processor-derived results are realized metadata, not reverse
configuration synchronization.

YAML reads are confined to declared startup and observation-resolution
boundaries. Numerical algorithms and hot loops do not parse configuration.
Compact config remains translation tooling rather than production authority.

## Consequences

- Requested expert values remain reproducible even when their feature is
  disabled or an effective value differs.
- Provenance can label requested, effective, observation-resolved, and realized
  state without conflating them.
- A second observation cannot inherit missing calibration or policy from the
  first.
- Compatibility adapters are explicit migration debt with a removal condition;
  they are not permanent parallel authorities.
- Adding a field requires a typed owner, validation, resolution stage, adapter
  disposition, provenance, and mode-appropriate evidence.
- TolTECA overlay rollout and compact authoring remain separate projects from
  Citlali's low-level execution contract.

## Rejected Alternatives

- **Bidirectional typed/legacy mirrors:** creates two writable authorities and
  makes provenance phase-dependent.
- **Mutating the request with automatic choices:** destroys the record of user
  intent.
- **Reading YAML in processors:** spreads policy across algorithms and prevents
  complete startup validation.
- **Reconstructing TolTECA overlay history:** Citlali does not receive enough
  authoritative source information to do so safely.

## Supersession

This ADR may be superseded only by a versioned configuration architecture that
preserves immutable intent, explicit resolution stages, observation atomicity,
and unambiguous provenance. Removing the legacy adapter after all consumers are
typed is consistent with this decision and does not supersede it.

## Evidence

- [`../CONFIG_AUTHORITY_AND_PROVENANCE_INVENTORY_2026-07-10.md`](../CONFIG_AUTHORITY_AND_PROVENANCE_INVENTORY_2026-07-10.md)
- [`../REFACTOR_STATUS.md`](../REFACTOR_STATUS.md), Phase 2 closeout
- `tools/config/config_leaf_contract_resolved.json`
- `tools/config/run_config_preflight.py --require-all`
