# Config Authority And Provenance Inventory - 2026-07-10

This document starts Phase 2 of the adopted refactor roadmap without changing
runtime behavior. The machine-readable source of truth is
`tools/config/config_authority_inventory.json`; its structural checks live in
`tools/config/validate_config_authority_inventory.py`.

The earlier exposure policy answers **which controls users should normally
see**. This inventory answers a different question: **which representation is
allowed to control execution**.

## Authority Contract

The target data flow is strictly one way:

```text
merged requested YAML -> typed requested config -> effective execution plan
                                            |                 |
                                            v                 v
                                  temporary legacy adapter   realized metadata
```

- Raw YAML may be read only at a declared loading boundary.
- Typed requested config preserves user intent after parsing and validation.
- An effective plan records defaults, normalization, activation, and policy
  resolution without mutating the requested values.
- Execution consumes the effective typed plan.
- Legacy processor objects may temporarily receive typed values. They must
  never populate or override typed config.
- Realized metadata records what actually ran and what products were emitted.

This contract prevents the common failure mode where requested values,
normalized values, and runtime side effects are all stored in one mutable
object and become impossible to distinguish after a reduction.

## Current Census

The inventory groups the current surface into 13 authority domains. This is a
subsystem census, not yet a leaf-by-leaf schema claim.

| Current state | Meaning | Domains |
| --- | --- | ---: |
| `typed-authoritative` | Execution policy is already typed and has no declared legacy adapter. | 4 |
| `typed-authoritative-with-adapter` | Typed policy is authoritative but one compatibility target remains. | 1 |
| `mixed-adapter` | Typed state exists, but execution still depends materially on legacy processor/map objects. | 7 |
| `external-boundary` | Configuration belongs to an external subsystem whose schema boundary must be recorded. | 1 |

Every domain currently has incomplete provenance. Runtime and product logs
already provide pieces of effective/realized state, but there is no uniform
requested/effective/realized record yet.

## Migration Order

1. **Freeze the inventory and loading boundaries.** Add a domain whenever a
   new config family appears; do not introduce execution-time raw-YAML reads.
2. **Introduce immutable effective plans at policy boundaries.** Start with
   runtime, output selection, and mapmaking activation because they already
   have focused policy helpers and broad mode coverage.
3. **Convert legacy processor reads by domain.** Raw and processed timestream
   processing require explicit read inventories before mirrored fields can be
   removed. Preserve the one-way typed-to-legacy adapter during migration.
4. **Persist provenance.** Emit requested config identity, effective policy,
   and realized product/cardinality summaries with stable schema versions.
5. **Remove adapters only after validation.** Pointing, beammap, science, and
   OOF coverage must show equivalent products and expected metadata before a
   legacy field is deleted.

## Migration Gates

A domain is complete only when all of the following are true:

1. Its raw-YAML reads are confined to the declared loader.
2. Its typed requested type covers every supported setting in that domain.
3. Normalization and cross-setting policy produce a separate effective plan.
4. Execution code reads the effective plan, not raw YAML or legacy policy
   fields.
5. Requested, effective, and realized states have stable machine-readable
   provenance.
6. Mode-appropriate product validation passes before the compatibility adapter
   is removed.

## Near-Term Scope

While Phase 1 Beammap and science validations run, Phase 2 work remains
non-operational: inventory checks, provenance schema design, and read-site
census tooling. Runtime authority will not switch until those validations land
and the relevant domain has a focused test and rollback boundary.

Validate the inventory with:

```bash
$HOME/tolteca/bin/python tools/config/validate_config_authority_inventory.py
```
