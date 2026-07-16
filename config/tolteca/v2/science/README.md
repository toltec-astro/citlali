# Science Configuration Review

This directory is the proposed human-facing science configuration structure.
It is a review prototype: TolPROJ does not install it yet, and the observation
file still contains placeholders.

## Suggested Review Order

1. `81_science_defaults.yaml` - routine science analysis choices, including
   the complete user-facing fruit-loop block and source-model cut levels.
2. `82_science_products.yaml` - requested maps, filtering, noise, fitting, and
   retained TOD products.
3. `71_science_runtime.yaml` - executable, threads, and output layout.
4. `72_science_observation.yaml` - data path and observation structure TolPROJ
   will populate from project metadata and directory layout.
5. `90_science_advanced_overrides.yaml` - optional additional user-facing
   controls, empty by default.
6. `99_science_expert_overrides.yaml` - exceptional low-level tuning, empty by
   default.

`60_science_internal_policy.yaml` is the complete generated compatibility
policy. Normal reducers should not edit or review it field by field. Automated
tests prove that all numbered files merge to the accepted science policy.

Nothing in this directory requires a Citlali compilation. Do not copy it to
Unity yet; the structure should be accepted here before TolPROJ is updated.
