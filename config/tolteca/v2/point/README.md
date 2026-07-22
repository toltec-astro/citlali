# Pointing Configuration

Review and ordinary edits should start with:

1. `81_pointing_defaults.yaml` for source strategy, map geometry, source
   protection, cleaning, weighting, fruit loops, and learning;
2. `82_pointing_products.yaml` for noise, filtering, fitting, source finding,
   and retained RTC/PTC TOD products; and
3. `71_pointing_runtime.yaml` for executable and CPU allocation.

TolPROJ generates `72_pointing_observation.yaml`. Do not edit
`60_pointing_internal_policy.yaml`; it is the complete accepted compatibility
policy. Advanced and expert override files are deliberately empty.

Source finding is experimental and disabled. Routine pointings produce raw
maps and fits; Wiener filtering remains available as an explicit product
choice but is disabled by default. The predecessor full-Wiener validation
remains recorded by accepted point `redu66`.
