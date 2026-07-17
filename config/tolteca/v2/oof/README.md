# OOF Configuration

Review and ordinary edits should start with:

1. `81_oof_defaults.yaml` for PSF-preserving source strategy, map geometry,
   source protection, cleaning, weighting, and bounded iteration choices;
2. `82_oof_products.yaml` for noise, filtering, fitting, source finding, and
   retained RTC/PTC TOD products; and
3. `71_oof_runtime.yaml` for executable and CPU allocation.

TolPROJ generates `72_oof_observation.yaml`. Do not edit
`60_oof_internal_policy.yaml`; it is the complete accepted compatibility
policy. Advanced and expert override files are deliberately empty.

Source finding is experimental and disabled. The unchanged files merge exactly
to accepted OOF `redu02`.
