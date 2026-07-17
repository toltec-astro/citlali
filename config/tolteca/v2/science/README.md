# Science Configuration

Review and ordinary edits should start with:

1. `81_science_defaults.yaml` for mapmaking, cleaning, weighting, learning,
   fruit-loop activation, iterations, S/N and flux cuts, and iteration retention;
2. `82_science_products.yaml` for coadds, noise, filtering, fitting, source
   finding, and retained TOD products; and
3. `71_science_runtime.yaml` for executable and CPU allocation.

TolPROJ generates `72_science_observation.yaml`. Do not edit
`60_science_internal_policy.yaml`; it is the complete accepted compatibility
policy. Advanced and expert override files are deliberately empty.

The unchanged files merge exactly to accepted science `redu31`.
