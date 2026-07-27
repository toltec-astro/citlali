# Fruit-loop population Stage B Unity bundle

This is the frozen 92-observation completion setup authorized by the passing
Stage A gate in
`validation/fruit_loop_population_stage_a_analysis_2026-07-26/stage_a_gate.json`.

The bundle contains one config per remaining observation, a checksummed job
table, and Unity-side scripts that:

1. require the exact Stage A Citlali binary SHA256
   `0f7685ad2b89cc2fc2cbe330c9e5ed75fc8972dc1bf60ab37e3a4b9209965330`;
2. verify every config, APT, raw input, output collision, and executable;
3. require at least 350 GiB free before launch;
4. launch one observation per Slurm array task, throttled to four concurrent
   tasks by default;
5. restore the setup config's mode and verify every copied config after each
   successful task; and
6. report scheduler state, saved-iteration counts, and potential error-level
   log lines.

The 92 Stage B observations and 16 Stage A observations are disjoint and
together cover the frozen 108-observation population.

Do not edit a generated config on Unity. Regenerate the complete bundle if the
policy changes. The default output root is:

```text
/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloop_population_v1/stage_b
```

The exact owner-run upload, launch, monitoring, and download commands are in
`handoff/FRUIT_LOOP_POPULATION_STAGE_B_UNITY_HANDOFF_2026-07-26.md`.
