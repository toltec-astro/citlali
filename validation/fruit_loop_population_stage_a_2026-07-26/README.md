# Fruit-loop population Stage A Unity bundle

This is the frozen 16-observation sentinel setup described in
`doc/FRUIT_LOOP_POPULATION_EXTENSION_PLAN_2026-07-26.md`.

The bundle contains one config per observation, a checksummed job table, and
Unity-side scripts that:

1. copy the selected Citlali executable to a SHA256-named immutable snapshot;
2. verify every config, APT, raw input, output collision, and executable;
3. launch one observation per Slurm array task, throttled to four concurrent
   tasks by default; and
4. report scheduler state, saved-iteration counts, and potential error-level
   log lines.

Do not edit a generated config on Unity. Regenerate the complete bundle if the
policy changes. The default output root is:

```text
/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloop_population_v1/stage_a
```

The exact owner-run commands, download verification, and cleanup boundaries
are in `handoff/FRUIT_LOOP_POPULATION_STAGE_A_UNITY_HANDOFF_2026-07-26.md`.
