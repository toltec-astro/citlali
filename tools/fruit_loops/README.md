# Fruit-loop feedback diagnostics

This package prepares controlled, single-observation ablations of the pointing
fruit-loop recurrence. It does not change production defaults.

Generate the observation 133410 configs from the frozen five-iteration
low-level YAML:

```bash
$HOME/tolteca/bin/python tools/fruit_loops/prepare_feedback_ablation.py \
  --input /path/to/citlali_rc1_fruitloops5_o133410.yaml \
  --output-dir /path/to/setup \
  --runtime-output-root \
    /work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloops5_rc1_ablation/obs133410
```

The generator verifies the seed policy and writes:

- `full_policy_diagnostic`: unchanged policy, with diagnostics enabled;
- `learning_disabled`: only reduction learning is disabled;
- `weight_feedback_disabled`: only map-template weight taper is disabled;
- `recompute_weights_after_addback`: only detector weights are recomputed
  after restoring the source model; and
- `all_three`: all three diagnostic changes together.

Every variant keeps `max_iters: 5` and `save_all_iters: true`, and every
variant has an independent output root. The unchanged-policy diagnostic is a
control for the new instrumentation and supplies the same stage-level
diagnostics as the ablations.

Run each generated low-level config with the same executable and resources:

```bash
/path/to/citlali -l info /path/to/config.yaml
```

Do not submit two configs with the same output directory concurrently.

After downloading all `redu00` through `redu04` directories, create one
comparison table:

```bash
$HOME/tolteca/bin/python tools/fruit_loops/compare_feedback_ablation.py \
  --run existing_full=/path/to/existing/obs133410/reduced \
  --run learning_disabled=/path/to/learning_disabled/reduced \
  --run weight_feedback_disabled=/path/to/weight_feedback_disabled/reduced \
  --run recompute_weights=/path/to/recompute_weights_after_addback/reduced \
  --run all_three=/path/to/all_three/reduced \
  --output /path/to/fruit_loop_ablation_metrics.csv
```

The table records fitted source amplitude, widths, S/N, centroid, kernel peak,
median map weight, absolute and relative successive-map RMS changes, and the
size and off-source fraction of the flux-selected feedback model.
