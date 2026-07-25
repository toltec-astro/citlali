# Fruit-loop feedback diagnostics

This package prepares controlled, single-observation ablations of the pointing
fruit-loop recurrence. It does not change production defaults.

Generate the observation 133410 configs from the frozen five-iteration
low-level YAML:

```bash
$HOME/tolteca/bin/python tools/fruit_loops/prepare_feedback_ablation.py \
  --input /path/to/citlali_rc1_fruitloops5_o133410.yaml \
  --output-dir /path/to/setup \
  --matrix followup \
  --runtime-output-root \
    /work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloops5_rc1_ablation/obs133410
```

The generator verifies the seed policy. `--matrix initial` writes the five
completed first-round variants:

- `full_policy_diagnostic`: unchanged policy, with diagnostics enabled;
- `learning_disabled`: only reduction learning is disabled;
- `weight_feedback_disabled`: only map-template weight taper is disabled;
- `recompute_weights_after_addback`: only detector weights are recomputed
  after restoring the source model; and
- `all_three`: all three diagnostic changes together.

`--matrix followup` writes the second-round matrix:

| Variant | Isolated question |
|---|---|
| `snr_only_model` | Does removing broad absolute-flux support change the trajectory at the existing S/N threshold of 100? |
| `snr_only_s50` | How does a larger high-S/N source model change the trajectory? |
| `snr_only_s200` | How does a smaller high-S/N source model change the trajectory? |
| `adaptive_peak_5pct` | Does a compact, source-centered model selected above 5% of the local peak converge? |
| `adaptive_local_snr5` | Does an independently defined compact model selected above local S/N 5 converge? |
| `ptc_cleaning_disabled` | Does growth disappear when the production PTC cleaner is removed? |
| `ptc_pca_one_mode` | Does growth decrease with weaker PCA cleaning? |
| `ptc_pca_ten_modes` | Does growth increase with stronger PCA cleaning? |
| `ptc_source_mask_30arcsec` | Does explicit source protection inside the PTC cleaner change the trajectory? |
| `projection_bilinear` | Is the map-to-TOD Jinc projection materially involved? |
| `projection_legacy_trunc` | Does the historical truncating projection and center convention change the trajectory? |
| `naive_mapmaking` | Is the Jinc mapmaking/projection pair materially involved? |
| `full_policy_10_iters` | Does the unchanged policy reach an asymptote or continue drifting through ten iterations? |

`--matrix all` writes both matrices and remains the default.

The obsnum 133410 follow-up matrix completed on Unity. `snr_only_s200`
intentionally stopped at the no-op guard because it selected zero
detector-samples. The other twelve variants completed, and their results are
recorded in
`doc/FRUIT_LOOP_FEEDBACK_INVESTIGATION_2026-07-24.md`.

Every variant keeps `save_all_iters: true` and has an independent output root.
All variants retain five iterations except `full_policy_10_iters`. The
unchanged-policy diagnostic is a control for the instrumentation and supplies
the same stage-level diagnostics as the ablations.

Run each generated low-level config with the same executable and resources:

```bash
/path/to/citlali -l info /path/to/config.yaml
```

Do not submit two configs with the same output directory concurrently.

After downloading every saved iteration, create one comparison table. The
comparison tool discovers contiguous `reduNN` directories, including all ten
iterations of `full_policy_10_iters`:

```bash
$HOME/tolteca/bin/python tools/fruit_loops/compare_feedback_ablation.py \
  --run existing_full=/path/to/existing/obs133410/reduced \
  --run learning_disabled=/path/to/learning_disabled/reduced \
  --run weight_feedback_disabled=/path/to/weight_feedback_disabled/reduced \
  --run recompute_weights=/path/to/recompute_weights_after_addback/reduced \
  --run all_three=/path/to/all_three/reduced \
  --run snr_only_model=/path/to/snr_only_model/reduced \
  --output /path/to/fruit_loop_ablation_metrics.csv
```

The table records fitted source amplitude, widths, S/N, centroid, kernel peak,
median map weight, absolute and relative successive-map RMS changes, and the
size and off-source fraction of the flux-selected feedback model.
