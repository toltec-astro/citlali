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

## Full-PTC injected-source pair

The final transfer test starts from one converged restart checkpoint and runs
two otherwise identical branches:

- `control` processes the converged, source-subtracted residual normally;
- `injected` adds a known source to the pristine unit-kernel TOD immediately
  before the previous map is subtracted.

The resulting PTC input is therefore the real converged residual plus a source
with known amplitude. The injection repeats on every subsequent iteration.
Subtracting the control maps from the injected maps isolates the known source
while retaining the production cleaner, flags, weights, and fruit-loop
recurrence.

Prepare a five-iteration pair beginning at iteration 9:

```bash
$HOME/tolteca/bin/python \
  tools/fruit_loops/prepare_injected_source_pair.py \
  --input /path/to/full_policy_config.yaml \
  --restart-path /path/to/full_policy_10_iters/reduced/redu08 \
  --output-dir /path/to/injected_source_setup \
  --runtime-output-root /path/to/injected_source_run \
  --start-iteration 9 \
  --additional-iterations 5 \
  --amplitudes-mjy-beam 3981.3 4799.7 6331.6
```

The amplitudes above are the matched-APT values for the frozen 3C273
observation 133410. They define the injected truth; they are not used to
calibrate the output. The generator enables kernel production, diagnostics,
saved iterations, and exact restart state in both branches. Only
`injected_source_test.enabled` differs.

Run the generated `control` and `injected` YAML files with the same Citlali
executable and resources. Their output roots are independent, so they may run
concurrently.

Because each branch uses a fresh output root, absolute iterations 9--13 are
normally stored in directories `redu00`--`redu04`. The comparator reads the
authoritative `FRUITLOOPS_ITER` FITS header rather than treating the directory
suffix as the iteration number. It also rejects any low-level config difference
beyond the paired output root and injection enable switch.

After downloading both branches:

```bash
$HOME/tolteca/bin/python \
  tools/fruit_loops/compare_injected_source_pair.py \
  --control /path/to/run/control/reduced \
  --injected /path/to/run/injected/reduced \
  --manifest /path/to/injected_source_setup/manifest.yaml \
  --obsnum 133410 \
  --output /path/to/injected_source_metrics.csv
```

The comparator writes CSV and Markdown summaries containing the
control-subtracted source amplitude, amplitude recovery fraction, source and
kernel widths, centroid separation, iteration-to-iteration transfer-map
change, kernel difference, weight difference, and ordinary pointing-fit
metrics.

This is a diagnostic-only mode. Startup rejects it unless:

- the reduction is pointing/OOF;
- fruit loops, kernel generation, diagnostics, and saved iterations are
  enabled;
- the start iteration is at least one and below `max_iters`; and
- exactly three finite nonnegative amplitudes are supplied in
  `[a1100, a1400, a2000]` order.
