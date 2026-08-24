# Citlali Config Key Classification Starter - 2026-06-29

This is a first-pass classification for reducing the user-facing Citlali config
surface. It should guide profile/template design, not remove or rename keys yet.

Static inventory:

- Full default config leaf keys: 491
- `timestream` leaf keys: 353
- Static source-reference scan: 233 simple references

The source-reference scan is intentionally conservative and regex-based. It can
miss dynamic and multi-line accesses. Treat it as an inventory aid only.

## Classification Levels

| Level | Meaning |
| --- | --- |
| Core | User should see this in compact templates. |
| Common Advanced | Useful, but not needed in the shortest examples. |
| Expert | Valid maintainer/operator knob, hidden behind `expert:` overrides. |
| Diagnostic | Primarily for investigation, heavy products, or sidecars. |
| Experimental | Still being evaluated or not broadly validated. |
| Deprecated | Kept for compatibility, should warn once replacement exists. |

## First-Pass Group Classification

| Config group | Initial level | Reason |
| --- | --- | --- |
| `runtime.reduction_type` | Core | Selects science, pointing, OOF, or beammap; pointing and OOF share the Pointing numerical processor. |
| `runtime.output_dir` | Core | Required operational output choice. |
| `runtime.n_threads` | Core | Common operational performance choice. |
| `runtime.parallel_policy` | Core | Common operational performance choice. |
| `runtime.use_subdir` | Core | Controls reduction directory layout. |
| `runtime.verbose` | Common Advanced | Useful but should not dominate compact configs. |
| `runtime.interp_over_gaps` | Expert | Currently required true; should be profile-managed. |
| `inputs` | Core | Required, but likely supplied by TolTECA or an input manifest. |
| `kids.fitter` | Expert | Fit internals should usually come from a profile. |
| `kids.solver.fitreportdir` | Common Advanced | Operational path may need user control. |
| `kids.solver.parallel_policy` | Expert | Separate from Citlali runtime parallelism. |
| `mapmaking.enabled` | Core | Allows TOD-only or map reductions. |
| `mapmaking.cunit` | Core | Science output unit. |
| `mapmaking.grouping` | Core | Common reducer choice. |
| `mapmaking.method` | Core | Naive/jinc/ML selection. |
| `mapmaking.pixel_axes` | Common Advanced | Important but often profile-derived. |
| `mapmaking.pixel_size_arcsec` | Core | Common science/product choice. |
| `mapmaking.crpix*`, `crval*`, `x_size_pix`, `y_size_pix` | Common Advanced | Needed for manual maps, not normal shortest path. |
| `mapmaking.jinc_filter` | Expert | Hot-path algorithm tuning. |
| `mapmaking.maximum_likelihood` | Experimental | Under-development mapmaker controls. |
| `coadd.enabled` | Core | Common product-level choice. |
| `noise_maps.enabled` | Core | Common product/runtime choice. |
| `noise_maps.n_noise_maps` | Core | Common when noise maps are enabled. |
| `noise_maps.randomize_dets` | Expert | Algorithm policy. |
| `noise_maps.write_realizations` | Common Advanced | Disk/product-volume choice. |
| `noise_maps.products` | Common Advanced | Product/statistics policy. |
| `source.map_regime` | Diagnostic | Interpretation metadata for diagnostics. |
| `pointing.source_strategy` | Core for pointing | Important high-level pointing behavior. |
| `beammap.iter_max` | Core for beammap | Main beammap iteration control. |
| `beammap.derotate` | Core for beammap | Common beammap product choice. |
| `beammap.subtract_reference_det` | Common Advanced | Operational policy. |
| `beammap.reference_det` | Common Advanced | Operational policy. |
| `beammap.phase_strategy` | Expert | Algorithm phase internals. |
| `beammap.detector_weighting` | Common Advanced | Useful for detector maps. |
| `beammap.detector_tod_output` | Diagnostic | Sidecar product control. |
| `beammap.fitting` | Common Advanced | Source-fit support choice. |
| `beammap.flagging` | Common Advanced | User may choose strictness, not raw thresholds. |
| `beammap.priors.enabled`, `beammap.priors.filepath` | Core for prior profiles | High-level prior use. |
| Other `beammap.priors.*` | Expert | Scoring/alignment details. |
| `beammap.rfi_mask` | Diagnostic | Specialized artifact mitigation. |
| `beammap.scan_band_mask` | Diagnostic | Specialized artifact mitigation. |
| `beammap.split_fits_by_flag` | Diagnostic | Product inspection feature. |
| `timestream.type` | Common Advanced | Usually `xs`; profile should own default. |
| `timestream.enabled` | Core | Allows map-only/TOD-only pathway control. |
| `timestream.precompute_pointing` | Deprecated or Experimental | Currently marked ignored. |
| `timestream.chunking` | Common Advanced | Sometimes needed for performance/scans. |
| `timestream.polarimetry` | Core for polarimetry profile | Hidden otherwise. |
| `timestream.output` | Diagnostic | TOD product selection. |
| `timestream.raw_time_chunk.despike.enabled` | Common Advanced | High-level cleaning/flagging choice. |
| `timestream.raw_time_chunk.despike.*` thresholds | Expert | Algorithm tuning. |
| `timestream.raw_time_chunk.downsample` | Expert | Runtime/signal-processing tuning. |
| `timestream.raw_time_chunk.filter.enabled` | Common Advanced | High-level filtering choice. |
| `timestream.raw_time_chunk.filter.*` | Expert | Signal-processing tuning. |
| `timestream.raw_time_chunk.IIR_filter` | Expert | Signal-processing tuning. |
| `timestream.raw_time_chunk.kernel` | Diagnostic | Synthetic characterization feature. |
| `timestream.raw_time_chunk.flux_calibration` | Common Advanced | Product calibration control. |
| `timestream.raw_time_chunk.extinction_correction` | Common Advanced | Product calibration control. |
| `timestream.raw_time_chunk.flagging.delta_f_min_Hz` | Expert | Instrument/tone flagging detail. |
| `timestream.raw_time_chunk.flagging.network_step_mask` | Diagnostic | Specialized artifact mitigation. |
| `timestream.raw_time_chunk.flagging.impulsive_*` | Diagnostic | Specialized artifact diagnostics/mitigation. |
| `timestream.raw_time_chunk.line_audit` | Diagnostic | Large specialized diagnostic surface. |
| `timestream.processed_time_chunk.clean.enabled` | Core | Main PTC cleaning switch. |
| `timestream.processed_time_chunk.clean.grouping` | Core | Common reducer choice. |
| `timestream.processed_time_chunk.clean.standard_pca.enabled` | Common Advanced | Profile can pick default. |
| `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut` | Common Advanced | Common science tuning. |
| Other `timestream.processed_time_chunk.clean.*` | Expert/Experimental | Adaptive/null/MP/correlation internals. |
| `timestream.processed_time_chunk.weighting.type` | Core | Main map-weight policy. |
| `timestream.processed_time_chunk.weighting.source_mask_radius_arcsec` | Common Advanced | Important for source-aware weights. |
| Other `timestream.processed_time_chunk.weighting.*` | Expert/Diagnostic | Validation/correlation/busy-row internals. |
| `timestream.processed_time_chunk.flagging.second_pass_local.enabled` | Common Advanced | High-level post-clean deglitch choice. |
| Other `timestream.processed_time_chunk.flagging.*` | Diagnostic | Specialized artifact mitigation. |
| `timestream.fruit_loops.enabled` | Core | High-level source-subtraction iteration choice. |
| `timestream.fruit_loops.max_iters` | Core | Common iteration control. |
| `timestream.fruit_loops.path`, `type`, `save_all_iters` | Common Advanced | Operational iteration/product control. |
| Other `timestream.fruit_loops.*` | Expert | Template/support/weight-feedback internals. |
| `timestream.learning.enabled` | Common Advanced | High-level feature switch. |
| `timestream.learning.diagnostics_enabled` | Diagnostic | Diagnostic output policy. |
| Other `timestream.learning.*` | Expert/Experimental | Learned-mask/pathology thresholds. |
| `post_processing.source_fitting` | Core for pointing | Important source-fit behavior. |
| `post_processing.source_fitting.gauss_model` | Common Advanced | Fit-bound tuning. |
| `post_processing.map_filtering.enabled` | Core | High-level filtered-map product choice. |
| `post_processing.map_filtering.type` | Common Advanced | Filter algorithm choice. |
| `post_processing.map_filtering.edge_guard` | Expert | FFT/filter edge-conditioning detail. |
| `post_processing.source_finding.enabled` | Common Advanced | Optional source catalog behavior. |
| `post_processing.source_finding.*` | Expert | Source finder threshold/window details. |
| `wiener_filter.template_type` | Common Advanced | Filter shape choice. |
| `wiener_filter.template_fwhm_arcsec` | Common Advanced | Beam/template widths. |
| Other `wiener_filter.*` | Expert | Convergence/stability internals. |

## Compact Profile Defaults Should Own These

These groups should usually not appear in user-authored compact configs:

- `runtime.interp_over_gaps`
- `kids.fitter`
- `mapmaking.jinc_filter`
- `mapmaking.maximum_likelihood`
- `beammap.phase_strategy`
- `beammap.priors` scoring/alignment thresholds
- `timestream.raw_time_chunk` filter/despike/line-audit thresholds
- `timestream.processed_time_chunk` adaptive cleaner internals
- `timestream.learning` thresholds
- `post_processing.map_filtering.edge_guard`
- Wiener denominator convergence thresholds

Users should still be able to set them explicitly under an `expert:` override
until a maintainer deliberately deprecates a key.

## Immediate Follow-Up

1. Collect representative current configs from science, pointing, beammap, and
   TOD-output use cases.
2. Mark each key as "appears in real user config", "TolTECA-generated",
   "maintainer-only", or "legacy/example-only".
3. Convert this starter table into machine-readable metadata only after review.
4. Use that metadata to generate compact profile docs and validation warnings.
