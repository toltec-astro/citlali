# Compact Surface Coverage Audit

This audit checks whether low-level keys classified as `user-facing` are
represented by current compact-config fields for the representative
pointing, OOF, beammap, and science baselines. It is a shadow/config
tooling check only; it does not change Citlali runtime parsing.

## Summary

- Configs checked: 4
- Unique mode/path user-facing keys: 282
- Covered by compact fields: 265
- Profile-owned inactive defaults: 17
- Actionable gaps: 0
- Actionable coverage: 100.0%

`Covered` means the current compact fields expand back to that low-level
path without using `expert:`. `Profile-owned` means the key is present
in the low-level baseline but is inactive for that reduction mode and
should be owned by a profile/default rather than normal user authoring.
`Gap` means the value is still preserved only through the expert escape
hatch and needs a compact-field or policy decision.

| Mode | User-Facing Paths | Covered | Profile-Owned | Gaps | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `beammap` | 80 | 80 | 0 | 0 | 100.0% |
| `oof` | 56 | 51 | 5 | 0 | 100.0% |
| `pointing` | 87 | 80 | 7 | 0 | 100.0% |
| `science` | 59 | 54 | 5 | 0 | 100.0% |

## Gaps

_No user-facing compact coverage gaps were found._

## Profile-Owned Inactive Defaults

| Mode | Low-Level Path | Reason |
| --- | --- | --- |
| `oof` | `beammap.derotate` | Beammap defaults are present in the low-level baseline but inactive for oof reductions; the mode/profile owns them. |
| `oof` | `beammap.iter_max` | Beammap defaults are present in the low-level baseline but inactive for oof reductions; the mode/profile owns them. |
| `oof` | `beammap.iter_tolerance` | Beammap defaults are present in the low-level baseline but inactive for oof reductions; the mode/profile owns them. |
| `oof` | `beammap.reference_det` | Beammap defaults are present in the low-level baseline but inactive for oof reductions; the mode/profile owns them. |
| `oof` | `beammap.subtract_reference_det` | Beammap defaults are present in the low-level baseline but inactive for oof reductions; the mode/profile owns them. |
| `pointing` | `beammap.convergence_radius_arcsec` | Beammap defaults are present in the low-level baseline but inactive for pointing reductions; the mode/profile owns them. |
| `pointing` | `beammap.derotate` | Beammap defaults are present in the low-level baseline but inactive for pointing reductions; the mode/profile owns them. |
| `pointing` | `beammap.detector_weighting.mode` | Beammap defaults are present in the low-level baseline but inactive for pointing reductions; the mode/profile owns them. |
| `pointing` | `beammap.iter_max` | Beammap defaults are present in the low-level baseline but inactive for pointing reductions; the mode/profile owns them. |
| `pointing` | `beammap.iter_tolerance` | Beammap defaults are present in the low-level baseline but inactive for pointing reductions; the mode/profile owns them. |
| `pointing` | `beammap.reference_det` | Beammap defaults are present in the low-level baseline but inactive for pointing reductions; the mode/profile owns them. |
| `pointing` | `beammap.subtract_reference_det` | Beammap defaults are present in the low-level baseline but inactive for pointing reductions; the mode/profile owns them. |
| `science` | `beammap.derotate` | Beammap defaults are present in the low-level baseline but inactive for science reductions; the mode/profile owns them. |
| `science` | `beammap.iter_max` | Beammap defaults are present in the low-level baseline but inactive for science reductions; the mode/profile owns them. |
| `science` | `beammap.iter_tolerance` | Beammap defaults are present in the low-level baseline but inactive for science reductions; the mode/profile owns them. |
| `science` | `beammap.reference_det` | Beammap defaults are present in the low-level baseline but inactive for science reductions; the mode/profile owns them. |
| `science` | `beammap.subtract_reference_det` | Beammap defaults are present in the low-level baseline but inactive for science reductions; the mode/profile owns them. |

## Covered User-Facing Paths

| Mode | Low-Level Path | Compact Field(s) |
| --- | --- | --- |
| `beammap` | `beammap.convergence_radius_arcsec` | `beammap.convergence_radius_arcsec` |
| `beammap` | `beammap.derotate` | `beammap.derotate` |
| `beammap` | `beammap.detector_tod_output.enabled` | `beammap.detector_tod` |
| `beammap` | `beammap.detector_weighting.mode` | `beammap.detector_weighting` |
| `beammap` | `beammap.iter_max` | `beammap.iterations` |
| `beammap` | `beammap.iter_tolerance` | `beammap.convergence_tolerance` |
| `beammap` | `beammap.priors.enabled` | `beammap.priors` |
| `beammap` | `beammap.priors.filepath` | `beammap.priors` |
| `beammap` | `beammap.reference_det` | `beammap.reference_det` |
| `beammap` | `beammap.rfi_mask.enabled` | `beammap.rfi_mask` |
| `beammap` | `beammap.scan_band_mask.enabled` | `beammap.scan_band_mask` |
| `beammap` | `beammap.split_fits_by_flag.enabled` | `beammap.split_fits` |
| `beammap` | `beammap.subtract_reference_det` | `beammap.subtract_reference_det` |
| `beammap` | `coadd.enabled` | `map.coadd` |
| `beammap` | `kids.solver.fitreportdir` | `runtime.fitreport_dir` |
| `beammap` | `mapmaking.cunit` | `map.unit` |
| `beammap` | `mapmaking.method` | `map.method` |
| `beammap` | `mapmaking.pixel_axes` | `map.pixel_axes` |
| `beammap` | `mapmaking.pixel_size_arcsec` | `map.pixel_size_arcsec` |
| `beammap` | `noise_maps.enabled` | `products.noise` |
| `beammap` | `noise_maps.n_noise_maps` | `products.noise_count` |
| `beammap` | `noise_maps.randomize_dets` | `products.noise_randomize_dets` |
| `beammap` | `post_processing.map_filtering.enabled` | `products.map_filtering` |
| `beammap` | `post_processing.map_filtering.normalize_errors` | `products.map_filtering` |
| `beammap` | `post_processing.map_filtering.type` | `products.map_filtering` |
| `beammap` | `post_processing.map_histogram_n_bins` | `products.map_histogram_bins` |
| `beammap` | `post_processing.source_finding.enabled` | `products.source_finding` |
| `beammap` | `post_processing.source_finding.mode` | `products.source_finding` |
| `beammap` | `post_processing.source_finding.source_sigma` | `products.source_finding` |
| `beammap` | `post_processing.source_finding.source_window_arcsec` | `products.source_finding` |
| `beammap` | `post_processing.source_fitting.bounding_box_arcsec` | `source.fit_box_arcsec` |
| `beammap` | `post_processing.source_fitting.fitting_radius_arcsec` | `source.fit_radius_arcsec` |
| `beammap` | `post_processing.source_fitting.model` | `source.fit_model` |
| `beammap` | `runtime.n_threads` | `runtime.threads` |
| `beammap` | `runtime.output_dir` | `output.dir` |
| `beammap` | `runtime.reduction_type` | `<derived>` |
| `beammap` | `runtime.use_subdir` | `output.subdir` |
| `beammap` | `runtime.verbose` | `output.verbose`, `products.diagnostics` |
| `beammap` | `source.map_regime` | `source.map_regime` |
| `beammap` | `timestream.chunking.chunk_mode` | `processing.chunking` |
| `beammap` | `timestream.chunking.force_chunking` | `processing.chunking` |
| `beammap` | `timestream.chunking.value` | `processing.chunking` |
| `beammap` | `timestream.fruit_loops.adaptive_support_radius_arcsec` | `processing.fruitloops_support_radius_arcsec` |
| `beammap` | `timestream.fruit_loops.adaptive_support_radius_fwhm` | `processing.fruitloops_support_radius_fwhm` |
| `beammap` | `timestream.fruit_loops.center_keep_radius_arcsec` | `processing.fruitloops_center_keep_radius_arcsec` |
| `beammap` | `timestream.fruit_loops.enabled` | `processing.fruitloops` |
| `beammap` | `timestream.fruit_loops.max_iters` | `processing.fruitloops_iters` |
| `beammap` | `timestream.fruit_loops.path` | `processing.fruitloops_source` |
| `beammap` | `timestream.fruit_loops.save_all_iters` | `processing.fruitloops_save_all_iters` |
| `beammap` | `timestream.fruit_loops.type` | `processing.fruitloops_type` |
| `beammap` | `timestream.polarimetry.enabled` | `processing.polarimetry` |
| `beammap` | `timestream.polarimetry.grouping` | `processing.polarimetry` |
| `beammap` | `timestream.polarimetry.ignore_hwpr` | `processing.polarimetry` |
| `beammap` | `timestream.processed_time_chunk.clean.adaptive_selector.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.grouping[]` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.marchenko_pastur.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.null_model.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.enabled` | `processing.clean`, `processing.standard_pca` |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.n_calc` | `processing.clean`, `processing.standard_pca` |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a1100[]` | `processing.clean`, `processing.standard_pca` |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a1400[]` | `processing.clean`, `processing.standard_pca` |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a2000[]` | `processing.clean`, `processing.standard_pca` |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.stddev_limit` | `processing.clean`, `processing.standard_pca` |
| `beammap` | `timestream.processed_time_chunk.flagging.second_pass_local.enabled` | `processing.second_pass_local` |
| `beammap` | `timestream.processed_time_chunk.output.enabled` | `products.tod` |
| `beammap` | `timestream.processed_time_chunk.weighting.type` | `processing.weighting` |
| `beammap` | `timestream.raw_time_chunk.IIR_filter.enabled` | `processing.raw` |
| `beammap` | `timestream.raw_time_chunk.despike.enabled` | `processing.raw` |
| `beammap` | `timestream.raw_time_chunk.downsample.enabled` | `processing.raw` |
| `beammap` | `timestream.raw_time_chunk.extinction_correction.enabled` | `processing.raw` |
| `beammap` | `timestream.raw_time_chunk.filter.enabled` | `processing.raw` |
| `beammap` | `timestream.raw_time_chunk.flux_calibration.enabled` | `processing.raw` |
| `beammap` | `timestream.raw_time_chunk.line_audit.enabled` | `processing.raw` |
| `beammap` | `timestream.raw_time_chunk.output.enabled` | `processing.raw`, `products.tod` |
| `beammap` | `wiener_filter.lowpass_only` | `filter.wiener` |
| `beammap` | `wiener_filter.template_fwhm_arcsec.a1100` | `filter.wiener` |
| `beammap` | `wiener_filter.template_fwhm_arcsec.a1400` | `filter.wiener` |
| `beammap` | `wiener_filter.template_fwhm_arcsec.a2000` | `filter.wiener` |
| `beammap` | `wiener_filter.template_type` | `filter.wiener` |
| `oof` | `coadd.enabled` | `map.coadd` |
| `oof` | `kids.solver.fitreportdir` | `runtime.fitreport_dir` |
| `oof` | `mapmaking.cunit` | `map.unit` |
| `oof` | `mapmaking.method` | `map.method` |
| `oof` | `mapmaking.pixel_axes` | `map.pixel_axes` |
| `oof` | `mapmaking.pixel_size_arcsec` | `map.pixel_size_arcsec` |
| `oof` | `noise_maps.enabled` | `products.noise` |
| `oof` | `noise_maps.n_noise_maps` | `products.noise_count` |
| `oof` | `noise_maps.randomize_dets` | `products.noise_randomize_dets` |
| `oof` | `post_processing.map_filtering.enabled` | `products.map_filtering` |
| `oof` | `post_processing.map_filtering.normalize_errors` | `products.map_filtering` |
| `oof` | `post_processing.map_filtering.type` | `products.map_filtering` |
| `oof` | `post_processing.map_histogram_n_bins` | `products.map_histogram_bins` |
| `oof` | `post_processing.source_finding.enabled` | `products.source_finding` |
| `oof` | `post_processing.source_finding.mode` | `products.source_finding` |
| `oof` | `post_processing.source_finding.source_sigma` | `products.source_finding` |
| `oof` | `post_processing.source_finding.source_window_arcsec` | `products.source_finding` |
| `oof` | `post_processing.source_fitting.bounding_box_arcsec` | `oof.fit_box_arcsec`, `source.fit_box_arcsec` |
| `oof` | `post_processing.source_fitting.fitting_radius_arcsec` | `oof.fit_radius_arcsec`, `source.fit_radius_arcsec` |
| `oof` | `post_processing.source_fitting.model` | `source.fit_model` |
| `oof` | `runtime.n_threads` | `runtime.threads` |
| `oof` | `runtime.output_dir` | `output.dir` |
| `oof` | `runtime.reduction_type` | `<derived>` |
| `oof` | `runtime.use_subdir` | `output.subdir` |
| `oof` | `runtime.verbose` | `output.verbose`, `products.diagnostics` |
| `oof` | `timestream.chunking.chunk_mode` | `processing.chunking` |
| `oof` | `timestream.chunking.force_chunking` | `processing.chunking` |
| `oof` | `timestream.chunking.value` | `processing.chunking` |
| `oof` | `timestream.fruit_loops.enabled` | `processing.fruitloops` |
| `oof` | `timestream.fruit_loops.max_iters` | `processing.fruitloops_iters` |
| `oof` | `timestream.fruit_loops.path` | `processing.fruitloops_source` |
| `oof` | `timestream.fruit_loops.save_all_iters` | `processing.fruitloops_save_all_iters` |
| `oof` | `timestream.fruit_loops.type` | `processing.fruitloops_type` |
| `oof` | `timestream.polarimetry.enabled` | `processing.polarimetry` |
| `oof` | `timestream.polarimetry.grouping` | `processing.polarimetry` |
| `oof` | `timestream.polarimetry.ignore_hwpr` | `processing.polarimetry` |
| `oof` | `timestream.processed_time_chunk.clean.enabled` | `processing.clean_enabled` |
| `oof` | `timestream.processed_time_chunk.clean.grouping[]` | `<derived>` |
| `oof` | `timestream.processed_time_chunk.output.enabled` | `products.tod` |
| `oof` | `timestream.processed_time_chunk.weighting.type` | `processing.weighting` |
| `oof` | `timestream.raw_time_chunk.despike.enabled` | `processing.raw` |
| `oof` | `timestream.raw_time_chunk.downsample.enabled` | `processing.raw` |
| `oof` | `timestream.raw_time_chunk.extinction_correction.enabled` | `processing.raw` |
| `oof` | `timestream.raw_time_chunk.filter.enabled` | `processing.raw` |
| `oof` | `timestream.raw_time_chunk.flux_calibration.enabled` | `processing.raw` |
| `oof` | `timestream.raw_time_chunk.output.enabled` | `processing.raw`, `products.tod` |
| `oof` | `wiener_filter.lowpass_only` | `filter.wiener` |
| `oof` | `wiener_filter.template_fwhm_arcsec.a1100` | `filter.wiener` |
| `oof` | `wiener_filter.template_fwhm_arcsec.a1400` | `filter.wiener` |
| `oof` | `wiener_filter.template_fwhm_arcsec.a2000` | `filter.wiener` |
| `oof` | `wiener_filter.template_type` | `filter.wiener` |
| `pointing` | `coadd.enabled` | `map.coadd` |
| `pointing` | `kids.solver.fitreportdir` | `runtime.fitreport_dir` |
| `pointing` | `mapmaking.cunit` | `map.unit` |
| `pointing` | `mapmaking.method` | `map.method` |
| `pointing` | `mapmaking.pixel_axes` | `map.pixel_axes` |
| `pointing` | `mapmaking.pixel_size_arcsec` | `map.pixel_size_arcsec` |
| `pointing` | `noise_maps.enabled` | `products.noise` |
| `pointing` | `noise_maps.n_noise_maps` | `products.noise_count` |
| `pointing` | `noise_maps.products.apply_empirical_weights` | `products.noise_apply_empirical_weights` |
| `pointing` | `noise_maps.products.enabled` | `products.noise_products` |
| `pointing` | `noise_maps.randomize_dets` | `products.noise_randomize_dets` |
| `pointing` | `noise_maps.write_realizations` | `products.noise_realizations` |
| `pointing` | `pointing.source_strategy.fit_gaussian` | `pointing.source_strategy` |
| `pointing` | `pointing.source_strategy.fruitloops_center_mode` | `pointing.source_strategy` |
| `pointing` | `pointing.source_strategy.header_max_radius_arcsec` | `pointing.source_strategy` |
| `pointing` | `pointing.source_strategy.header_require_coverage` | `pointing.source_strategy` |
| `pointing` | `pointing.source_strategy.mode` | `pointing.source_strategy` |
| `pointing` | `post_processing.map_filtering.enabled` | `products.map_filtering` |
| `pointing` | `post_processing.map_filtering.normalize_errors` | `products.map_filtering` |
| `pointing` | `post_processing.map_filtering.type` | `products.map_filtering` |
| `pointing` | `post_processing.map_histogram_n_bins` | `products.map_histogram_bins` |
| `pointing` | `post_processing.source_finding.enabled` | `products.source_finding` |
| `pointing` | `post_processing.source_finding.mode` | `products.source_finding` |
| `pointing` | `post_processing.source_finding.source_sigma` | `products.source_finding` |
| `pointing` | `post_processing.source_finding.source_window_arcsec` | `products.source_finding` |
| `pointing` | `post_processing.source_fitting.bounding_box_arcsec` | `pointing.fit_box_arcsec`, `source.fit_box_arcsec` |
| `pointing` | `post_processing.source_fitting.fitting_radius_arcsec` | `pointing.fit_radius_arcsec`, `source.fit_radius_arcsec` |
| `pointing` | `post_processing.source_fitting.model` | `source.fit_model` |
| `pointing` | `runtime.n_threads` | `runtime.threads` |
| `pointing` | `runtime.output_dir` | `output.dir` |
| `pointing` | `runtime.reduction_type` | `<derived>` |
| `pointing` | `runtime.use_subdir` | `output.subdir` |
| `pointing` | `runtime.verbose` | `output.verbose`, `products.diagnostics` |
| `pointing` | `source.map_regime` | `source.map_regime` |
| `pointing` | `timestream.chunking.chunk_mode` | `processing.chunking` |
| `pointing` | `timestream.chunking.force_chunking` | `processing.chunking` |
| `pointing` | `timestream.chunking.value` | `processing.chunking` |
| `pointing` | `timestream.fruit_loops.adaptive_support_radius_arcsec` | `processing.fruitloops_support_radius_arcsec` |
| `pointing` | `timestream.fruit_loops.adaptive_support_radius_fwhm` | `processing.fruitloops_support_radius_fwhm` |
| `pointing` | `timestream.fruit_loops.center_keep_radius_arcsec` | `processing.fruitloops_center_keep_radius_arcsec` |
| `pointing` | `timestream.fruit_loops.enabled` | `processing.fruitloops` |
| `pointing` | `timestream.fruit_loops.max_iters` | `processing.fruitloops_iters` |
| `pointing` | `timestream.fruit_loops.path` | `processing.fruitloops_source` |
| `pointing` | `timestream.fruit_loops.save_all_iters` | `processing.fruitloops_save_all_iters` |
| `pointing` | `timestream.fruit_loops.type` | `processing.fruitloops_type` |
| `pointing` | `timestream.learning.enabled` | `processing.learning` |
| `pointing` | `timestream.polarimetry.enabled` | `processing.polarimetry` |
| `pointing` | `timestream.polarimetry.grouping` | `processing.polarimetry` |
| `pointing` | `timestream.polarimetry.ignore_hwpr` | `processing.polarimetry` |
| `pointing` | `timestream.processed_time_chunk.clean.adaptive_selector.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.grouping[]` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.marchenko_pastur.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.null_model.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.enabled` | `processing.clean`, `processing.standard_pca` |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.n_calc` | `processing.clean`, `processing.standard_pca` |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a1100[]` | `processing.clean`, `processing.standard_pca` |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a1400[]` | `processing.clean`, `processing.standard_pca` |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a2000[]` | `processing.clean`, `processing.standard_pca` |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.stddev_limit` | `processing.clean`, `processing.standard_pca` |
| `pointing` | `timestream.processed_time_chunk.flagging.second_pass_local.enabled` | `processing.second_pass_local` |
| `pointing` | `timestream.processed_time_chunk.flagging.second_pass_local.source_protection.radius_arcsec` | `pointing.source_protection_radius_arcsec` |
| `pointing` | `timestream.processed_time_chunk.output.enabled` | `products.tod` |
| `pointing` | `timestream.processed_time_chunk.output.indices` | `products.tod_indices` |
| `pointing` | `timestream.processed_time_chunk.weighting.source_mask_radius_arcsec` | `processing.source_mask_radius_arcsec` |
| `pointing` | `timestream.processed_time_chunk.weighting.type` | `processing.weighting` |
| `pointing` | `timestream.raw_time_chunk.IIR_filter.enabled` | `processing.raw` |
| `pointing` | `timestream.raw_time_chunk.despike.enabled` | `processing.raw` |
| `pointing` | `timestream.raw_time_chunk.despike.source_protection.radius_arcsec` | `pointing.source_protection_radius_arcsec` |
| `pointing` | `timestream.raw_time_chunk.downsample.enabled` | `processing.raw` |
| `pointing` | `timestream.raw_time_chunk.extinction_correction.enabled` | `processing.raw` |
| `pointing` | `timestream.raw_time_chunk.filter.enabled` | `processing.raw` |
| `pointing` | `timestream.raw_time_chunk.flux_calibration.enabled` | `processing.raw` |
| `pointing` | `timestream.raw_time_chunk.output.enabled` | `processing.raw`, `products.tod` |
| `pointing` | `timestream.raw_time_chunk.output.indices` | `products.tod_indices` |
| `pointing` | `wiener_filter.lowpass_only` | `filter.wiener` |
| `pointing` | `wiener_filter.template_fwhm_arcsec.a1100` | `filter.wiener` |
| `pointing` | `wiener_filter.template_fwhm_arcsec.a1400` | `filter.wiener` |
| `pointing` | `wiener_filter.template_fwhm_arcsec.a2000` | `filter.wiener` |
| `pointing` | `wiener_filter.template_type` | `filter.wiener` |
| `science` | `coadd.enabled` | `map.coadd` |
| `science` | `kids.solver.fitreportdir` | `runtime.fitreport_dir` |
| `science` | `mapmaking.cunit` | `map.unit` |
| `science` | `mapmaking.method` | `map.method` |
| `science` | `mapmaking.pixel_axes` | `map.pixel_axes` |
| `science` | `mapmaking.pixel_size_arcsec` | `map.pixel_size_arcsec` |
| `science` | `noise_maps.enabled` | `products.noise` |
| `science` | `noise_maps.n_noise_maps` | `products.noise_count` |
| `science` | `noise_maps.randomize_dets` | `products.noise_randomize_dets` |
| `science` | `post_processing.map_filtering.enabled` | `products.map_filtering` |
| `science` | `post_processing.map_filtering.normalize_errors` | `products.map_filtering` |
| `science` | `post_processing.map_filtering.type` | `products.map_filtering` |
| `science` | `post_processing.map_histogram_n_bins` | `products.map_histogram_bins` |
| `science` | `post_processing.source_finding.enabled` | `products.source_finding` |
| `science` | `post_processing.source_finding.mode` | `products.source_finding` |
| `science` | `post_processing.source_finding.source_sigma` | `products.source_finding` |
| `science` | `post_processing.source_finding.source_window_arcsec` | `products.source_finding` |
| `science` | `post_processing.source_fitting.bounding_box_arcsec` | `source.fit_box_arcsec` |
| `science` | `post_processing.source_fitting.fitting_radius_arcsec` | `source.fit_radius_arcsec` |
| `science` | `post_processing.source_fitting.model` | `source.fit_model` |
| `science` | `runtime.n_threads` | `runtime.threads` |
| `science` | `runtime.output_dir` | `output.dir` |
| `science` | `runtime.reduction_type` | `<derived>` |
| `science` | `runtime.use_subdir` | `output.subdir` |
| `science` | `runtime.verbose` | `output.verbose`, `products.diagnostics` |
| `science` | `timestream.chunking.chunk_mode` | `processing.chunking` |
| `science` | `timestream.chunking.force_chunking` | `processing.chunking` |
| `science` | `timestream.chunking.value` | `processing.chunking` |
| `science` | `timestream.fruit_loops.enabled` | `processing.fruitloops` |
| `science` | `timestream.fruit_loops.max_iters` | `processing.fruitloops_iters` |
| `science` | `timestream.fruit_loops.path` | `processing.fruitloops_source` |
| `science` | `timestream.fruit_loops.save_all_iters` | `processing.fruitloops_save_all_iters` |
| `science` | `timestream.fruit_loops.type` | `processing.fruitloops_type` |
| `science` | `timestream.polarimetry.enabled` | `processing.polarimetry` |
| `science` | `timestream.polarimetry.grouping` | `processing.polarimetry` |
| `science` | `timestream.polarimetry.ignore_hwpr` | `processing.polarimetry` |
| `science` | `timestream.processed_time_chunk.clean.enabled` | `processing.clean_enabled` |
| `science` | `timestream.processed_time_chunk.clean.grouping[]` | `<derived>` |
| `science` | `timestream.processed_time_chunk.output.enabled` | `products.tod` |
| `science` | `timestream.processed_time_chunk.output.indices[]` | `<derived>` |
| `science` | `timestream.processed_time_chunk.weighting.type` | `processing.weighting` |
| `science` | `timestream.raw_time_chunk.IIR_filter.enabled` | `processing.raw` |
| `science` | `timestream.raw_time_chunk.despike.enabled` | `processing.raw` |
| `science` | `timestream.raw_time_chunk.downsample.enabled` | `processing.raw` |
| `science` | `timestream.raw_time_chunk.extinction_correction.enabled` | `processing.raw` |
| `science` | `timestream.raw_time_chunk.filter.enabled` | `processing.raw` |
| `science` | `timestream.raw_time_chunk.flux_calibration.enabled` | `processing.raw` |
| `science` | `timestream.raw_time_chunk.output.enabled` | `processing.raw`, `products.tod` |
| `science` | `timestream.raw_time_chunk.output.indices[]` | `<derived>` |
| `science` | `wiener_filter.lowpass_only` | `filter.wiener` |
| `science` | `wiener_filter.template_fwhm_arcsec.a1100` | `filter.wiener` |
| `science` | `wiener_filter.template_fwhm_arcsec.a1400` | `filter.wiener` |
| `science` | `wiener_filter.template_fwhm_arcsec.a2000` | `filter.wiener` |
| `science` | `wiener_filter.template_type` | `filter.wiener` |
