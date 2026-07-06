# Compact Surface Coverage Audit

This audit checks whether low-level keys classified as `user-facing` are
represented by current compact-config fields for the representative
pointing, OOF, beammap, and science baselines. It is a shadow/config
tooling check only; it does not change Citlali runtime parsing.

## Summary

- Configs checked: 4
- Unique mode/path user-facing keys: 294
- Covered by compact fields: 105
- Gaps: 189
- Coverage: 35.7%

`Covered` means the current compact fields expand back to that low-level
path without using `expert:`. `Gap` means the value is still preserved
only through the expert escape hatch. Some gaps are expected policy
questions because the provisional classification includes inactive
defaults and conditional product families that may stay profile-owned.

| Mode | User-Facing Paths | Covered | Gaps | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `beammap` | 85 | 33 | 52 | 38.8% |
| `oof` | 56 | 20 | 36 | 35.7% |
| `pointing` | 92 | 34 | 58 | 37.0% |
| `science` | 61 | 18 | 43 | 29.5% |

## Gaps

| Mode | Low-Level Path | Rule | Reason |
| --- | --- | --- | --- |
| `beammap` | `beammap.rfi_mask.enabled` | `user-beammap-rfi-mask-toggle` | High-level beammap RFI burst-mask toggle. |
| `beammap` | `beammap.scan_band_mask.enabled` | `user-beammap-scan-band-mask-toggle` | High-level coherent scan-band mask toggle. |
| `beammap` | `beammap.split_fits_by_flag.enabled` | `user-beammap-split-fits-toggle` | Optional split beammap FITS product toggle. |
| `beammap` | `kids.solver.fitreportdir` | `user-kids-fitreportdir` | Operational path that may need authoring control. |
| `beammap` | `noise_maps.randomize_dets` | `expert-noise-maps` | Remaining noise-map fields are normal noise-product authoring choices. |
| `beammap` | `post_processing.map_filtering.enabled` | `user-map-filtering-core` | High-level filtered-map product switch. |
| `beammap` | `post_processing.map_filtering.normalize_errors` | `user-map-filtering-normalize-errors` | Filtered-map error policy. |
| `beammap` | `post_processing.map_filtering.type` | `user-map-filtering-type` | Filter algorithm choice. |
| `beammap` | `post_processing.map_histogram_n_bins` | `user-map-histogram` | Product-summary choice. |
| `beammap` | `post_processing.source_finding.enabled` | `user-source-finding` | Optional source-catalog behavior. |
| `beammap` | `post_processing.source_finding.mode` | `user-source-finding` | Optional source-catalog behavior. |
| `beammap` | `post_processing.source_finding.source_sigma` | `user-source-finding` | Optional source-catalog behavior. |
| `beammap` | `post_processing.source_finding.source_window_arcsec` | `user-source-finding` | Optional source-catalog behavior. |
| `beammap` | `post_processing.source_fitting.bounding_box_arcsec` | `user-source-fitting-core` | Point-source fit support choice. |
| `beammap` | `post_processing.source_fitting.fitting_radius_arcsec` | `user-source-fitting-radius` | Point-source fit support choice. |
| `beammap` | `post_processing.source_fitting.model` | `user-source-fitting-model` | Source-fit model choice. |
| `beammap` | `source.map_regime` | `user-source-map-regime` | High-level source-context metadata for diagnostic interpretation. |
| `beammap` | `timestream.chunking.chunk_mode` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `beammap` | `timestream.chunking.force_chunking` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `beammap` | `timestream.chunking.value` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `beammap` | `timestream.fruit_loops.adaptive_support_radius_arcsec` | `user-fruitloops-oof-support` | OOF/pointing support-radius policy. |
| `beammap` | `timestream.fruit_loops.adaptive_support_radius_fwhm` | `user-fruitloops-oof-support` | OOF/pointing support-radius policy. |
| `beammap` | `timestream.fruit_loops.center_keep_radius_arcsec` | `user-fruitloops-center-keep` | OOF/pointing central-region policy. |
| `beammap` | `timestream.fruit_loops.path` | `user-fruitloops-source` | Operational map-template source choice. |
| `beammap` | `timestream.fruit_loops.save_all_iters` | `user-fruitloops-products` | Product-volume choice. |
| `beammap` | `timestream.fruit_loops.type` | `user-fruitloops-type` | Operational map-template source choice. |
| `beammap` | `timestream.polarimetry.enabled` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `beammap` | `timestream.polarimetry.grouping` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `beammap` | `timestream.polarimetry.ignore_hwpr` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.n_calc` | `user-standard-pca-ncalc` | Advanced standard-PCA runtime/accuracy tuning. |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a1100[]` | `user-standard-pca-depth` | Common standard-PCA tuning for current cleaner schema. |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a1400[]` | `user-standard-pca-depth` | Common standard-PCA tuning for current cleaner schema. |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a2000[]` | `user-standard-pca-depth` | Common standard-PCA tuning for current cleaner schema. |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.stddev_limit` | `user-standard-pca-stddev` | Common standard-PCA tuning for current cleaner schema. |
| `beammap` | `timestream.processed_time_chunk.flagging.second_pass_local.enabled` | `user-second-pass-local-toggle` | High-level post-clean deglitch toggle. |
| `beammap` | `timestream.raw_time_chunk.IIR_filter.enabled` | `user-raw-iir-toggle` | High-level IIR filter toggle. |
| `beammap` | `timestream.raw_time_chunk.despike.enabled` | `user-raw-despike-toggle` | High-level raw despiking toggle. |
| `beammap` | `timestream.raw_time_chunk.downsample.enabled` | `user-raw-downsample-toggle` | High-level downsampling toggle. |
| `beammap` | `timestream.raw_time_chunk.extinction_correction.enabled` | `user-extinction-correction-toggle` | Calibration product policy. |
| `beammap` | `timestream.raw_time_chunk.filter.enabled` | `user-raw-filter-toggle` | High-level raw TOD filtering toggle. |
| `beammap` | `timestream.raw_time_chunk.flux_calibration.enabled` | `user-flux-calibration-toggle` | Calibration product policy. |
| `beammap` | `timestream.raw_time_chunk.line_audit.enabled` | `user-raw-line-audit-toggle` | High-level diagnostic line-audit mode toggle. |
| `beammap` | `wiener_filter.denom_check_iters` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `beammap` | `wiener_filter.denom_rel_tol` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `beammap` | `wiener_filter.lowpass_only` | `user-wiener-lowpass` | Filter behavior choice. |
| `beammap` | `wiener_filter.max_denom_iters` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `beammap` | `wiener_filter.max_loops` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `beammap` | `wiener_filter.tail_frac_tol` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `beammap` | `wiener_filter.template_fwhm_arcsec.a1100` | `user-wiener-template` | Filter template shape choice. |
| `beammap` | `wiener_filter.template_fwhm_arcsec.a1400` | `user-wiener-template` | Filter template shape choice. |
| `beammap` | `wiener_filter.template_fwhm_arcsec.a2000` | `user-wiener-template` | Filter template shape choice. |
| `beammap` | `wiener_filter.template_type` | `user-wiener-template` | Filter template shape choice. |
| `oof` | `beammap.derotate` | `user-beammap-derotate` | Common beammap product choice. |
| `oof` | `beammap.iter_max` | `user-beammap-main` | Main beammap iteration controls. |
| `oof` | `beammap.iter_tolerance` | `user-beammap-main` | Main beammap iteration controls. |
| `oof` | `beammap.reference_det` | `user-beammap-reference-det` | Common beammap operational policy. |
| `oof` | `beammap.subtract_reference_det` | `user-beammap-reference-subtraction` | Common beammap operational policy. |
| `oof` | `kids.solver.fitreportdir` | `user-kids-fitreportdir` | Operational path that may need authoring control. |
| `oof` | `noise_maps.randomize_dets` | `expert-noise-maps` | Remaining noise-map fields are normal noise-product authoring choices. |
| `oof` | `post_processing.map_filtering.enabled` | `user-map-filtering-core` | High-level filtered-map product switch. |
| `oof` | `post_processing.map_filtering.normalize_errors` | `user-map-filtering-normalize-errors` | Filtered-map error policy. |
| `oof` | `post_processing.map_filtering.type` | `user-map-filtering-type` | Filter algorithm choice. |
| `oof` | `post_processing.map_histogram_n_bins` | `user-map-histogram` | Product-summary choice. |
| `oof` | `post_processing.source_finding.enabled` | `user-source-finding` | Optional source-catalog behavior. |
| `oof` | `post_processing.source_finding.mode` | `user-source-finding` | Optional source-catalog behavior. |
| `oof` | `post_processing.source_finding.source_sigma` | `user-source-finding` | Optional source-catalog behavior. |
| `oof` | `post_processing.source_finding.source_window_arcsec` | `user-source-finding` | Optional source-catalog behavior. |
| `oof` | `post_processing.source_fitting.model` | `user-source-fitting-model` | Source-fit model choice. |
| `oof` | `timestream.chunking.chunk_mode` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `oof` | `timestream.chunking.force_chunking` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `oof` | `timestream.chunking.value` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `oof` | `timestream.fruit_loops.path` | `user-fruitloops-source` | Operational map-template source choice. |
| `oof` | `timestream.fruit_loops.save_all_iters` | `user-fruitloops-products` | Product-volume choice. |
| `oof` | `timestream.fruit_loops.type` | `user-fruitloops-type` | Operational map-template source choice. |
| `oof` | `timestream.polarimetry.enabled` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `oof` | `timestream.polarimetry.grouping` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `oof` | `timestream.polarimetry.ignore_hwpr` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `oof` | `timestream.processed_time_chunk.clean.enabled` | `user-ptc-clean-enabled` | Main PTC cleaning switch. |
| `oof` | `timestream.raw_time_chunk.despike.enabled` | `user-raw-despike-toggle` | High-level raw despiking toggle. |
| `oof` | `timestream.raw_time_chunk.downsample.enabled` | `user-raw-downsample-toggle` | High-level downsampling toggle. |
| `oof` | `timestream.raw_time_chunk.extinction_correction.enabled` | `user-extinction-correction-toggle` | Calibration product policy. |
| `oof` | `timestream.raw_time_chunk.filter.enabled` | `user-raw-filter-toggle` | High-level raw TOD filtering toggle. |
| `oof` | `timestream.raw_time_chunk.flux_calibration.enabled` | `user-flux-calibration-toggle` | Calibration product policy. |
| `oof` | `wiener_filter.lowpass_only` | `user-wiener-lowpass` | Filter behavior choice. |
| `oof` | `wiener_filter.template_fwhm_arcsec.a1100` | `user-wiener-template` | Filter template shape choice. |
| `oof` | `wiener_filter.template_fwhm_arcsec.a1400` | `user-wiener-template` | Filter template shape choice. |
| `oof` | `wiener_filter.template_fwhm_arcsec.a2000` | `user-wiener-template` | Filter template shape choice. |
| `oof` | `wiener_filter.template_type` | `user-wiener-template` | Filter template shape choice. |
| `pointing` | `beammap.convergence_radius_arcsec` | `user-beammap-convergence-radius` | Main beammap convergence aperture. |
| `pointing` | `beammap.derotate` | `user-beammap-derotate` | Common beammap product choice. |
| `pointing` | `beammap.detector_weighting.mode` | `user-beammap-detector-weighting` | Common detector-beammap weighting choice. |
| `pointing` | `beammap.iter_max` | `user-beammap-main` | Main beammap iteration controls. |
| `pointing` | `beammap.iter_tolerance` | `user-beammap-main` | Main beammap iteration controls. |
| `pointing` | `beammap.reference_det` | `user-beammap-reference-det` | Common beammap operational policy. |
| `pointing` | `beammap.subtract_reference_det` | `user-beammap-reference-subtraction` | Common beammap operational policy. |
| `pointing` | `kids.solver.fitreportdir` | `user-kids-fitreportdir` | Operational path that may need authoring control. |
| `pointing` | `noise_maps.products.apply_empirical_weights` | `expert-noise-maps` | Remaining noise-map fields are normal noise-product authoring choices. |
| `pointing` | `noise_maps.randomize_dets` | `expert-noise-maps` | Remaining noise-map fields are normal noise-product authoring choices. |
| `pointing` | `post_processing.map_filtering.enabled` | `user-map-filtering-core` | High-level filtered-map product switch. |
| `pointing` | `post_processing.map_filtering.normalize_errors` | `user-map-filtering-normalize-errors` | Filtered-map error policy. |
| `pointing` | `post_processing.map_filtering.type` | `user-map-filtering-type` | Filter algorithm choice. |
| `pointing` | `post_processing.map_histogram_n_bins` | `user-map-histogram` | Product-summary choice. |
| `pointing` | `post_processing.source_finding.enabled` | `user-source-finding` | Optional source-catalog behavior. |
| `pointing` | `post_processing.source_finding.mode` | `user-source-finding` | Optional source-catalog behavior. |
| `pointing` | `post_processing.source_finding.source_sigma` | `user-source-finding` | Optional source-catalog behavior. |
| `pointing` | `post_processing.source_finding.source_window_arcsec` | `user-source-finding` | Optional source-catalog behavior. |
| `pointing` | `post_processing.source_fitting.model` | `user-source-fitting-model` | Source-fit model choice. |
| `pointing` | `source.map_regime` | `user-source-map-regime` | High-level source-context metadata for diagnostic interpretation. |
| `pointing` | `timestream.chunking.chunk_mode` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `pointing` | `timestream.chunking.force_chunking` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `pointing` | `timestream.chunking.value` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `pointing` | `timestream.fruit_loops.adaptive_support_radius_arcsec` | `user-fruitloops-oof-support` | OOF/pointing support-radius policy. |
| `pointing` | `timestream.fruit_loops.adaptive_support_radius_fwhm` | `user-fruitloops-oof-support` | OOF/pointing support-radius policy. |
| `pointing` | `timestream.fruit_loops.center_keep_radius_arcsec` | `user-fruitloops-center-keep` | OOF/pointing central-region policy. |
| `pointing` | `timestream.fruit_loops.path` | `user-fruitloops-source` | Operational map-template source choice. |
| `pointing` | `timestream.fruit_loops.save_all_iters` | `user-fruitloops-products` | Product-volume choice. |
| `pointing` | `timestream.fruit_loops.type` | `user-fruitloops-type` | Operational map-template source choice. |
| `pointing` | `timestream.learning.enabled` | `user-learning-enabled` | High-level learning-state feature switch. |
| `pointing` | `timestream.polarimetry.enabled` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `pointing` | `timestream.polarimetry.grouping` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `pointing` | `timestream.polarimetry.ignore_hwpr` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.n_calc` | `user-standard-pca-ncalc` | Advanced standard-PCA runtime/accuracy tuning. |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a1100[]` | `user-standard-pca-depth` | Common standard-PCA tuning for current cleaner schema. |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a1400[]` | `user-standard-pca-depth` | Common standard-PCA tuning for current cleaner schema. |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.n_eig_to_cut.a2000[]` | `user-standard-pca-depth` | Common standard-PCA tuning for current cleaner schema. |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.stddev_limit` | `user-standard-pca-stddev` | Common standard-PCA tuning for current cleaner schema. |
| `pointing` | `timestream.processed_time_chunk.flagging.second_pass_local.enabled` | `user-second-pass-local-toggle` | High-level post-clean deglitch toggle. |
| `pointing` | `timestream.processed_time_chunk.output.indices` | `user-tod-output-selection` | TOD sidecar selection is a product-volume choice. |
| `pointing` | `timestream.processed_time_chunk.weighting.source_mask_radius_arcsec` | `user-ptc-source-mask` | Important source-aware weighting control. |
| `pointing` | `timestream.raw_time_chunk.IIR_filter.enabled` | `user-raw-iir-toggle` | High-level IIR filter toggle. |
| `pointing` | `timestream.raw_time_chunk.despike.enabled` | `user-raw-despike-toggle` | High-level raw despiking toggle. |
| `pointing` | `timestream.raw_time_chunk.downsample.enabled` | `user-raw-downsample-toggle` | High-level downsampling toggle. |
| `pointing` | `timestream.raw_time_chunk.extinction_correction.enabled` | `user-extinction-correction-toggle` | Calibration product policy. |
| `pointing` | `timestream.raw_time_chunk.filter.enabled` | `user-raw-filter-toggle` | High-level raw TOD filtering toggle. |
| `pointing` | `timestream.raw_time_chunk.flux_calibration.enabled` | `user-flux-calibration-toggle` | Calibration product policy. |
| `pointing` | `timestream.raw_time_chunk.output.indices` | `user-tod-output-selection` | TOD sidecar selection is a product-volume choice. |
| `pointing` | `wiener_filter.denom_check_iters` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `pointing` | `wiener_filter.denom_rel_tol` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `pointing` | `wiener_filter.lowpass_only` | `user-wiener-lowpass` | Filter behavior choice. |
| `pointing` | `wiener_filter.max_denom_iters` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `pointing` | `wiener_filter.max_loops` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `pointing` | `wiener_filter.tail_frac_tol` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `pointing` | `wiener_filter.template_fwhm_arcsec.a1100` | `user-wiener-template` | Filter template shape choice. |
| `pointing` | `wiener_filter.template_fwhm_arcsec.a1400` | `user-wiener-template` | Filter template shape choice. |
| `pointing` | `wiener_filter.template_fwhm_arcsec.a2000` | `user-wiener-template` | Filter template shape choice. |
| `pointing` | `wiener_filter.template_type` | `user-wiener-template` | Filter template shape choice. |
| `science` | `beammap.derotate` | `user-beammap-derotate` | Common beammap product choice. |
| `science` | `beammap.iter_max` | `user-beammap-main` | Main beammap iteration controls. |
| `science` | `beammap.iter_tolerance` | `user-beammap-main` | Main beammap iteration controls. |
| `science` | `beammap.reference_det` | `user-beammap-reference-det` | Common beammap operational policy. |
| `science` | `beammap.subtract_reference_det` | `user-beammap-reference-subtraction` | Common beammap operational policy. |
| `science` | `kids.solver.fitreportdir` | `user-kids-fitreportdir` | Operational path that may need authoring control. |
| `science` | `noise_maps.randomize_dets` | `expert-noise-maps` | Remaining noise-map fields are normal noise-product authoring choices. |
| `science` | `post_processing.map_filtering.enabled` | `user-map-filtering-core` | High-level filtered-map product switch. |
| `science` | `post_processing.map_filtering.normalize_errors` | `user-map-filtering-normalize-errors` | Filtered-map error policy. |
| `science` | `post_processing.map_filtering.type` | `user-map-filtering-type` | Filter algorithm choice. |
| `science` | `post_processing.map_histogram_n_bins` | `user-map-histogram` | Product-summary choice. |
| `science` | `post_processing.source_finding.enabled` | `user-source-finding` | Optional source-catalog behavior. |
| `science` | `post_processing.source_finding.mode` | `user-source-finding` | Optional source-catalog behavior. |
| `science` | `post_processing.source_finding.source_sigma` | `user-source-finding` | Optional source-catalog behavior. |
| `science` | `post_processing.source_finding.source_window_arcsec` | `user-source-finding` | Optional source-catalog behavior. |
| `science` | `post_processing.source_fitting.bounding_box_arcsec` | `user-source-fitting-core` | Point-source fit support choice. |
| `science` | `post_processing.source_fitting.fitting_radius_arcsec` | `user-source-fitting-radius` | Point-source fit support choice. |
| `science` | `post_processing.source_fitting.model` | `user-source-fitting-model` | Source-fit model choice. |
| `science` | `timestream.chunking.chunk_mode` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `science` | `timestream.chunking.force_chunking` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `science` | `timestream.chunking.value` | `user-timestream-chunking` | Advanced but legitimate performance/scan subdivision choice. |
| `science` | `timestream.fruit_loops.path` | `user-fruitloops-source` | Operational map-template source choice. |
| `science` | `timestream.fruit_loops.save_all_iters` | `user-fruitloops-products` | Product-volume choice. |
| `science` | `timestream.fruit_loops.type` | `user-fruitloops-type` | Operational map-template source choice. |
| `science` | `timestream.polarimetry.enabled` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `science` | `timestream.polarimetry.grouping` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `science` | `timestream.polarimetry.ignore_hwpr` | `user-polarimetry` | User-facing only for polarimetry profiles. |
| `science` | `timestream.processed_time_chunk.clean.enabled` | `user-ptc-clean-enabled` | Main PTC cleaning switch. |
| `science` | `timestream.processed_time_chunk.output.indices[]` | `user-tod-output-selection` | TOD sidecar selection is a product-volume choice. |
| `science` | `timestream.raw_time_chunk.IIR_filter.enabled` | `user-raw-iir-toggle` | High-level IIR filter toggle. |
| `science` | `timestream.raw_time_chunk.despike.enabled` | `user-raw-despike-toggle` | High-level raw despiking toggle. |
| `science` | `timestream.raw_time_chunk.downsample.enabled` | `user-raw-downsample-toggle` | High-level downsampling toggle. |
| `science` | `timestream.raw_time_chunk.extinction_correction.enabled` | `user-extinction-correction-toggle` | Calibration product policy. |
| `science` | `timestream.raw_time_chunk.filter.enabled` | `user-raw-filter-toggle` | High-level raw TOD filtering toggle. |
| `science` | `timestream.raw_time_chunk.flux_calibration.enabled` | `user-flux-calibration-toggle` | Calibration product policy. |
| `science` | `timestream.raw_time_chunk.output.indices[]` | `user-tod-output-selection` | TOD sidecar selection is a product-volume choice. |
| `science` | `wiener_filter.denom_rel_tol` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `science` | `wiener_filter.lowpass_only` | `user-wiener-lowpass` | Filter behavior choice. |
| `science` | `wiener_filter.tail_frac_tol` | `expert-wiener` | Remaining Wiener-filter fields are normal filter authoring choices. |
| `science` | `wiener_filter.template_fwhm_arcsec.a1100` | `user-wiener-template` | Filter template shape choice. |
| `science` | `wiener_filter.template_fwhm_arcsec.a1400` | `user-wiener-template` | Filter template shape choice. |
| `science` | `wiener_filter.template_fwhm_arcsec.a2000` | `user-wiener-template` | Filter template shape choice. |
| `science` | `wiener_filter.template_type` | `user-wiener-template` | Filter template shape choice. |

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
| `beammap` | `beammap.subtract_reference_det` | `beammap.subtract_reference_det` |
| `beammap` | `coadd.enabled` | `map.coadd` |
| `beammap` | `mapmaking.cunit` | `map.unit` |
| `beammap` | `mapmaking.method` | `map.method` |
| `beammap` | `mapmaking.pixel_axes` | `map.pixel_axes` |
| `beammap` | `mapmaking.pixel_size_arcsec` | `map.pixel_size_arcsec` |
| `beammap` | `noise_maps.enabled` | `products.noise` |
| `beammap` | `noise_maps.n_noise_maps` | `products.noise_count` |
| `beammap` | `runtime.n_threads` | `runtime.threads` |
| `beammap` | `runtime.output_dir` | `output.dir` |
| `beammap` | `runtime.reduction_type` | `<derived>` |
| `beammap` | `runtime.use_subdir` | `output.subdir` |
| `beammap` | `runtime.verbose` | `output.verbose`, `products.diagnostics` |
| `beammap` | `timestream.fruit_loops.enabled` | `processing.fruitloops` |
| `beammap` | `timestream.fruit_loops.max_iters` | `processing.fruitloops_iters` |
| `beammap` | `timestream.processed_time_chunk.clean.adaptive_selector.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.grouping[]` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.marchenko_pastur.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.null_model.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.clean.standard_pca.enabled` | `processing.clean` |
| `beammap` | `timestream.processed_time_chunk.output.enabled` | `products.tod` |
| `beammap` | `timestream.processed_time_chunk.weighting.type` | `processing.weighting` |
| `beammap` | `timestream.raw_time_chunk.output.enabled` | `products.tod` |
| `oof` | `coadd.enabled` | `map.coadd` |
| `oof` | `mapmaking.cunit` | `map.unit` |
| `oof` | `mapmaking.method` | `map.method` |
| `oof` | `mapmaking.pixel_axes` | `map.pixel_axes` |
| `oof` | `mapmaking.pixel_size_arcsec` | `map.pixel_size_arcsec` |
| `oof` | `noise_maps.enabled` | `products.noise` |
| `oof` | `noise_maps.n_noise_maps` | `products.noise_count` |
| `oof` | `post_processing.source_fitting.bounding_box_arcsec` | `oof.fit_box_arcsec` |
| `oof` | `post_processing.source_fitting.fitting_radius_arcsec` | `oof.fit_radius_arcsec` |
| `oof` | `runtime.n_threads` | `runtime.threads` |
| `oof` | `runtime.output_dir` | `output.dir` |
| `oof` | `runtime.reduction_type` | `<derived>` |
| `oof` | `runtime.use_subdir` | `output.subdir` |
| `oof` | `runtime.verbose` | `output.verbose`, `products.diagnostics` |
| `oof` | `timestream.fruit_loops.enabled` | `processing.fruitloops` |
| `oof` | `timestream.fruit_loops.max_iters` | `processing.fruitloops_iters` |
| `oof` | `timestream.processed_time_chunk.clean.grouping[]` | `<derived>` |
| `oof` | `timestream.processed_time_chunk.output.enabled` | `products.tod` |
| `oof` | `timestream.processed_time_chunk.weighting.type` | `processing.weighting` |
| `oof` | `timestream.raw_time_chunk.output.enabled` | `products.tod` |
| `pointing` | `coadd.enabled` | `map.coadd` |
| `pointing` | `mapmaking.cunit` | `map.unit` |
| `pointing` | `mapmaking.method` | `map.method` |
| `pointing` | `mapmaking.pixel_axes` | `map.pixel_axes` |
| `pointing` | `mapmaking.pixel_size_arcsec` | `map.pixel_size_arcsec` |
| `pointing` | `noise_maps.enabled` | `products.noise` |
| `pointing` | `noise_maps.n_noise_maps` | `products.noise_count` |
| `pointing` | `noise_maps.products.enabled` | `products.noise_products` |
| `pointing` | `noise_maps.write_realizations` | `products.noise_realizations` |
| `pointing` | `pointing.source_strategy.fit_gaussian` | `pointing.source_strategy` |
| `pointing` | `pointing.source_strategy.fruitloops_center_mode` | `pointing.source_strategy` |
| `pointing` | `pointing.source_strategy.header_max_radius_arcsec` | `pointing.source_strategy` |
| `pointing` | `pointing.source_strategy.header_require_coverage` | `pointing.source_strategy` |
| `pointing` | `pointing.source_strategy.mode` | `pointing.source_strategy` |
| `pointing` | `post_processing.source_fitting.bounding_box_arcsec` | `pointing.fit_box_arcsec` |
| `pointing` | `post_processing.source_fitting.fitting_radius_arcsec` | `pointing.fit_radius_arcsec` |
| `pointing` | `runtime.n_threads` | `runtime.threads` |
| `pointing` | `runtime.output_dir` | `output.dir` |
| `pointing` | `runtime.reduction_type` | `<derived>` |
| `pointing` | `runtime.use_subdir` | `output.subdir` |
| `pointing` | `runtime.verbose` | `output.verbose`, `products.diagnostics` |
| `pointing` | `timestream.fruit_loops.enabled` | `processing.fruitloops` |
| `pointing` | `timestream.fruit_loops.max_iters` | `processing.fruitloops_iters` |
| `pointing` | `timestream.processed_time_chunk.clean.adaptive_selector.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.grouping[]` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.marchenko_pastur.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.null_model.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.clean.standard_pca.enabled` | `processing.clean` |
| `pointing` | `timestream.processed_time_chunk.flagging.second_pass_local.source_protection.radius_arcsec` | `pointing.source_protection_radius_arcsec` |
| `pointing` | `timestream.processed_time_chunk.output.enabled` | `products.tod` |
| `pointing` | `timestream.processed_time_chunk.weighting.type` | `processing.weighting` |
| `pointing` | `timestream.raw_time_chunk.despike.source_protection.radius_arcsec` | `pointing.source_protection_radius_arcsec` |
| `pointing` | `timestream.raw_time_chunk.output.enabled` | `products.tod` |
| `science` | `coadd.enabled` | `map.coadd` |
| `science` | `mapmaking.cunit` | `map.unit` |
| `science` | `mapmaking.method` | `map.method` |
| `science` | `mapmaking.pixel_axes` | `map.pixel_axes` |
| `science` | `mapmaking.pixel_size_arcsec` | `map.pixel_size_arcsec` |
| `science` | `noise_maps.enabled` | `products.noise` |
| `science` | `noise_maps.n_noise_maps` | `products.noise_count` |
| `science` | `runtime.n_threads` | `runtime.threads` |
| `science` | `runtime.output_dir` | `output.dir` |
| `science` | `runtime.reduction_type` | `<derived>` |
| `science` | `runtime.use_subdir` | `output.subdir` |
| `science` | `runtime.verbose` | `output.verbose`, `products.diagnostics` |
| `science` | `timestream.fruit_loops.enabled` | `processing.fruitloops` |
| `science` | `timestream.fruit_loops.max_iters` | `processing.fruitloops_iters` |
| `science` | `timestream.processed_time_chunk.clean.grouping[]` | `<derived>` |
| `science` | `timestream.processed_time_chunk.output.enabled` | `products.tod` |
| `science` | `timestream.processed_time_chunk.weighting.type` | `processing.weighting` |
| `science` | `timestream.raw_time_chunk.output.enabled` | `products.tod` |
