#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

template <class Config, class MissingKeys, class InvalidKeys>
citlali::config::BeammapDetectorTodOutputConfig
read_beammap_detector_tod_output_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    citlali::config::BeammapDetectorTodOutputConfig values;
    read_optional_beammap_config_value(
        config, values.enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "enabled"});
    read_optional_beammap_config_value(
        config, values.subdir_name, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "subdir_name"});
    read_optional_beammap_config_value(
        config, values.n_uniform, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "n_uniform"}, {}, {0});
    read_optional_beammap_config_value(
        config, values.n_source_dense, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "n_source_dense"},
        {}, {0});
    return values;
}

inline int default_beammap_tod_output_iter() {
    return -1;
}

inline void reset_beammap_config_mirror(
    citlali::config::BeammapConfig &target) {
    target = citlali::config::BeammapConfig{};
}

template <class BeammapControls>
void sync_beammap_core_controls(BeammapControls &controls,
                                const BeammapCoreConfigValues &values) {
    controls.beammap_iter_max = values.iteration.max_iterations;
    controls.beammap_iter_tolerance = values.iteration.tolerance;
    controls.beammap_convergence_radius_arcsec =
        values.iteration.convergence_radius_arcsec;
    controls.beammap_phase_split_enabled = values.phase_strategy.enabled;
    controls.beammap_locator_iter = values.phase_strategy.locator_iter;
    controls.beammap_measurement_start_iter =
        values.phase_strategy.measurement_start_iter;
    controls.beammap_subtract_reference =
        values.reference.subtract_reference_detector;
    controls.beammap_reference_det =
        static_cast<std::decay_t<decltype(controls.beammap_reference_det)>>(
            values.reference.reference_detector);
    controls.beammap_derotate = values.reference.derotate;
    controls.beammap_rfi_mask_enabled = values.rfi_mask.enabled;
    controls.beammap_rfi_mask_block_size_samples =
        values.rfi_mask.block_size_samples;
    controls.beammap_rfi_mask_min_good_samples =
        values.rfi_mask.min_good_samples;
    controls.beammap_rfi_mask_dilate_blocks =
        values.rfi_mask.dilate_blocks;
    controls.beammap_rfi_mask_sigma_threshold =
        values.rfi_mask.sigma_threshold;
    controls.beammap_rfi_mask_sigma_floor = values.rfi_mask.sigma_floor;
    controls.beammap_rfi_mask_max_flagged_fraction =
        values.rfi_mask.max_flagged_fraction;
}

template <class BeammapControls, class MapFitter>
void sync_beammap_map_controls(
    BeammapControls &controls, const BeammapFittingConfigValues &fitting_values,
    const citlali::config::BeammapScanBandMaskConfig &scan_band_mask,
    const citlali::config::BeammapSplitFitsByFlagConfig &split_fits_by_flag,
    MapFitter &map_fitter) {
    controls.beammap_detector_weighting_mode =
        std::string(citlali::config::to_string(
            fitting_values.detector_weighting_mode));
    controls.beammap_fit_radius_fwhm =
        fitting_values.fitting.fit_radius_fwhm;
    map_fitter.beammap_fit_radius_fwhm =
        fitting_values.fitting.fit_radius_fwhm;
    controls.beammap_scan_band_mask_enabled = scan_band_mask.enabled;
    controls.beammap_scan_band_mask_edge_rows = scan_band_mask.edge_rows;
    controls.beammap_scan_band_mask_min_row_pixels =
        scan_band_mask.min_row_pixels;
    controls.beammap_scan_band_mask_min_contiguous_rows =
        scan_band_mask.min_contiguous_rows;
    controls.beammap_scan_band_mask_row_median_sigma_threshold =
        scan_band_mask.row_median_sigma_threshold;
    controls.beammap_scan_band_mask_row_sigma_ratio_threshold =
        scan_band_mask.row_sigma_ratio_threshold;
    controls.beammap_scan_band_mask_max_flagged_fraction =
        scan_band_mask.max_flagged_fraction;
    controls.beammap_split_fits_by_flag = split_fits_by_flag.enabled;
    controls.beammap_split_flag_values = split_fits_by_flag.flag_values;
}

template <class BeammapControls>
void sync_beammap_detector_tod_output_controls(
    BeammapControls &controls,
    const citlali::config::BeammapDetectorTodOutputConfig &values) {
    controls.beammap_detector_tod_output_enabled = values.enabled;
    controls.beammap_detector_tod_output_subdir_name = values.subdir_name;
    controls.beammap_detector_tod_output_n_uniform = values.n_uniform;
    controls.beammap_detector_tod_output_n_source_dense =
        values.n_source_dense;
}

template <class BeammapControls>
void mirror_beammap_core_config(citlali::config::BeammapConfig &target,
                                const BeammapControls &controls) {
    target.iteration.max_iterations = controls.beammap_iter_max;
    target.iteration.tolerance = controls.beammap_iter_tolerance;
    target.iteration.convergence_radius_arcsec =
        controls.beammap_convergence_radius_arcsec;
    target.phase_strategy.enabled = controls.beammap_phase_split_enabled;
    target.phase_strategy.locator_iter = controls.beammap_locator_iter;
    target.phase_strategy.measurement_start_iter =
        controls.beammap_measurement_start_iter;
    target.reference.subtract_reference_detector =
        controls.beammap_subtract_reference;
    target.reference.reference_detector =
        static_cast<long>(controls.beammap_reference_det);
    target.reference.derotate = controls.beammap_derotate;
    target.rfi_mask.enabled = controls.beammap_rfi_mask_enabled;
    target.rfi_mask.block_size_samples =
        controls.beammap_rfi_mask_block_size_samples;
    target.rfi_mask.min_good_samples =
        controls.beammap_rfi_mask_min_good_samples;
    target.rfi_mask.dilate_blocks = controls.beammap_rfi_mask_dilate_blocks;
    target.rfi_mask.sigma_threshold =
        controls.beammap_rfi_mask_sigma_threshold;
    target.rfi_mask.sigma_floor = controls.beammap_rfi_mask_sigma_floor;
    target.rfi_mask.max_flagged_fraction =
        controls.beammap_rfi_mask_max_flagged_fraction;
    if (auto parsed = citlali::config::parse_beammap_detector_weighting_mode(
            controls.beammap_detector_weighting_mode)) {
        target.detector_weighting_mode = *parsed;
    }
    target.fitting.fit_radius_fwhm = controls.beammap_fit_radius_fwhm;
    target.scan_band_mask.enabled = controls.beammap_scan_band_mask_enabled;
    target.scan_band_mask.edge_rows = controls.beammap_scan_band_mask_edge_rows;
    target.scan_band_mask.min_row_pixels =
        controls.beammap_scan_band_mask_min_row_pixels;
    target.scan_band_mask.min_contiguous_rows =
        controls.beammap_scan_band_mask_min_contiguous_rows;
    target.scan_band_mask.row_median_sigma_threshold =
        controls.beammap_scan_band_mask_row_median_sigma_threshold;
    target.scan_band_mask.row_sigma_ratio_threshold =
        controls.beammap_scan_band_mask_row_sigma_ratio_threshold;
    target.scan_band_mask.max_flagged_fraction =
        controls.beammap_scan_band_mask_max_flagged_fraction;
    target.split_fits_by_flag.enabled = controls.beammap_split_fits_by_flag;
    target.split_fits_by_flag.flag_values = controls.beammap_split_flag_values;
}

inline void mirror_beammap_priors_config(
    citlali::config::BeammapConfig &target,
    const BeammapPriorsConfigValues &priors) {
    target.priors.enabled = priors.enabled;
    target.priors.filepath = priors.filepath;
    target.priors.candidate_top_n = priors.candidate_top_n;
    target.priors.min_snr = priors.min_snr;
    target.priors.max_d2 = priors.max_d2;
    target.priors.max_d2_iter0 = priors.max_d2_iter0;
    target.priors.max_d2_after_iter0 = priors.max_d2_after_iter0;
    target.priors.score_lambda = priors.score_lambda;
    target.priors.score_lambda_iter0 = priors.score_lambda_iter0;
    target.priors.score_lambda_after_iter0 = priors.score_lambda_after_iter0;
    target.priors.fallback_blind = priors.fallback_blind;
    target.priors.align_after_iter0 = priors.align_after_iter0;
    target.priors.alignment_scope = priors.alignment_scope;
    target.priors.alignment_common_support = priors.alignment_common_support;
    target.priors.alignment_common_support_quantile =
        priors.alignment_common_support_quantile;
    target.priors.alignment_min_matches = priors.alignment_min_matches;
    target.priors.alignment_max_d2 = priors.alignment_max_d2;
    target.priors.alignment_fit_rotation = priors.alignment_fit_rotation;
    target.priors.alignment_max_rotation_deg =
        priors.alignment_max_rotation_deg;
}

void mirror_beammap_output_and_flagging_config(
    citlali::config::BeammapConfig &target,
    const citlali::config::BeammapDetectorTodOutputConfig &detector_tod_output,
    const BeammapFlaggingVectors &flagging_vectors,
    const std::vector<double> &sens_factors,
    const std::vector<double> &sens_psd_limits_hz) {
    target.detector_tod_output = detector_tod_output;
    target.flagging.array_lower_fwhm_arcsec =
        flagging_vectors.lower_fwhm_arcsec;
    target.flagging.array_upper_fwhm_arcsec =
        flagging_vectors.upper_fwhm_arcsec;
    target.flagging.array_lower_sig2noise = flagging_vectors.lower_sig2noise;
    target.flagging.array_upper_sig2noise = flagging_vectors.upper_sig2noise;
    target.flagging.array_max_dist_arcsec = flagging_vectors.max_dist_arcsec;
    target.flagging.array_network_robust_z = flagging_vectors.network_robust_z;
    target.flagging.sens_factors = sens_factors;
    target.flagging.sens_psd_limits_hz = sens_psd_limits_hz;
    target.flagging.max_prior_d2 = flagging_vectors.max_prior_d2;
}
