#pragma once

// Included by beammap_config_loading.h inside namespace citlali::pipeline.

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_detector_tod_output_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    bool &enabled, std::string &subdir_name, int &n_uniform,
    int &n_source_dense) {
    enabled = false;
    subdir_name = "source_crossing_tod";
    n_uniform = 10;
    n_source_dense = 10;
    read_optional_beammap_config_value(
        config, enabled, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "enabled"});
    read_optional_beammap_config_value(
        config, subdir_name, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "subdir_name"});
    read_optional_beammap_config_value(
        config, n_uniform, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "n_uniform"}, {}, {0});
    read_optional_beammap_config_value(
        config, n_source_dense, missing_keys, invalid_keys,
        std::tuple{"beammap", "detector_tod_output", "n_source_dense"},
        {}, {0});
}

inline int default_beammap_tod_output_iter() {
    return -1;
}

inline void reset_beammap_config_mirror(
    citlali::config::BeammapConfig &target) {
    target = citlali::config::BeammapConfig{};
}

inline void mirror_beammap_core_config(
    citlali::config::BeammapConfig &target,
    int iter_max, double iter_tolerance, double convergence_radius_arcsec,
    bool phase_split_enabled, int locator_iter, int measurement_start_iter,
    bool subtract_reference, long reference_det, bool derotate,
    bool rfi_mask_enabled, int rfi_mask_block_size_samples,
    int rfi_mask_min_good_samples, int rfi_mask_dilate_blocks,
    double rfi_mask_sigma_threshold, double rfi_mask_sigma_floor,
    double rfi_mask_max_flagged_fraction,
    const std::string &detector_weighting_mode,
    double fit_radius_fwhm,
    bool scan_band_mask_enabled, int scan_band_mask_edge_rows,
    int scan_band_mask_min_row_pixels,
    int scan_band_mask_min_contiguous_rows,
    double scan_band_mask_row_median_sigma_threshold,
    double scan_band_mask_row_sigma_ratio_threshold,
    double scan_band_mask_max_flagged_fraction,
    bool split_fits_by_flag,
    const std::vector<int> &split_flag_values) {
    target.iteration.max_iterations = iter_max;
    target.iteration.tolerance = iter_tolerance;
    target.iteration.convergence_radius_arcsec = convergence_radius_arcsec;
    target.phase_strategy.enabled = phase_split_enabled;
    target.phase_strategy.locator_iter = locator_iter;
    target.phase_strategy.measurement_start_iter = measurement_start_iter;
    target.reference.subtract_reference_detector = subtract_reference;
    target.reference.reference_detector = reference_det;
    target.reference.derotate = derotate;
    target.rfi_mask.enabled = rfi_mask_enabled;
    target.rfi_mask.block_size_samples = rfi_mask_block_size_samples;
    target.rfi_mask.min_good_samples = rfi_mask_min_good_samples;
    target.rfi_mask.dilate_blocks = rfi_mask_dilate_blocks;
    target.rfi_mask.sigma_threshold = rfi_mask_sigma_threshold;
    target.rfi_mask.sigma_floor = rfi_mask_sigma_floor;
    target.rfi_mask.max_flagged_fraction = rfi_mask_max_flagged_fraction;
    if (auto parsed = citlali::config::parse_beammap_detector_weighting_mode(
            detector_weighting_mode)) {
        target.detector_weighting_mode = *parsed;
    }
    target.fitting.fit_radius_fwhm = fit_radius_fwhm;
    target.scan_band_mask.enabled = scan_band_mask_enabled;
    target.scan_band_mask.edge_rows = scan_band_mask_edge_rows;
    target.scan_band_mask.min_row_pixels = scan_band_mask_min_row_pixels;
    target.scan_band_mask.min_contiguous_rows =
        scan_band_mask_min_contiguous_rows;
    target.scan_band_mask.row_median_sigma_threshold =
        scan_band_mask_row_median_sigma_threshold;
    target.scan_band_mask.row_sigma_ratio_threshold =
        scan_band_mask_row_sigma_ratio_threshold;
    target.scan_band_mask.max_flagged_fraction =
        scan_band_mask_max_flagged_fraction;
    target.split_fits_by_flag.enabled = split_fits_by_flag;
    target.split_fits_by_flag.flag_values = split_flag_values;
}

inline void mirror_beammap_priors_config(
    citlali::config::BeammapConfig &target,
    bool enabled, const std::string &filepath, int candidate_top_n,
    double min_snr, double max_d2, double max_d2_iter0,
    double max_d2_after_iter0, double score_lambda,
    double score_lambda_iter0, double score_lambda_after_iter0,
    bool fallback_blind, bool align_after_iter0,
    const std::string &alignment_scope,
    const std::string &alignment_common_support,
    double alignment_common_support_quantile,
    int alignment_min_matches, double alignment_max_d2,
    bool alignment_fit_rotation, double alignment_max_rotation_deg) {
    target.priors.enabled = enabled;
    target.priors.filepath = filepath;
    target.priors.candidate_top_n = candidate_top_n;
    target.priors.min_snr = min_snr;
    target.priors.max_d2 = max_d2;
    target.priors.max_d2_iter0 = max_d2_iter0;
    target.priors.max_d2_after_iter0 = max_d2_after_iter0;
    target.priors.score_lambda = score_lambda;
    target.priors.score_lambda_iter0 = score_lambda_iter0;
    target.priors.score_lambda_after_iter0 = score_lambda_after_iter0;
    target.priors.fallback_blind = fallback_blind;
    target.priors.align_after_iter0 = align_after_iter0;
    target.priors.alignment_scope = alignment_scope;
    target.priors.alignment_common_support = alignment_common_support;
    target.priors.alignment_common_support_quantile =
        alignment_common_support_quantile;
    target.priors.alignment_min_matches = alignment_min_matches;
    target.priors.alignment_max_d2 = alignment_max_d2;
    target.priors.alignment_fit_rotation = alignment_fit_rotation;
    target.priors.alignment_max_rotation_deg = alignment_max_rotation_deg;
}

inline void mirror_beammap_output_and_flagging_config(
    citlali::config::BeammapConfig &target,
    bool detector_tod_output_enabled,
    const std::string &detector_tod_output_subdir_name,
    int detector_tod_output_n_uniform,
    int detector_tod_output_n_source_dense,
    const std::vector<double> &lower_fwhm_arcsec,
    const std::vector<double> &upper_fwhm_arcsec,
    const std::vector<double> &lower_sig2noise,
    const std::vector<double> &upper_sig2noise,
    const std::vector<double> &max_dist_arcsec,
    const std::vector<double> &network_robust_z,
    const std::vector<double> &sens_factors,
    const std::vector<double> &sens_psd_limits_hz,
    double max_prior_d2) {
    target.detector_tod_output.enabled = detector_tod_output_enabled;
    target.detector_tod_output.subdir_name = detector_tod_output_subdir_name;
    target.detector_tod_output.n_uniform = detector_tod_output_n_uniform;
    target.detector_tod_output.n_source_dense =
        detector_tod_output_n_source_dense;
    target.flagging.array_lower_fwhm_arcsec = lower_fwhm_arcsec;
    target.flagging.array_upper_fwhm_arcsec = upper_fwhm_arcsec;
    target.flagging.array_lower_sig2noise = lower_sig2noise;
    target.flagging.array_upper_sig2noise = upper_sig2noise;
    target.flagging.array_max_dist_arcsec = max_dist_arcsec;
    target.flagging.array_network_robust_z = network_robust_z;
    target.flagging.sens_factors = sens_factors;
    target.flagging.sens_psd_limits_hz = sens_psd_limits_hz;
    target.flagging.max_prior_d2 = max_prior_d2;
}
