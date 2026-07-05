#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_config_loading.h>

template<typename CT>
void Engine::get_beammap_config(CT &config) {
    logger->info("getting beammap config options");
    citlali::pipeline::read_beammap_iteration_config(
        config, missing_keys, invalid_keys, beammap_iter_max,
        beammap_iter_tolerance, beammap_convergence_radius_arcsec);

    citlali::pipeline::read_beammap_phase_strategy_config(
        config, missing_keys, invalid_keys, beammap_phase_split_enabled,
        beammap_locator_iter, beammap_measurement_start_iter);
    citlali::pipeline::normalize_beammap_phase_strategy(
        beammap_iter_max, beammap_locator_iter,
        beammap_measurement_start_iter, logger);

    citlali::pipeline::read_beammap_reference_config(
        config, missing_keys, invalid_keys, beammap_reference_det,
        beammap_subtract_reference, beammap_derotate);

    // optional robust sample-level RFI masking (detector grouping)
    beammap_rfi_mask_enabled = false;
    beammap_rfi_mask_block_size_samples = 64;
    beammap_rfi_mask_min_good_samples = 32;
    beammap_rfi_mask_dilate_blocks = 1;
    beammap_rfi_mask_sigma_threshold = 6.0;
    beammap_rfi_mask_sigma_floor = 0.0;
    beammap_rfi_mask_max_flagged_fraction = 0.35;

    if (config.template has_typed<bool>(std::tuple{"beammap","rfi_mask","enabled"})) {
        get_config_value(config, beammap_rfi_mask_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","block_size_samples"})) {
        get_config_value(config, beammap_rfi_mask_block_size_samples, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","block_size_samples"},
                         {}, {8});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","min_good_samples"})) {
        get_config_value(config, beammap_rfi_mask_min_good_samples, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","min_good_samples"},
                         {}, {4});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","rfi_mask","dilate_blocks"})) {
        get_config_value(config, beammap_rfi_mask_dilate_blocks, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","dilate_blocks"},
                         {}, {0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","sigma_threshold"})) {
        get_config_value(config, beammap_rfi_mask_sigma_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","sigma_threshold"},
                         {}, {1.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","sigma_floor"})) {
        get_config_value(config, beammap_rfi_mask_sigma_floor, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","sigma_floor"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","rfi_mask","max_flagged_fraction"})) {
        get_config_value(config, beammap_rfi_mask_max_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"beammap","rfi_mask","max_flagged_fraction"},
                         {}, {0.0}, {1.0});
    }

    beammap_detector_weighting_mode = "const";
    if (config.template has_typed<std::string>(std::tuple{"beammap","detector_weighting","mode"})) {
        get_config_value(config, beammap_detector_weighting_mode, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_weighting","mode"},
                         {"const", "ptc", "ptc_after_iter0"});
    }

    beammap_fit_radius_fwhm = 0.0;
    if (config.template has_typed<double>(std::tuple{"beammap","fitting","fit_radius_fwhm"})) {
        get_config_value(config, beammap_fit_radius_fwhm, missing_keys, invalid_keys,
                         std::tuple{"beammap","fitting","fit_radius_fwhm"},
                         {}, {0.0});
    }
    map_fitter.beammap_fit_radius_fwhm = beammap_fit_radius_fwhm;

    // optional detector-map edge-band masking for coherent bad scan legs
    beammap_scan_band_mask_enabled = false;
    beammap_scan_band_mask_edge_rows = 24;
    beammap_scan_band_mask_min_row_pixels = 8;
    beammap_scan_band_mask_min_contiguous_rows = 2;
    beammap_scan_band_mask_row_median_sigma_threshold = 4.0;
    beammap_scan_band_mask_row_sigma_ratio_threshold = 2.5;
    beammap_scan_band_mask_max_flagged_fraction = 0.30;

    if (config.template has_typed<bool>(std::tuple{"beammap","scan_band_mask","enabled"})) {
        get_config_value(config, beammap_scan_band_mask_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","edge_rows"})) {
        get_config_value(config, beammap_scan_band_mask_edge_rows, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","edge_rows"},
                         {}, {2});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","min_row_pixels"})) {
        get_config_value(config, beammap_scan_band_mask_min_row_pixels, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","min_row_pixels"},
                         {}, {1});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","scan_band_mask","min_contiguous_rows"})) {
        get_config_value(config, beammap_scan_band_mask_min_contiguous_rows, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","min_contiguous_rows"},
                         {}, {1});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","row_median_sigma_threshold"})) {
        get_config_value(config, beammap_scan_band_mask_row_median_sigma_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","row_median_sigma_threshold"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","row_sigma_ratio_threshold"})) {
        get_config_value(config, beammap_scan_band_mask_row_sigma_ratio_threshold, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","row_sigma_ratio_threshold"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","scan_band_mask","max_flagged_fraction"})) {
        get_config_value(config, beammap_scan_band_mask_max_flagged_fraction, missing_keys, invalid_keys,
                         std::tuple{"beammap","scan_band_mask","max_flagged_fraction"},
                         {}, {0.0}, {1.0});
    }

    // optional split output detector-map FITS files by detector quality flag
    beammap_split_fits_by_flag = false;
    beammap_split_flag_values = {0, 1};
    if (config.template has_typed<bool>(std::tuple{"beammap","split_fits_by_flag","enabled"})) {
        get_config_value(config, beammap_split_fits_by_flag, missing_keys, invalid_keys,
                         std::tuple{"beammap","split_fits_by_flag","enabled"});
    }
    citlali::pipeline::read_beammap_split_flag_values(
        config, beammap_split_flag_values, logger);

    // optional soft priors for beammap peak initialization
    beammap_priors_enabled = false;
    beammap_priors_filepath = "null";
    beammap_priors_candidate_top_n = 64;
    beammap_priors_min_snr = 0.0;
    beammap_priors_max_d2 = 25.0;
    beammap_priors_max_d2_iter0 = 25.0;
    beammap_priors_max_d2_after_iter0 = 25.0;
    beammap_priors_score_lambda = 2.0;
    beammap_priors_score_lambda_iter0 = 2.0;
    beammap_priors_score_lambda_after_iter0 = 2.0;
    beammap_priors_fallback_blind = true;
    beammap_priors_align_after_iter0 = true;
    beammap_priors_alignment_scope = "array";
    beammap_priors_alignment_common_support = "all";
    beammap_priors_alignment_common_support_quantile = 0.02;
    beammap_priors_alignment_min_matches = 30;
    beammap_priors_alignment_max_d2 = 25.0;
    beammap_priors_alignment_fit_rotation = true;
    beammap_priors_alignment_max_rotation_deg = 8.0;

    if (config.template has_typed<bool>(std::tuple{"beammap","priors","enabled"})) {
        get_config_value(config, beammap_priors_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","enabled"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","filepath"})) {
        get_config_value(config, beammap_priors_filepath, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","filepath"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","priors","candidate_top_n"})) {
        get_config_value(config, beammap_priors_candidate_top_n, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","candidate_top_n"},
                         {}, {1});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","min_snr"})) {
        get_config_value(config, beammap_priors_min_snr, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","min_snr"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2"})) {
        get_config_value(config, beammap_priors_max_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2"},
                         {}, {0.0});
    }
    beammap_priors_max_d2_iter0 = beammap_priors_max_d2;
    beammap_priors_max_d2_after_iter0 = beammap_priors_max_d2;
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda"})) {
        get_config_value(config, beammap_priors_score_lambda, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda"},
                         {}, {0.0});
    }
    beammap_priors_score_lambda_iter0 = beammap_priors_score_lambda;
    beammap_priors_score_lambda_after_iter0 = beammap_priors_score_lambda;
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2_iter0"})) {
        get_config_value(config, beammap_priors_max_d2_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","max_d2_after_iter0"})) {
        get_config_value(config, beammap_priors_max_d2_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","max_d2_after_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda_iter0"})) {
        get_config_value(config, beammap_priors_score_lambda_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","score_lambda_after_iter0"})) {
        get_config_value(config, beammap_priors_score_lambda_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","score_lambda_after_iter0"},
                         {}, {0.0});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","fallback_blind"})) {
        get_config_value(config, beammap_priors_fallback_blind, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","fallback_blind"});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","align_after_iter0"})) {
        get_config_value(config, beammap_priors_align_after_iter0, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","align_after_iter0"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","alignment_scope"})) {
        get_config_value(config, beammap_priors_alignment_scope, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_scope"},
                         {"array", "common"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","priors","alignment_common_support"})) {
        get_config_value(config, beammap_priors_alignment_common_support, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_common_support"},
                         {"all", "overlap_box"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_common_support_quantile"})) {
        get_config_value(config, beammap_priors_alignment_common_support_quantile, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_common_support_quantile"},
                         {}, {0.0}, {0.45});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","priors","alignment_min_matches"})) {
        get_config_value(config, beammap_priors_alignment_min_matches, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_min_matches"},
                         {}, {3});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_max_d2"})) {
        get_config_value(config, beammap_priors_alignment_max_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_max_d2"},
                         {}, {0.0});
    }
    if (config.template has_typed<bool>(std::tuple{"beammap","priors","alignment_fit_rotation"})) {
        get_config_value(config, beammap_priors_alignment_fit_rotation, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_fit_rotation"});
    }
    if (config.template has_typed<double>(std::tuple{"beammap","priors","alignment_max_rotation_deg"})) {
        get_config_value(config, beammap_priors_alignment_max_rotation_deg, missing_keys, invalid_keys,
                         std::tuple{"beammap","priors","alignment_max_rotation_deg"},
                         {}, {0.0});
    }
    citlali::pipeline::disable_missing_beammap_priors(
        beammap_priors_enabled, beammap_priors_filepath, logger);

    const std::size_t n_toltec_arrays = toltec_io.array_name_map.size();
    // lower fwhm limit
    auto lower_fwhm_arcsec_vec =
        citlali::pipeline::beammap_fixed_double_vector(
            config, {"beammap","flagging","array_lower_fwhm_arcsec"},
            n_toltec_arrays, invalid_keys);
    // upper fwhm limit
    auto upper_fwhm_arcsec_vec =
        citlali::pipeline::beammap_fixed_double_vector(
            config, {"beammap","flagging","array_upper_fwhm_arcsec"},
            n_toltec_arrays, invalid_keys);
    // lower signal-to-noise limit
    auto lower_sig2noise_vec =
        citlali::pipeline::beammap_fixed_double_vector(
            config, {"beammap","flagging","array_lower_sig2noise"},
            n_toltec_arrays, invalid_keys);
    // upper signal-to-noise limit
    auto upper_sig2noise_vec =
        citlali::pipeline::beammap_fixed_double_vector(
            config, {"beammap","flagging","array_upper_sig2noise"},
            n_toltec_arrays, invalid_keys);
    // maximum allowed distance limit
    auto max_dist_arcsec_vec =
        citlali::pipeline::beammap_fixed_double_vector(
            config, {"beammap","flagging","array_max_dist_arcsec"},
            n_toltec_arrays, invalid_keys);
    // per-array post-derotation network geometry cut
    auto network_robust_z_vec =
        citlali::pipeline::beammap_fixed_double_vector(
            config, {"beammap","flagging","array_network_robust_z"},
            n_toltec_arrays, invalid_keys);
    beammap_flag_max_prior_d2 = 0.0;
    if (config.template has_typed<double>(std::tuple{"beammap","flagging","max_prior_d2"})) {
        get_config_value(config, beammap_flag_max_prior_d2, missing_keys, invalid_keys,
                         std::tuple{"beammap","flagging","max_prior_d2"},
                         {}, {0.0});
    }

    citlali::pipeline::assign_beammap_array_flag_limits(
        toltec_io.array_name_map, lower_fwhm_arcsec_vec,
        upper_fwhm_arcsec_vec, lower_sig2noise_vec, upper_sig2noise_vec,
        max_dist_arcsec_vec, network_robust_z_vec, lower_fwhm_arcsec,
        upper_fwhm_arcsec, lower_sig2noise, upper_sig2noise,
        max_dist_arcsec, network_robust_z);

    // sensitivity factors
    auto sens_factors_vec = citlali::pipeline::beammap_fixed_double_vector(
        config, {"beammap","flagging","sens_factors"}, 2, invalid_keys);
    lower_sens_factor = sens_factors_vec[0];
    upper_sens_factor = sens_factors_vec[1];

    // upper and lower frequencies over which to calculate sensitivity
    sens_psd_limits_Hz.resize(2);
    // get psd limits for sens from config
    auto sens_psd_limits_Hz_vec =
        citlali::pipeline::beammap_fixed_double_vector(
            config, {"beammap","sens_psd_limits_Hz"}, 2, invalid_keys);
    // map sens limits back to Eigen vector
    sens_psd_limits_Hz = (Eigen::Map<Eigen::VectorXd>(sens_psd_limits_Hz_vec.data(), sens_psd_limits_Hz_vec.size()));

    // Beammap PTC TOD/diagnostics are written after the convergence decision.
    // The default is the actual last attempted iteration, including early
    // convergence, so the saved PTC reflects the final cleaning state.
    beammap_tod_output_iter = -1;

    citlali::pipeline::read_beammap_detector_tod_output_config(
        config, missing_keys, invalid_keys,
        beammap_detector_tod_output_enabled,
        beammap_detector_tod_output_subdir_name,
        beammap_detector_tod_output_n_uniform,
        beammap_detector_tod_output_n_source_dense);

    typed_beammap_config = citlali::config::BeammapConfig{};
    citlali::pipeline::mirror_beammap_core_config(
        typed_beammap_config, beammap_iter_max, beammap_iter_tolerance,
        beammap_convergence_radius_arcsec, beammap_phase_split_enabled,
        beammap_locator_iter, beammap_measurement_start_iter,
        beammap_subtract_reference, static_cast<long>(beammap_reference_det),
        beammap_derotate, beammap_rfi_mask_enabled,
        beammap_rfi_mask_block_size_samples, beammap_rfi_mask_min_good_samples,
        beammap_rfi_mask_dilate_blocks, beammap_rfi_mask_sigma_threshold,
        beammap_rfi_mask_sigma_floor, beammap_rfi_mask_max_flagged_fraction,
        beammap_detector_weighting_mode, beammap_fit_radius_fwhm,
        beammap_scan_band_mask_enabled, beammap_scan_band_mask_edge_rows,
        beammap_scan_band_mask_min_row_pixels,
        beammap_scan_band_mask_min_contiguous_rows,
        beammap_scan_band_mask_row_median_sigma_threshold,
        beammap_scan_band_mask_row_sigma_ratio_threshold,
        beammap_scan_band_mask_max_flagged_fraction,
        beammap_split_fits_by_flag, beammap_split_flag_values);
    citlali::pipeline::mirror_beammap_priors_config(
        typed_beammap_config, beammap_priors_enabled,
        beammap_priors_filepath, beammap_priors_candidate_top_n,
        beammap_priors_min_snr, beammap_priors_max_d2,
        beammap_priors_max_d2_iter0, beammap_priors_max_d2_after_iter0,
        beammap_priors_score_lambda, beammap_priors_score_lambda_iter0,
        beammap_priors_score_lambda_after_iter0,
        beammap_priors_fallback_blind, beammap_priors_align_after_iter0,
        beammap_priors_alignment_scope,
        beammap_priors_alignment_common_support,
        beammap_priors_alignment_common_support_quantile,
        beammap_priors_alignment_min_matches,
        beammap_priors_alignment_max_d2,
        beammap_priors_alignment_fit_rotation,
        beammap_priors_alignment_max_rotation_deg);
    citlali::pipeline::mirror_beammap_output_and_flagging_config(
        typed_beammap_config, beammap_detector_tod_output_enabled,
        beammap_detector_tod_output_subdir_name,
        beammap_detector_tod_output_n_uniform,
        beammap_detector_tod_output_n_source_dense,
        lower_fwhm_arcsec_vec, upper_fwhm_arcsec_vec,
        lower_sig2noise_vec, upper_sig2noise_vec,
        max_dist_arcsec_vec, network_robust_z_vec,
        sens_factors_vec, sens_psd_limits_Hz_vec,
        beammap_flag_max_prior_d2);
}
