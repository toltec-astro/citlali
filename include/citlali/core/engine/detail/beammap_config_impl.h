#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_config_loading.h>

template<typename CT>
void Engine::get_beammap_config(CT &config) {
    logger->info("getting beammap config options");
    auto &beammap_config = typed_config.beammap;
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

    citlali::pipeline::read_beammap_rfi_mask_config(
        config, missing_keys, invalid_keys, beammap_rfi_mask_enabled,
        beammap_rfi_mask_block_size_samples,
        beammap_rfi_mask_min_good_samples, beammap_rfi_mask_dilate_blocks,
        beammap_rfi_mask_sigma_threshold, beammap_rfi_mask_sigma_floor,
        beammap_rfi_mask_max_flagged_fraction);

    citlali::pipeline::read_beammap_fitting_config(
        config, missing_keys, invalid_keys, beammap_detector_weighting_mode,
        beammap_fit_radius_fwhm, map_fitter);

    citlali::pipeline::read_beammap_scan_band_mask_config(
        config, missing_keys, invalid_keys, beammap_scan_band_mask_enabled,
        beammap_scan_band_mask_edge_rows,
        beammap_scan_band_mask_min_row_pixels,
        beammap_scan_band_mask_min_contiguous_rows,
        beammap_scan_band_mask_row_median_sigma_threshold,
        beammap_scan_band_mask_row_sigma_ratio_threshold,
        beammap_scan_band_mask_max_flagged_fraction);

    citlali::pipeline::read_beammap_split_fits_config(
        config, missing_keys, invalid_keys, beammap_split_fits_by_flag,
        beammap_split_flag_values, logger);

    citlali::pipeline::initialize_beammap_priors_defaults(
        beammap_priors_enabled, beammap_priors_filepath,
        beammap_priors_candidate_top_n, beammap_priors_min_snr,
        beammap_priors_max_d2, beammap_priors_max_d2_iter0,
        beammap_priors_max_d2_after_iter0, beammap_priors_score_lambda,
        beammap_priors_score_lambda_iter0,
        beammap_priors_score_lambda_after_iter0,
        beammap_priors_fallback_blind, beammap_priors_align_after_iter0,
        beammap_priors_alignment_scope,
        beammap_priors_alignment_common_support,
        beammap_priors_alignment_common_support_quantile,
        beammap_priors_alignment_min_matches,
        beammap_priors_alignment_max_d2,
        beammap_priors_alignment_fit_rotation,
        beammap_priors_alignment_max_rotation_deg);

    citlali::pipeline::read_beammap_priors_core_config(
        config, missing_keys, invalid_keys, beammap_priors_enabled,
        beammap_priors_filepath, beammap_priors_candidate_top_n,
        beammap_priors_min_snr, beammap_priors_max_d2,
        beammap_priors_score_lambda);
    citlali::pipeline::set_beammap_priors_iteration_defaults(
        beammap_priors_max_d2, beammap_priors_max_d2_iter0,
        beammap_priors_max_d2_after_iter0, beammap_priors_score_lambda,
        beammap_priors_score_lambda_iter0,
        beammap_priors_score_lambda_after_iter0);
    citlali::pipeline::read_beammap_priors_iteration_config(
        config, missing_keys, invalid_keys, beammap_priors_max_d2_iter0,
        beammap_priors_max_d2_after_iter0,
        beammap_priors_score_lambda_iter0,
        beammap_priors_score_lambda_after_iter0);
    citlali::pipeline::read_beammap_priors_behavior_config(
        config, missing_keys, invalid_keys, beammap_priors_fallback_blind,
        beammap_priors_align_after_iter0);
    citlali::pipeline::read_beammap_priors_alignment_config(
        config, missing_keys, invalid_keys,
        beammap_priors_alignment_scope,
        beammap_priors_alignment_common_support,
        beammap_priors_alignment_common_support_quantile,
        beammap_priors_alignment_min_matches,
        beammap_priors_alignment_max_d2,
        beammap_priors_alignment_fit_rotation,
        beammap_priors_alignment_max_rotation_deg);
    citlali::pipeline::disable_missing_beammap_priors(
        beammap_priors_enabled, beammap_priors_filepath, logger);

    const auto flagging_vectors =
        citlali::pipeline::read_beammap_flagging_vectors(
            config, missing_keys, invalid_keys, toltec_io.array_name_map.size());
    beammap_flag_max_prior_d2 = flagging_vectors.max_prior_d2;

    citlali::pipeline::assign_beammap_array_flag_limits(
        toltec_io.array_name_map, flagging_vectors.lower_fwhm_arcsec,
        flagging_vectors.upper_fwhm_arcsec,
        flagging_vectors.lower_sig2noise,
        flagging_vectors.upper_sig2noise,
        flagging_vectors.max_dist_arcsec, flagging_vectors.network_robust_z,
        lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise,
        upper_sig2noise, max_dist_arcsec, network_robust_z);

    std::vector<double> sens_factors_vec;
    std::vector<double> sens_psd_limits_Hz_vec;
    citlali::pipeline::read_beammap_sensitivity_config(
        config, invalid_keys, lower_sens_factor, upper_sens_factor,
        sens_psd_limits_Hz, sens_factors_vec, sens_psd_limits_Hz_vec);

    // Beammap PTC TOD/diagnostics are written after the convergence decision.
    // The default is the actual last attempted iteration, including early
    // convergence, so the saved PTC reflects the final cleaning state.
    beammap_tod_output_iter =
        citlali::pipeline::default_beammap_tod_output_iter();

    citlali::pipeline::read_beammap_detector_tod_output_config(
        config, missing_keys, invalid_keys,
        beammap_detector_tod_output_enabled,
        beammap_detector_tod_output_subdir_name,
        beammap_detector_tod_output_n_uniform,
        beammap_detector_tod_output_n_source_dense);

    citlali::pipeline::reset_beammap_config_mirror(beammap_config);
    citlali::pipeline::mirror_beammap_core_config(
        beammap_config, beammap_iter_max, beammap_iter_tolerance,
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
        beammap_config, beammap_priors_enabled,
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
        beammap_config, beammap_detector_tod_output_enabled,
        beammap_detector_tod_output_subdir_name,
        beammap_detector_tod_output_n_uniform,
        beammap_detector_tod_output_n_source_dense,
        flagging_vectors.lower_fwhm_arcsec,
        flagging_vectors.upper_fwhm_arcsec,
        flagging_vectors.lower_sig2noise,
        flagging_vectors.upper_sig2noise,
        flagging_vectors.max_dist_arcsec,
        flagging_vectors.network_robust_z,
        sens_factors_vec, sens_psd_limits_Hz_vec,
        beammap_flag_max_prior_d2);
}
