#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_config_loading.h>

template<typename CT>
void Engine::get_beammap_config(CT &config) {
    logger->info("getting beammap config options");
    // max beammap iteration
    get_config_value(config, beammap_iter_max, missing_keys, invalid_keys,
                     std::tuple{"beammap","iter_max"});
    // beammap iteration tolerance
    get_config_value(config, beammap_iter_tolerance, missing_keys, invalid_keys,
                     std::tuple{"beammap","iter_tolerance"});
    beammap_convergence_radius_arcsec = 10.0;
    if (config.template has_typed<double>(std::tuple{"beammap","convergence_radius_arcsec"})) {
        get_config_value(config, beammap_convergence_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"beammap","convergence_radius_arcsec"},
                         {}, {0.0});
    }

    beammap_phase_split_enabled = true;
    beammap_locator_iter = 0;
    beammap_measurement_start_iter = 1;
    if (config.template has_typed<bool>(std::tuple{"beammap","phase_strategy","enabled"})) {
        get_config_value(config, beammap_phase_split_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","enabled"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","phase_strategy","locator_iter"})) {
        get_config_value(config, beammap_locator_iter, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","locator_iter"},
                         {}, {0});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","phase_strategy","measurement_start_iter"})) {
        get_config_value(config, beammap_measurement_start_iter, missing_keys, invalid_keys,
                         std::tuple{"beammap","phase_strategy","measurement_start_iter"},
                         {}, {1});
    }
    if (beammap_locator_iter != 0) {
        logger->warn(
            "beammap.phase_strategy.locator_iter={} requested, but the locator pass must be iter 0; using 0",
            beammap_locator_iter);
        beammap_locator_iter = 0;
    }
    if (beammap_measurement_start_iter <= beammap_locator_iter) {
        logger->warn(
            "beammap.phase_strategy.measurement_start_iter={} must be after locator_iter={}; using {}",
            beammap_measurement_start_iter, beammap_locator_iter, beammap_locator_iter + 1);
        beammap_measurement_start_iter = beammap_locator_iter + 1;
    }
    if (beammap_iter_max <= beammap_measurement_start_iter) {
        logger->warn(
            "beammap.iter_max={} will not run a measurement pass with measurement_start_iter={}",
            beammap_iter_max, beammap_measurement_start_iter);
    }

    // beammap reference detector
    get_config_value(config, beammap_reference_det, missing_keys, invalid_keys,
                     std::tuple{"beammap","reference_det"});
    // subtract reference detector?
    get_config_value(config, beammap_subtract_reference, missing_keys, invalid_keys,
                     std::tuple{"beammap","subtract_reference_det"});
    // derotate apt?
    get_config_value(config, beammap_derotate, missing_keys, invalid_keys,
                     std::tuple{"beammap","derotate"});

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
    if (config.template has_typed<std::vector<int>>(std::tuple{"beammap","split_fits_by_flag","flag_values"})) {
        auto values = config.template get_typed<std::vector<int>>(
            std::tuple{"beammap","split_fits_by_flag","flag_values"});
        if (values.empty()) {
            logger->warn("beammap.split_fits_by_flag.flag_values is empty; using defaults [0, 1]");
        }
        else {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            beammap_split_flag_values = std::move(values);
        }
    }

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
    if (beammap_priors_enabled && beammap_priors_filepath == "null") {
        logger->warn("beammap.priors.enabled=true but beammap.priors.filepath is null; disabling priors");
        beammap_priors_enabled = false;
    }

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

    // add params to respective array values
    Eigen::Index i = 0;
    for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
        // lower fwhm limit
        lower_fwhm_arcsec[arr_name] = lower_fwhm_arcsec_vec[i];
        // upper fwhm limit
        upper_fwhm_arcsec[arr_name] = upper_fwhm_arcsec_vec[i];
        // lower signal-to-noise limit
        lower_sig2noise[arr_name] = lower_sig2noise_vec[i];
        // upper signal-to-noise limit
        upper_sig2noise[arr_name] = upper_sig2noise_vec[i];
        // maximum allowed distance limit
        max_dist_arcsec[arr_name] = max_dist_arcsec_vec[i];
        // post-process per-network robust-z limit
        network_robust_z[arr_name] = network_robust_z_vec[i];
        i++;
    }

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

    beammap_detector_tod_output_enabled = false;
    beammap_detector_tod_output_subdir_name = "source_crossing_tod";
    beammap_detector_tod_output_n_uniform = 10;
    beammap_detector_tod_output_n_source_dense = 10;
    if (config.template has_typed<bool>(std::tuple{"beammap","detector_tod_output","enabled"})) {
        get_config_value(config, beammap_detector_tod_output_enabled, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","enabled"});
    }
    if (config.template has_typed<std::string>(std::tuple{"beammap","detector_tod_output","subdir_name"})) {
        get_config_value(config, beammap_detector_tod_output_subdir_name, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","subdir_name"});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","detector_tod_output","n_uniform"})) {
        get_config_value(config, beammap_detector_tod_output_n_uniform, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","n_uniform"},
                         {}, {0});
    }
    if (config.template has_typed<int>(std::tuple{"beammap","detector_tod_output","n_source_dense"})) {
        get_config_value(config, beammap_detector_tod_output_n_source_dense, missing_keys, invalid_keys,
                         std::tuple{"beammap","detector_tod_output","n_source_dense"},
                         {}, {0});
    }

    typed_beammap_config = citlali::config::BeammapConfig{};
    typed_beammap_config.iteration.max_iterations = beammap_iter_max;
    typed_beammap_config.iteration.tolerance = beammap_iter_tolerance;
    typed_beammap_config.iteration.convergence_radius_arcsec =
        beammap_convergence_radius_arcsec;
    typed_beammap_config.phase_strategy.enabled = beammap_phase_split_enabled;
    typed_beammap_config.phase_strategy.locator_iter = beammap_locator_iter;
    typed_beammap_config.phase_strategy.measurement_start_iter =
        beammap_measurement_start_iter;
    typed_beammap_config.reference.subtract_reference_detector =
        beammap_subtract_reference;
    typed_beammap_config.reference.reference_detector =
        static_cast<long>(beammap_reference_det);
    typed_beammap_config.reference.derotate = beammap_derotate;
    typed_beammap_config.rfi_mask.enabled = beammap_rfi_mask_enabled;
    typed_beammap_config.rfi_mask.block_size_samples =
        beammap_rfi_mask_block_size_samples;
    typed_beammap_config.rfi_mask.min_good_samples =
        beammap_rfi_mask_min_good_samples;
    typed_beammap_config.rfi_mask.dilate_blocks = beammap_rfi_mask_dilate_blocks;
    typed_beammap_config.rfi_mask.sigma_threshold =
        beammap_rfi_mask_sigma_threshold;
    typed_beammap_config.rfi_mask.sigma_floor = beammap_rfi_mask_sigma_floor;
    typed_beammap_config.rfi_mask.max_flagged_fraction =
        beammap_rfi_mask_max_flagged_fraction;
    if (auto parsed = citlali::config::parse_beammap_detector_weighting_mode(
            beammap_detector_weighting_mode)) {
        typed_beammap_config.detector_weighting_mode = *parsed;
    }
    typed_beammap_config.fitting.fit_radius_fwhm = beammap_fit_radius_fwhm;
    typed_beammap_config.scan_band_mask.enabled = beammap_scan_band_mask_enabled;
    typed_beammap_config.scan_band_mask.edge_rows = beammap_scan_band_mask_edge_rows;
    typed_beammap_config.scan_band_mask.min_row_pixels =
        beammap_scan_band_mask_min_row_pixels;
    typed_beammap_config.scan_band_mask.min_contiguous_rows =
        beammap_scan_band_mask_min_contiguous_rows;
    typed_beammap_config.scan_band_mask.row_median_sigma_threshold =
        beammap_scan_band_mask_row_median_sigma_threshold;
    typed_beammap_config.scan_band_mask.row_sigma_ratio_threshold =
        beammap_scan_band_mask_row_sigma_ratio_threshold;
    typed_beammap_config.scan_band_mask.max_flagged_fraction =
        beammap_scan_band_mask_max_flagged_fraction;
    typed_beammap_config.split_fits_by_flag.enabled = beammap_split_fits_by_flag;
    typed_beammap_config.split_fits_by_flag.flag_values = beammap_split_flag_values;
    typed_beammap_config.priors.enabled = beammap_priors_enabled;
    typed_beammap_config.priors.filepath = beammap_priors_filepath;
    typed_beammap_config.priors.candidate_top_n =
        beammap_priors_candidate_top_n;
    typed_beammap_config.priors.min_snr = beammap_priors_min_snr;
    typed_beammap_config.priors.max_d2 = beammap_priors_max_d2;
    typed_beammap_config.priors.max_d2_iter0 = beammap_priors_max_d2_iter0;
    typed_beammap_config.priors.max_d2_after_iter0 =
        beammap_priors_max_d2_after_iter0;
    typed_beammap_config.priors.score_lambda = beammap_priors_score_lambda;
    typed_beammap_config.priors.score_lambda_iter0 =
        beammap_priors_score_lambda_iter0;
    typed_beammap_config.priors.score_lambda_after_iter0 =
        beammap_priors_score_lambda_after_iter0;
    typed_beammap_config.priors.fallback_blind = beammap_priors_fallback_blind;
    typed_beammap_config.priors.align_after_iter0 =
        beammap_priors_align_after_iter0;
    typed_beammap_config.priors.alignment_scope =
        beammap_priors_alignment_scope;
    typed_beammap_config.priors.alignment_common_support =
        beammap_priors_alignment_common_support;
    typed_beammap_config.priors.alignment_common_support_quantile =
        beammap_priors_alignment_common_support_quantile;
    typed_beammap_config.priors.alignment_min_matches =
        beammap_priors_alignment_min_matches;
    typed_beammap_config.priors.alignment_max_d2 =
        beammap_priors_alignment_max_d2;
    typed_beammap_config.priors.alignment_fit_rotation =
        beammap_priors_alignment_fit_rotation;
    typed_beammap_config.priors.alignment_max_rotation_deg =
        beammap_priors_alignment_max_rotation_deg;
    typed_beammap_config.detector_tod_output.enabled =
        beammap_detector_tod_output_enabled;
    typed_beammap_config.detector_tod_output.subdir_name =
        beammap_detector_tod_output_subdir_name;
    typed_beammap_config.detector_tod_output.n_uniform =
        beammap_detector_tod_output_n_uniform;
    typed_beammap_config.detector_tod_output.n_source_dense =
        beammap_detector_tod_output_n_source_dense;
    typed_beammap_config.flagging.array_lower_fwhm_arcsec =
        lower_fwhm_arcsec_vec;
    typed_beammap_config.flagging.array_upper_fwhm_arcsec =
        upper_fwhm_arcsec_vec;
    typed_beammap_config.flagging.array_lower_sig2noise =
        lower_sig2noise_vec;
    typed_beammap_config.flagging.array_upper_sig2noise =
        upper_sig2noise_vec;
    typed_beammap_config.flagging.array_max_dist_arcsec =
        max_dist_arcsec_vec;
    typed_beammap_config.flagging.array_network_robust_z =
        network_robust_z_vec;
    typed_beammap_config.flagging.sens_factors = sens_factors_vec;
    typed_beammap_config.flagging.sens_psd_limits_hz = sens_psd_limits_Hz_vec;
    typed_beammap_config.flagging.max_prior_d2 = beammap_flag_max_prior_d2;
}
