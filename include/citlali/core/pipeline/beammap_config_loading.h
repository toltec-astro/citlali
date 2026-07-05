#pragma once

#include <citlali/core/config/beammap_config.h>

#include <cstddef>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_iteration_config(Config &config, MissingKeys &missing_keys,
                                   InvalidKeys &invalid_keys,
                                   int &iter_max,
                                   double &iter_tolerance,
                                   double &convergence_radius_arcsec) {
    ::get_config_value(config, iter_max, missing_keys, invalid_keys,
                       std::tuple{"beammap", "iter_max"});
    ::get_config_value(config, iter_tolerance, missing_keys, invalid_keys,
                       std::tuple{"beammap", "iter_tolerance"});
    convergence_radius_arcsec = 10.0;
    if (config.template has_typed<double>(
            std::tuple{"beammap", "convergence_radius_arcsec"})) {
        ::get_config_value(
            config, convergence_radius_arcsec, missing_keys, invalid_keys,
            std::tuple{"beammap", "convergence_radius_arcsec"}, {}, {0.0});
    }
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_phase_strategy_config(Config &config,
                                        MissingKeys &missing_keys,
                                        InvalidKeys &invalid_keys,
                                        bool &enabled,
                                        int &locator_iter,
                                        int &measurement_start_iter) {
    enabled = true;
    locator_iter = 0;
    measurement_start_iter = 1;
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "phase_strategy", "enabled"})) {
        ::get_config_value(
            config, enabled, missing_keys, invalid_keys,
            std::tuple{"beammap", "phase_strategy", "enabled"});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "phase_strategy", "locator_iter"})) {
        ::get_config_value(
            config, locator_iter, missing_keys, invalid_keys,
            std::tuple{"beammap", "phase_strategy", "locator_iter"}, {}, {0});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "phase_strategy",
                       "measurement_start_iter"})) {
        ::get_config_value(
            config, measurement_start_iter, missing_keys, invalid_keys,
            std::tuple{"beammap", "phase_strategy",
                       "measurement_start_iter"},
            {}, {1});
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class ReferenceDetector>
void read_beammap_reference_config(Config &config, MissingKeys &missing_keys,
                                   InvalidKeys &invalid_keys,
                                   ReferenceDetector &reference_det,
                                   bool &subtract_reference,
                                   bool &derotate) {
    ::get_config_value(config, reference_det, missing_keys, invalid_keys,
                       std::tuple{"beammap", "reference_det"});
    ::get_config_value(config, subtract_reference, missing_keys, invalid_keys,
                       std::tuple{"beammap", "subtract_reference_det"});
    ::get_config_value(config, derotate, missing_keys, invalid_keys,
                       std::tuple{"beammap", "derotate"});
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_rfi_mask_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    bool &enabled, int &block_size_samples, int &min_good_samples,
    int &dilate_blocks, double &sigma_threshold, double &sigma_floor,
    double &max_flagged_fraction) {
    enabled = false;
    block_size_samples = 64;
    min_good_samples = 32;
    dilate_blocks = 1;
    sigma_threshold = 6.0;
    sigma_floor = 0.0;
    max_flagged_fraction = 0.35;
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "rfi_mask", "enabled"})) {
        ::get_config_value(config, enabled, missing_keys, invalid_keys,
                           std::tuple{"beammap", "rfi_mask", "enabled"});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "rfi_mask", "block_size_samples"})) {
        ::get_config_value(
            config, block_size_samples, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "block_size_samples"}, {}, {8});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "rfi_mask", "min_good_samples"})) {
        ::get_config_value(
            config, min_good_samples, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "min_good_samples"}, {}, {4});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "rfi_mask", "dilate_blocks"})) {
        ::get_config_value(
            config, dilate_blocks, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "dilate_blocks"}, {}, {0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "rfi_mask", "sigma_threshold"})) {
        ::get_config_value(
            config, sigma_threshold, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "sigma_threshold"}, {}, {1.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "rfi_mask", "sigma_floor"})) {
        ::get_config_value(
            config, sigma_floor, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "sigma_floor"}, {}, {0.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "rfi_mask", "max_flagged_fraction"})) {
        ::get_config_value(
            config, max_flagged_fraction, missing_keys, invalid_keys,
            std::tuple{"beammap", "rfi_mask", "max_flagged_fraction"}, {},
            {0.0}, {1.0});
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class MapFitter>
void read_beammap_fitting_config(Config &config, MissingKeys &missing_keys,
                                 InvalidKeys &invalid_keys,
                                 std::string &detector_weighting_mode,
                                 double &fit_radius_fwhm,
                                 MapFitter &map_fitter) {
    detector_weighting_mode = "const";
    if (config.template has_typed<std::string>(
            std::tuple{"beammap", "detector_weighting", "mode"})) {
        ::get_config_value(
            config, detector_weighting_mode, missing_keys, invalid_keys,
            std::tuple{"beammap", "detector_weighting", "mode"},
            {"const", "ptc", "ptc_after_iter0"});
    }
    fit_radius_fwhm = 0.0;
    if (config.template has_typed<double>(
            std::tuple{"beammap", "fitting", "fit_radius_fwhm"})) {
        ::get_config_value(
            config, fit_radius_fwhm, missing_keys, invalid_keys,
            std::tuple{"beammap", "fitting", "fit_radius_fwhm"}, {}, {0.0});
    }
    map_fitter.beammap_fit_radius_fwhm = fit_radius_fwhm;
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_scan_band_mask_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    bool &enabled, int &edge_rows, int &min_row_pixels,
    int &min_contiguous_rows, double &row_median_sigma_threshold,
    double &row_sigma_ratio_threshold, double &max_flagged_fraction) {
    enabled = false;
    edge_rows = 24;
    min_row_pixels = 8;
    min_contiguous_rows = 2;
    row_median_sigma_threshold = 4.0;
    row_sigma_ratio_threshold = 2.5;
    max_flagged_fraction = 0.30;
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "scan_band_mask", "enabled"})) {
        ::get_config_value(
            config, enabled, missing_keys, invalid_keys,
            std::tuple{"beammap", "scan_band_mask", "enabled"});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "scan_band_mask", "edge_rows"})) {
        ::get_config_value(
            config, edge_rows, missing_keys, invalid_keys,
            std::tuple{"beammap", "scan_band_mask", "edge_rows"}, {}, {2});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "scan_band_mask", "min_row_pixels"})) {
        ::get_config_value(
            config, min_row_pixels, missing_keys, invalid_keys,
            std::tuple{"beammap", "scan_band_mask", "min_row_pixels"}, {},
            {1});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "scan_band_mask",
                       "min_contiguous_rows"})) {
        ::get_config_value(
            config, min_contiguous_rows, missing_keys, invalid_keys,
            std::tuple{"beammap", "scan_band_mask",
                       "min_contiguous_rows"},
            {}, {1});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "scan_band_mask",
                       "row_median_sigma_threshold"})) {
        ::get_config_value(
            config, row_median_sigma_threshold, missing_keys, invalid_keys,
            std::tuple{"beammap", "scan_band_mask",
                       "row_median_sigma_threshold"},
            {}, {0.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "scan_band_mask",
                       "row_sigma_ratio_threshold"})) {
        ::get_config_value(
            config, row_sigma_ratio_threshold, missing_keys, invalid_keys,
            std::tuple{"beammap", "scan_band_mask",
                       "row_sigma_ratio_threshold"},
            {}, {0.0});
    }
    if (config.template has_typed<double>(
            std::tuple{"beammap", "scan_band_mask",
                       "max_flagged_fraction"})) {
        ::get_config_value(
            config, max_flagged_fraction, missing_keys, invalid_keys,
            std::tuple{"beammap", "scan_band_mask", "max_flagged_fraction"},
            {}, {0.0}, {1.0});
    }
}

template <class Config, class InvalidKeys>
std::vector<double> beammap_fixed_double_vector(
    Config &config, const std::vector<std::string> &path,
    std::size_t expected_size, InvalidKeys &invalid_keys) {
    std::vector<double> values;
    if (path.size() == 2) {
        values = config.template get_typed<std::vector<double>>(
            std::make_tuple(path[0], path[1]));
    }
    else {
        values = config.template get_typed<std::vector<double>>(
            std::make_tuple(path[0], path[1], path[2]));
    }
    if (values.size() != expected_size) {
        invalid_keys.push_back(path);
        values.resize(expected_size, 0.0);
    }
    return values;
}

template <class ArrayNameMap, class ValueMap>
void assign_beammap_array_flag_limits(
    const ArrayNameMap &array_name_map,
    const std::vector<double> &lower_fwhm_arcsec_vec,
    const std::vector<double> &upper_fwhm_arcsec_vec,
    const std::vector<double> &lower_sig2noise_vec,
    const std::vector<double> &upper_sig2noise_vec,
    const std::vector<double> &max_dist_arcsec_vec,
    const std::vector<double> &network_robust_z_vec,
    ValueMap &lower_fwhm_arcsec,
    ValueMap &upper_fwhm_arcsec,
    ValueMap &lower_sig2noise,
    ValueMap &upper_sig2noise,
    ValueMap &max_dist_arcsec,
    ValueMap &network_robust_z) {
    std::size_t i = 0;
    for (auto const& [arr_index, arr_name] : array_name_map) {
        (void)arr_index;
        lower_fwhm_arcsec[arr_name] = lower_fwhm_arcsec_vec[i];
        upper_fwhm_arcsec[arr_name] = upper_fwhm_arcsec_vec[i];
        lower_sig2noise[arr_name] = lower_sig2noise_vec[i];
        upper_sig2noise[arr_name] = upper_sig2noise_vec[i];
        max_dist_arcsec[arr_name] = max_dist_arcsec_vec[i];
        network_robust_z[arr_name] = network_robust_z_vec[i];
        ++i;
    }
}

inline std::vector<int> normalized_beammap_split_flag_values(
    std::vector<int> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

template <class Logger>
void normalize_beammap_phase_strategy(int iter_max, int &locator_iter,
                                      int &measurement_start_iter,
                                      const Logger &logger) {
    if (locator_iter != 0) {
        logger->warn(
            "beammap.phase_strategy.locator_iter={} requested, but the locator pass must be iter 0; using 0",
            locator_iter);
        locator_iter = 0;
    }
    if (measurement_start_iter <= locator_iter) {
        logger->warn(
            "beammap.phase_strategy.measurement_start_iter={} must be after locator_iter={}; using {}",
            measurement_start_iter, locator_iter, locator_iter + 1);
        measurement_start_iter = locator_iter + 1;
    }
    if (iter_max <= measurement_start_iter) {
        logger->warn(
            "beammap.iter_max={} will not run a measurement pass with measurement_start_iter={}",
            iter_max, measurement_start_iter);
    }
}

template <class Logger>
void disable_missing_beammap_priors(bool &enabled,
                                    const std::string &filepath,
                                    const Logger &logger) {
    if (!enabled || filepath != "null") {
        return;
    }
    logger->warn(
        "beammap.priors.enabled=true but beammap.priors.filepath is null; disabling priors");
    enabled = false;
}

template <class Config, class Logger>
void read_beammap_split_flag_values(Config &config,
                                    std::vector<int> &flag_values,
                                    const Logger &logger) {
    const auto key = std::tuple{"beammap", "split_fits_by_flag",
                                "flag_values"};
    if (!config.template has_typed<std::vector<int>>(key)) {
        return;
    }
    auto values = config.template get_typed<std::vector<int>>(key);
    if (values.empty()) {
        logger->warn(
            "beammap.split_fits_by_flag.flag_values is empty; using defaults [0, 1]");
        return;
    }
    flag_values = normalized_beammap_split_flag_values(std::move(values));
}

template <class Config, class MissingKeys, class InvalidKeys>
void read_beammap_detector_tod_output_config(
    Config &config, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    bool &enabled, std::string &subdir_name, int &n_uniform,
    int &n_source_dense) {
    enabled = false;
    subdir_name = "source_crossing_tod";
    n_uniform = 10;
    n_source_dense = 10;
    if (config.template has_typed<bool>(
            std::tuple{"beammap", "detector_tod_output", "enabled"})) {
        ::get_config_value(
            config, enabled, missing_keys, invalid_keys,
            std::tuple{"beammap", "detector_tod_output", "enabled"});
    }
    if (config.template has_typed<std::string>(
            std::tuple{"beammap", "detector_tod_output", "subdir_name"})) {
        ::get_config_value(
            config, subdir_name, missing_keys, invalid_keys,
            std::tuple{"beammap", "detector_tod_output", "subdir_name"});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "detector_tod_output", "n_uniform"})) {
        ::get_config_value(
            config, n_uniform, missing_keys, invalid_keys,
            std::tuple{"beammap", "detector_tod_output", "n_uniform"},
            {}, {0});
    }
    if (config.template has_typed<int>(
            std::tuple{"beammap", "detector_tod_output", "n_source_dense"})) {
        ::get_config_value(
            config, n_source_dense, missing_keys, invalid_keys,
            std::tuple{"beammap", "detector_tod_output", "n_source_dense"},
            {}, {0});
    }
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

}  // namespace citlali::pipeline
