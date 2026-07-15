#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/config_error.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace citlali::config {

inline void validate(const BeammapIterationConfig &config, ValidationReport &report) {
    check_minimum(config.max_iterations, 1, {"beammap", "iter_max"}, report);
    check_finite_value(config.tolerance,
                       {"beammap", "iter_tolerance"}, report);
    check_minimum(config.convergence_radius_arcsec, 0.0,
                  {"beammap", "convergence_radius_arcsec"}, report);
}

inline void validate(const BeammapPhaseStrategyConfig &config, ValidationReport &report) {
    check_minimum(config.locator_iter, 0,
                  {"beammap", "phase_strategy", "locator_iter"}, report);
    if (config.measurement_start_iter <= config.locator_iter) {
        report.add_error({"beammap", "phase_strategy", "measurement_start_iter"},
                         "must be greater than locator_iter");
    }
}

inline void validate(const BeammapRfiMaskConfig &config, ValidationReport &report) {
    check_minimum(config.block_size_samples, 8,
                  {"beammap", "rfi_mask", "block_size_samples"}, report);
    check_minimum(config.min_good_samples, 4,
                  {"beammap", "rfi_mask", "min_good_samples"}, report);
    check_minimum(config.dilate_blocks, 0,
                  {"beammap", "rfi_mask", "dilate_blocks"}, report);
    check_minimum(config.sigma_threshold, 1.0,
                  {"beammap", "rfi_mask", "sigma_threshold"}, report);
    check_minimum(config.sigma_floor, 0.0,
                  {"beammap", "rfi_mask", "sigma_floor"}, report);
    check_minimum(config.max_flagged_fraction, 0.0,
                  {"beammap", "rfi_mask", "max_flagged_fraction"}, report);
    check_maximum(config.max_flagged_fraction, 1.0,
                  {"beammap", "rfi_mask", "max_flagged_fraction"}, report);
}

inline void validate(const BeammapFittingConfig &config, ValidationReport &report) {
    check_minimum(config.fit_radius_fwhm, 0.0,
                  {"beammap", "fitting", "fit_radius_fwhm"}, report);
}

inline void validate(const BeammapScanBandMaskConfig &config, ValidationReport &report) {
    check_minimum(config.edge_rows, 2,
                  {"beammap", "scan_band_mask", "edge_rows"}, report);
    check_minimum(config.min_row_pixels, 1,
                  {"beammap", "scan_band_mask", "min_row_pixels"}, report);
    check_minimum(config.min_contiguous_rows, 1,
                  {"beammap", "scan_band_mask", "min_contiguous_rows"}, report);
    check_minimum(config.row_median_sigma_threshold, 0.0,
                  {"beammap", "scan_band_mask", "row_median_sigma_threshold"}, report);
    check_minimum(config.row_sigma_ratio_threshold, 0.0,
                  {"beammap", "scan_band_mask", "row_sigma_ratio_threshold"}, report);
    check_minimum(config.max_flagged_fraction, 0.0,
                  {"beammap", "scan_band_mask", "max_flagged_fraction"}, report);
    check_maximum(config.max_flagged_fraction, 1.0,
                  {"beammap", "scan_band_mask", "max_flagged_fraction"}, report);
}

inline void validate(const BeammapPriorsConfig &config, ValidationReport &report) {
    check_minimum(config.candidate_top_n, 1,
                  {"beammap", "priors", "candidate_top_n"}, report);
    check_finite_value(config.min_snr,
                       {"beammap", "priors", "min_snr"}, report);
    check_minimum(config.max_d2, 0.0, {"beammap", "priors", "max_d2"}, report);
    check_minimum(config.max_d2_iter0, 0.0,
                  {"beammap", "priors", "max_d2_iter0"}, report);
    check_minimum(config.max_d2_after_iter0, 0.0,
                  {"beammap", "priors", "max_d2_after_iter0"}, report);
    check_minimum(config.score_lambda, 0.0,
                  {"beammap", "priors", "score_lambda"}, report);
    check_minimum(config.score_lambda_iter0, 0.0,
                  {"beammap", "priors", "score_lambda_iter0"}, report);
    check_minimum(config.score_lambda_after_iter0, 0.0,
                  {"beammap", "priors", "score_lambda_after_iter0"}, report);
    check_minimum(config.alignment_common_support_quantile, 0.0,
                  {"beammap", "priors", "alignment_common_support_quantile"}, report);
    check_maximum(config.alignment_common_support_quantile, 0.45,
                  {"beammap", "priors", "alignment_common_support_quantile"}, report);
    check_minimum(config.alignment_min_matches, 3,
                  {"beammap", "priors", "alignment_min_matches"}, report);
    check_minimum(config.alignment_max_d2, 0.0,
                  {"beammap", "priors", "alignment_max_d2"}, report);
    check_minimum(config.alignment_max_rotation_deg, 0.0,
                  {"beammap", "priors", "alignment_max_rotation_deg"}, report);
}

inline void validate(const BeammapDetectorTodOutputConfig &config,
                     ValidationReport &report) {
    check_minimum(config.n_uniform, 0,
                  {"beammap", "detector_tod_output", "n_uniform"}, report);
    check_minimum(config.n_source_dense, 0,
                  {"beammap", "detector_tod_output", "n_source_dense"}, report);
    if (config.enabled && config.n_uniform == 0 &&
        config.n_source_dense == 0) {
        report.add_error(
            {"beammap", "detector_tod_output"},
            "enabled output requires at least one uniform or source-dense slot");
    }
    if (config.enabled && config.subdir_name.empty()) {
        report.add_error(
            {"beammap", "detector_tod_output", "subdir_name"},
            "must not be empty when detector TOD output is enabled");
    }
}

inline void validate_finite_beammap_vector(
    const std::vector<double> &values, const ConfigPath &path,
    ValidationReport &report) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        check_finite_value(
            values[index], append_config_path(path, {std::to_string(index)}),
            report);
    }
}

inline void validate_beammap_vector_size(
    const std::vector<double> &values, std::size_t expected_size,
    const ConfigPath &path, ValidationReport &report) {
    if (!values.empty() && values.size() != expected_size) {
        report.add_error(
            path, "must contain exactly " + std::to_string(expected_size) +
                      " values");
    }
}

inline void validate(const BeammapFlaggingConfig &config, ValidationReport &report) {
    check_minimum(config.max_prior_d2, 0.0,
                  {"beammap", "flagging", "max_prior_d2"}, report);

    const std::vector<std::pair<const std::vector<double> *, ConfigPath>>
        array_vectors{
            {&config.array_lower_fwhm_arcsec,
             {"beammap", "flagging", "array_lower_fwhm_arcsec"}},
            {&config.array_upper_fwhm_arcsec,
             {"beammap", "flagging", "array_upper_fwhm_arcsec"}},
            {&config.array_lower_sig2noise,
             {"beammap", "flagging", "array_lower_sig2noise"}},
            {&config.array_upper_sig2noise,
             {"beammap", "flagging", "array_upper_sig2noise"}},
            {&config.array_max_dist_arcsec,
             {"beammap", "flagging", "array_max_dist_arcsec"}},
            {&config.array_network_robust_z,
             {"beammap", "flagging", "array_network_robust_z"}},
        };

    std::size_t array_count = 0;
    for (const auto &[values, path] : array_vectors) {
        validate_finite_beammap_vector(*values, path, report);
        if (array_count == 0 && !values->empty()) {
            array_count = values->size();
        }
    }
    if (array_count > 0) {
        for (const auto &[values, path] : array_vectors) {
            if (values->size() == array_count) {
                continue;
            }
            report.add_error(
                path, "must have the same array cardinality as the other "
                      "beammap flagging vectors");
        }
    }

    const ConfigPath sensitivity_factor_path{
        "beammap", "flagging", "sens_factors"};
    validate_finite_beammap_vector(
        config.sens_factors, sensitivity_factor_path, report);
    validate_beammap_vector_size(
        config.sens_factors, 2, sensitivity_factor_path, report);

    const ConfigPath sensitivity_band_path{
        "beammap", "sens_psd_limits_Hz"};
    validate_finite_beammap_vector(
        config.sens_psd_limits_hz, sensitivity_band_path, report);
    validate_beammap_vector_size(
        config.sens_psd_limits_hz, 2, sensitivity_band_path, report);
}

inline void validate(const BeammapConfig &config, ValidationReport &report) {
    validate(config.iteration, report);
    validate(config.phase_strategy, report);
    validate(config.rfi_mask, report);
    validate(config.fitting, report);
    validate(config.scan_band_mask, report);
    validate(config.priors, report);
    validate(config.detector_tod_output, report);
    validate(config.flagging, report);
}

}  // namespace citlali::config
