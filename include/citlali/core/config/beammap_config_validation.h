#pragma once

#include <citlali/core/config/beammap_config.h>
#include <citlali/core/config/config_error.h>

#include <cmath>

namespace citlali::config {

inline void validate(const BeammapIterationConfig &config, ValidationReport &report) {
    check_minimum(config.max_iterations, 1, {"beammap", "iter_max"}, report);
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
}

inline void validate(const BeammapFlaggingConfig &config, ValidationReport &report) {
    check_minimum(config.max_prior_d2, 0.0,
                  {"beammap", "flagging", "max_prior_d2"}, report);
}

inline void validate(const BeammapSourceFluxConfig &config, ValidationReport &report) {
    if (config.array_name.empty()) {
        report.add_error({"beammap_source", "fluxes", "array_name"},
                         "must not be empty");
    }
    if (!std::isfinite(config.value_mjy) || config.value_mjy <= 0.0) {
        report.add_error({"beammap_source", "fluxes", "value_mJy"},
                         "must be positive and finite");
    }
    if (!std::isfinite(config.uncertainty_mjy) || config.uncertainty_mjy < 0.0) {
        report.add_error({"beammap_source", "fluxes", "uncertainty_mJy"},
                         "must be greater than or equal to 0 and finite");
    }
}

inline void validate(const BeammapSourceConfig &config, ValidationReport &report) {
    if (!std::isfinite(config.ra_deg)) {
        report.add_error({"beammap_source", "ra_deg"}, "must be finite");
    }
    if (!std::isfinite(config.dec_deg)) {
        report.add_error({"beammap_source", "dec_deg"}, "must be finite");
    }
    for (const auto &flux : config.fluxes) {
        validate(flux, report);
    }
}

inline void validate(const BeammapConfig &config, ValidationReport &report) {
    validate(config.source, report);
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
