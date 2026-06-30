#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/enum_parser.h>

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::config {

enum class BeammapDetectorWeightingMode {
    constant,
    ptc,
    ptc_after_iter0
};

inline constexpr std::array<EnumName<BeammapDetectorWeightingMode>, 3>
    beammap_detector_weighting_mode_names{{
        {BeammapDetectorWeightingMode::constant, "const"},
        {BeammapDetectorWeightingMode::ptc, "ptc"},
        {BeammapDetectorWeightingMode::ptc_after_iter0, "ptc_after_iter0"},
    }};

inline std::optional<BeammapDetectorWeightingMode>
parse_beammap_detector_weighting_mode(std::string_view value) {
    return parse_enum(value, beammap_detector_weighting_mode_names);
}

inline std::string_view to_string(BeammapDetectorWeightingMode value) {
    return enum_name(value, beammap_detector_weighting_mode_names);
}

struct BeammapIterationConfig {
    int max_iterations = 1;
    double tolerance = 0.0;
    double convergence_radius_arcsec = 10.0;
};

struct BeammapPhaseStrategyConfig {
    bool enabled = true;
    int locator_iter = 0;
    int measurement_start_iter = 1;
};

struct BeammapReferenceConfig {
    bool subtract_reference_detector = false;
    long reference_detector = -99;
    bool derotate = false;
};

struct BeammapRfiMaskConfig {
    bool enabled = false;
    int block_size_samples = 64;
    int min_good_samples = 32;
    int dilate_blocks = 1;
    double sigma_threshold = 6.0;
    double sigma_floor = 0.0;
    double max_flagged_fraction = 0.35;
};

struct BeammapFittingConfig {
    double fit_radius_fwhm = 0.0;
};

struct BeammapScanBandMaskConfig {
    bool enabled = false;
    int edge_rows = 24;
    int min_row_pixels = 8;
    int min_contiguous_rows = 2;
    double row_median_sigma_threshold = 4.0;
    double row_sigma_ratio_threshold = 2.5;
    double max_flagged_fraction = 0.30;
};

struct BeammapSplitFitsByFlagConfig {
    bool enabled = false;
    std::vector<int> flag_values{0, 1};
};

struct BeammapPriorsConfig {
    bool enabled = false;
    std::string filepath = "null";
    int candidate_top_n = 64;
    double min_snr = 0.0;
    double max_d2 = 25.0;
    double max_d2_iter0 = 25.0;
    double max_d2_after_iter0 = 25.0;
    double score_lambda = 2.0;
    double score_lambda_iter0 = 2.0;
    double score_lambda_after_iter0 = 2.0;
    bool fallback_blind = true;
    bool align_after_iter0 = true;
    std::string alignment_scope = "array";
    std::string alignment_common_support = "all";
    double alignment_common_support_quantile = 0.02;
    int alignment_min_matches = 30;
    double alignment_max_d2 = 25.0;
    bool alignment_fit_rotation = true;
    double alignment_max_rotation_deg = 8.0;
};

struct BeammapDetectorTodOutputConfig {
    bool enabled = false;
    std::string subdir_name = "source_crossing_tod";
    int n_uniform = 10;
    int n_source_dense = 10;
};

struct BeammapFlaggingConfig {
    std::vector<double> array_lower_fwhm_arcsec;
    std::vector<double> array_upper_fwhm_arcsec;
    std::vector<double> array_lower_sig2noise;
    std::vector<double> array_upper_sig2noise;
    std::vector<double> array_max_dist_arcsec;
    std::vector<double> array_network_robust_z;
    std::vector<double> sens_factors;
    std::vector<double> sens_psd_limits_hz;
    double max_prior_d2 = 0.0;
};

struct BeammapSourceFluxConfig {
    std::string array_name;
    double value_mjy = 0.0;
    double uncertainty_mjy = 0.0;
};

struct BeammapSourceConfig {
    std::string name;
    double ra_deg = 0.0;
    double dec_deg = 0.0;
    std::vector<BeammapSourceFluxConfig> fluxes;
};

struct BeammapConfig {
    BeammapSourceConfig source;
    BeammapIterationConfig iteration;
    BeammapPhaseStrategyConfig phase_strategy;
    BeammapReferenceConfig reference;
    BeammapRfiMaskConfig rfi_mask;
    BeammapDetectorWeightingMode detector_weighting_mode =
        BeammapDetectorWeightingMode::constant;
    BeammapFittingConfig fitting;
    BeammapScanBandMaskConfig scan_band_mask;
    BeammapSplitFitsByFlagConfig split_fits_by_flag;
    BeammapPriorsConfig priors;
    BeammapDetectorTodOutputConfig detector_tod_output;
    BeammapFlaggingConfig flagging;
};

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
