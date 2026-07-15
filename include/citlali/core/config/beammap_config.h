#pragma once

#include <citlali/core/config/enum_parser.h>

#include <array>
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

enum class BeammapPriorAlignmentScope {
    array,
    common
};

enum class BeammapPriorAlignmentSupport {
    all,
    overlap_box
};

inline constexpr std::array<EnumName<BeammapDetectorWeightingMode>, 3>
    beammap_detector_weighting_mode_names{{
        {BeammapDetectorWeightingMode::constant, "const"},
        {BeammapDetectorWeightingMode::ptc, "ptc"},
        {BeammapDetectorWeightingMode::ptc_after_iter0, "ptc_after_iter0"},
    }};

inline constexpr std::array<EnumName<BeammapPriorAlignmentScope>, 2>
    beammap_prior_alignment_scope_names{{
        {BeammapPriorAlignmentScope::array, "array"},
        {BeammapPriorAlignmentScope::common, "common"},
    }};

inline constexpr std::array<EnumName<BeammapPriorAlignmentSupport>, 2>
    beammap_prior_alignment_support_names{{
        {BeammapPriorAlignmentSupport::all, "all"},
        {BeammapPriorAlignmentSupport::overlap_box, "overlap_box"},
    }};

inline std::optional<BeammapDetectorWeightingMode>
parse_beammap_detector_weighting_mode(std::string_view value) {
    return parse_enum(value, beammap_detector_weighting_mode_names);
}

inline std::optional<BeammapPriorAlignmentScope>
parse_beammap_prior_alignment_scope(std::string_view value) {
    return parse_enum(value, beammap_prior_alignment_scope_names);
}

inline std::optional<BeammapPriorAlignmentSupport>
parse_beammap_prior_alignment_support(std::string_view value) {
    return parse_enum(value, beammap_prior_alignment_support_names);
}

inline std::string_view to_string(BeammapDetectorWeightingMode value) {
    return enum_name(value, beammap_detector_weighting_mode_names);
}

inline std::string_view to_string(BeammapPriorAlignmentScope value) {
    return enum_name(value, beammap_prior_alignment_scope_names);
}

inline std::string_view to_string(BeammapPriorAlignmentSupport value) {
    return enum_name(value, beammap_prior_alignment_support_names);
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
    BeammapPriorAlignmentScope alignment_scope =
        BeammapPriorAlignmentScope::array;
    BeammapPriorAlignmentSupport alignment_common_support =
        BeammapPriorAlignmentSupport::all;
    double alignment_common_support_quantile = 0.02;
    int alignment_min_matches = 30;
    double alignment_max_d2 = 25.0;
    bool alignment_fit_rotation = true;
    double alignment_max_rotation_deg = 8.0;
};

inline bool uses_common_prior_alignment(
    const BeammapPriorsConfig &config) {
    return config.alignment_scope == BeammapPriorAlignmentScope::common;
}

inline bool uses_overlap_box_prior_alignment_support(
    const BeammapPriorsConfig &config) {
    return config.alignment_common_support ==
           BeammapPriorAlignmentSupport::overlap_box;
}

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

struct BeammapConfig {
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

}  // namespace citlali::config
