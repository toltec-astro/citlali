#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/enum_parser.h>

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::config {

enum class TodType {
    xs,
    rs,
    is,
    qs
};

enum class TodOutputStream {
    rtc,
    ptc
};

enum class TodOutputType {
    none,
    rtc,
    ptc,
    both
};

enum class TodStreamOutputMode {
    full,
    mini,
    full_outer,
    mini_outer
};

enum class TodOutputSelectionMode {
    indices,
    all,
    uniform_plus_source_crossing
};

enum class RawTimeChunkFilterEdgeGuardMode {
    flag,
    none
};

enum class RawTimeChunkFilterEdgeGuardCombine {
    sum,
    max
};

enum class ProcessedTimeChunkWeightingType {
    full,
    approximate,
    hybrid,
    validated,
    constant
};

enum class ProcessedTimeChunkWeightGrouping {
    array,
    network,
    all
};

enum class ProcessedTimeChunkCleanerMode {
    none,
    standard_pca,
    null_model,
    marchenko_pastur,
    adaptive_selector
};

enum class ProcessedTimeChunkCorrGroupingMetric {
    abs,
    signed_metric
};

enum class FruitLoopsMode {
    upper,
    lower,
    both
};

enum class FruitLoopsWeightFeedbackReference {
    p95,
    p90,
    p99,
    median,
    p50,
    max,
    peak
};

enum class FruitLoopsInterpModeOverride {
    automatic,
    nearest,
    bilinear,
    jinc,
    trunc
};

inline constexpr std::array<EnumName<TodType>, 4> tod_type_names{{
    {TodType::xs, "xs"},
    {TodType::rs, "rs"},
    {TodType::is, "is"},
    {TodType::qs, "qs"},
}};

inline constexpr std::array<EnumName<TodOutputStream>, 2>
    tod_output_stream_names{{
        {TodOutputStream::rtc, "rtc"},
        {TodOutputStream::ptc, "ptc"},
    }};

inline constexpr std::array<EnumName<TodOutputType>, 4> tod_output_type_names{{
    {TodOutputType::none, "none"},
    {TodOutputType::rtc, "rtc"},
    {TodOutputType::ptc, "ptc"},
    {TodOutputType::both, "both"},
}};

inline constexpr std::array<EnumName<TodStreamOutputMode>, 4>
    tod_stream_output_mode_names{{
        {TodStreamOutputMode::full, "full"},
        {TodStreamOutputMode::mini, "mini"},
        {TodStreamOutputMode::full_outer, "full_outer"},
        {TodStreamOutputMode::mini_outer, "mini_outer"},
    }};

inline constexpr std::array<EnumName<TodOutputSelectionMode>, 3>
    tod_output_selection_mode_names{{
        {TodOutputSelectionMode::indices, "indices"},
        {TodOutputSelectionMode::all, "all"},
        {TodOutputSelectionMode::uniform_plus_source_crossing,
         "uniform_plus_source_crossing"},
    }};

inline constexpr std::array<EnumName<RawTimeChunkFilterEdgeGuardMode>, 2>
    raw_filter_edge_guard_mode_names{{
        {RawTimeChunkFilterEdgeGuardMode::flag, "flag"},
        {RawTimeChunkFilterEdgeGuardMode::none, "none"},
    }};

inline constexpr std::array<EnumName<RawTimeChunkFilterEdgeGuardCombine>, 2>
    raw_filter_edge_guard_combine_names{{
        {RawTimeChunkFilterEdgeGuardCombine::sum, "sum"},
        {RawTimeChunkFilterEdgeGuardCombine::max, "max"},
    }};

inline constexpr std::array<EnumName<ProcessedTimeChunkWeightingType>, 5>
    processed_weighting_type_names{{
        {ProcessedTimeChunkWeightingType::full, "full"},
        {ProcessedTimeChunkWeightingType::approximate, "approximate"},
        {ProcessedTimeChunkWeightingType::hybrid, "hybrid"},
        {ProcessedTimeChunkWeightingType::validated, "validated"},
        {ProcessedTimeChunkWeightingType::constant, "const"},
    }};

inline constexpr std::array<EnumName<ProcessedTimeChunkWeightGrouping>, 3>
    processed_weight_grouping_names{{
        {ProcessedTimeChunkWeightGrouping::array, "array"},
        {ProcessedTimeChunkWeightGrouping::network, "nw"},
        {ProcessedTimeChunkWeightGrouping::all, "all"},
    }};

inline constexpr std::array<EnumName<ProcessedTimeChunkCleanerMode>, 5>
    processed_cleaner_mode_names{{
        {ProcessedTimeChunkCleanerMode::none, "none"},
        {ProcessedTimeChunkCleanerMode::standard_pca, "standard_pca"},
        {ProcessedTimeChunkCleanerMode::null_model, "null_model"},
        {ProcessedTimeChunkCleanerMode::marchenko_pastur,
         "marchenko_pastur"},
        {ProcessedTimeChunkCleanerMode::adaptive_selector,
         "adaptive_selector"},
    }};

inline constexpr std::array<EnumName<ProcessedTimeChunkCorrGroupingMetric>, 2>
    processed_corr_grouping_metric_names{{
        {ProcessedTimeChunkCorrGroupingMetric::abs, "abs"},
        {ProcessedTimeChunkCorrGroupingMetric::signed_metric, "signed"},
    }};

inline constexpr std::array<EnumName<FruitLoopsMode>, 3>
    fruit_loops_mode_names{{
        {FruitLoopsMode::upper, "upper"},
        {FruitLoopsMode::lower, "lower"},
        {FruitLoopsMode::both, "both"},
    }};

inline constexpr std::array<EnumName<FruitLoopsWeightFeedbackReference>, 7>
    fruit_loops_weight_feedback_reference_names{{
        {FruitLoopsWeightFeedbackReference::p95, "p95"},
        {FruitLoopsWeightFeedbackReference::p90, "p90"},
        {FruitLoopsWeightFeedbackReference::p99, "p99"},
        {FruitLoopsWeightFeedbackReference::median, "median"},
        {FruitLoopsWeightFeedbackReference::p50, "p50"},
        {FruitLoopsWeightFeedbackReference::max, "max"},
        {FruitLoopsWeightFeedbackReference::peak, "peak"},
    }};

inline constexpr std::array<EnumName<FruitLoopsInterpModeOverride>, 5>
    fruit_loops_interp_mode_override_names{{
        {FruitLoopsInterpModeOverride::automatic, "auto"},
        {FruitLoopsInterpModeOverride::nearest, "nearest"},
        {FruitLoopsInterpModeOverride::bilinear, "bilinear"},
        {FruitLoopsInterpModeOverride::jinc, "jinc"},
        {FruitLoopsInterpModeOverride::trunc, "trunc"},
    }};

inline std::optional<TodType> parse_tod_type(std::string_view value) {
    return parse_enum(value, tod_type_names);
}

inline std::optional<TodOutputStream> parse_tod_output_stream(
    std::string_view value) {
    return parse_enum(value, tod_output_stream_names);
}

inline std::optional<TodOutputType> parse_tod_output_type(std::string_view value) {
    return parse_enum(value, tod_output_type_names);
}

inline std::optional<TodStreamOutputMode> parse_tod_stream_output_mode(
    std::string_view value) {
    return parse_enum(value, tod_stream_output_mode_names);
}

inline std::optional<TodOutputSelectionMode> parse_tod_output_selection_mode(
    std::string_view value) {
    return parse_enum(value, tod_output_selection_mode_names);
}

inline std::optional<RawTimeChunkFilterEdgeGuardMode>
parse_raw_filter_edge_guard_mode(std::string_view value) {
    return parse_enum(value, raw_filter_edge_guard_mode_names);
}

inline std::optional<RawTimeChunkFilterEdgeGuardCombine>
parse_raw_filter_edge_guard_combine(std::string_view value) {
    return parse_enum(value, raw_filter_edge_guard_combine_names);
}

inline std::optional<ProcessedTimeChunkWeightingType> parse_processed_weighting_type(
    std::string_view value) {
    return parse_enum(value, processed_weighting_type_names);
}

inline std::optional<ProcessedTimeChunkWeightGrouping>
parse_processed_weight_grouping(std::string_view value) {
    return parse_enum(value, processed_weight_grouping_names);
}

inline std::optional<ProcessedTimeChunkCleanerMode>
parse_processed_cleaner_mode(std::string_view value) {
    return parse_enum(value, processed_cleaner_mode_names);
}

inline std::optional<ProcessedTimeChunkCorrGroupingMetric>
parse_processed_corr_grouping_metric(std::string_view value) {
    return parse_enum(value, processed_corr_grouping_metric_names);
}

inline std::optional<FruitLoopsMode> parse_fruit_loops_mode(
    std::string_view value) {
    return parse_enum(value, fruit_loops_mode_names);
}

inline std::optional<FruitLoopsWeightFeedbackReference>
parse_fruit_loops_weight_feedback_reference(std::string_view value) {
    return parse_enum(value, fruit_loops_weight_feedback_reference_names);
}

inline std::optional<FruitLoopsInterpModeOverride>
parse_fruit_loops_interp_mode_override(std::string_view value) {
    return parse_enum(value, fruit_loops_interp_mode_override_names);
}

inline std::string_view to_string(TodType value) {
    return enum_name(value, tod_type_names);
}

inline std::string_view to_string(TodOutputStream value) {
    return enum_name(value, tod_output_stream_names);
}

inline std::string_view to_string(TodOutputType value) {
    return enum_name(value, tod_output_type_names);
}

inline std::string_view to_string(TodStreamOutputMode value) {
    return enum_name(value, tod_stream_output_mode_names);
}

inline std::string_view to_string(TodOutputSelectionMode value) {
    return enum_name(value, tod_output_selection_mode_names);
}

inline std::string_view to_string(RawTimeChunkFilterEdgeGuardMode value) {
    return enum_name(value, raw_filter_edge_guard_mode_names);
}

inline std::string_view to_string(RawTimeChunkFilterEdgeGuardCombine value) {
    return enum_name(value, raw_filter_edge_guard_combine_names);
}

inline std::string_view to_string(ProcessedTimeChunkWeightingType value) {
    return enum_name(value, processed_weighting_type_names);
}

inline std::string_view to_string(ProcessedTimeChunkWeightGrouping value) {
    return enum_name(value, processed_weight_grouping_names);
}

inline std::string_view to_string(ProcessedTimeChunkCleanerMode value) {
    return enum_name(value, processed_cleaner_mode_names);
}

inline std::string_view to_string(ProcessedTimeChunkCorrGroupingMetric value) {
    return enum_name(value, processed_corr_grouping_metric_names);
}

inline std::string_view to_string(FruitLoopsMode value) {
    return enum_name(value, fruit_loops_mode_names);
}

inline std::string_view to_string(FruitLoopsWeightFeedbackReference value) {
    return enum_name(value, fruit_loops_weight_feedback_reference_names);
}

inline std::string_view to_string(FruitLoopsInterpModeOverride value) {
    return enum_name(value, fruit_loops_interp_mode_override_names);
}

inline bool is_tod_output_stream(TodOutputStream value,
                                 TodOutputStream stream) {
    return value == stream;
}

inline bool is_rtc_tod_output_stream(TodOutputStream value) {
    return is_tod_output_stream(value, TodOutputStream::rtc);
}

inline bool is_ptc_tod_output_stream(TodOutputStream value) {
    return is_tod_output_stream(value, TodOutputStream::ptc);
}

inline bool is_tod_output_enabled(TodOutputType value) {
    return value != TodOutputType::none;
}

inline bool tod_output_includes_rtc(TodOutputType value) {
    return value == TodOutputType::rtc || value == TodOutputType::both;
}

inline bool tod_output_includes_ptc(TodOutputType value) {
    return value == TodOutputType::ptc || value == TodOutputType::both;
}

inline TodOutputType enabled_tod_output_type(bool raw_time_chunk_enabled,
                                             bool processed_time_chunk_enabled) {
    if (raw_time_chunk_enabled && processed_time_chunk_enabled) {
        return TodOutputType::both;
    }
    if (raw_time_chunk_enabled) {
        return TodOutputType::rtc;
    }
    if (processed_time_chunk_enabled) {
        return TodOutputType::ptc;
    }
    return TodOutputType::none;
}

inline bool is_mini_tod_stream_output_mode(TodStreamOutputMode value) {
    return value == TodStreamOutputMode::mini ||
           value == TodStreamOutputMode::mini_outer;
}

inline bool is_outer_tod_stream_output_mode(TodStreamOutputMode value) {
    return value == TodStreamOutputMode::full_outer ||
           value == TodStreamOutputMode::mini_outer;
}

inline bool is_indices_tod_output_selection_mode(
    TodOutputSelectionMode value) {
    return value == TodOutputSelectionMode::indices;
}

inline bool is_all_tod_output_selection_mode(TodOutputSelectionMode value) {
    return value == TodOutputSelectionMode::all;
}

inline bool is_uniform_source_tod_output_selection_mode(
    TodOutputSelectionMode value) {
    return value ==
           TodOutputSelectionMode::uniform_plus_source_crossing;
}

inline bool is_fruit_loops_interp_mode(
    std::string_view value, FruitLoopsInterpModeOverride mode) {
    return value == to_string(mode);
}

inline bool is_fruit_loops_auto_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::automatic);
}

inline bool is_fruit_loops_nearest_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::nearest);
}

inline bool is_fruit_loops_bilinear_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::bilinear);
}

inline bool is_fruit_loops_jinc_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::jinc);
}

inline bool is_fruit_loops_trunc_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::trunc);
}

struct TodStreamOutputConfig {
    bool enabled = false;
    TodStreamOutputMode mode = TodStreamOutputMode::full;
    int outer_context_samples = 0;
    bool chunk_select_enabled = false;
    std::vector<int> chunks_1based;
    TodOutputSelectionMode selection_mode = TodOutputSelectionMode::indices;
    int selection_n_uniform = 10;
    int selection_n_source_dense = 10;
};

struct TimestreamOutputConfig {
    bool raw_time_chunk_enabled = false;
    bool processed_time_chunk_enabled = false;
    TodOutputType type = TodOutputType::none;
    std::string subdir_name;
    bool write_eigenvalues = false;
    TodStreamOutputConfig raw_time_chunk;
    TodStreamOutputConfig processed_time_chunk;
};

struct TimestreamChunkingConfig {
    std::string mode;
    double value = 0.0;
    bool force = false;
};

struct TimestreamSourceProtectionConfig {
    bool enabled = true;
    bool active = false;
    double radius_arcsec = 20.0;
};

struct RawTimeChunkDespikeCompactRawGateConfig {
    bool enabled = true;
    double candidate_rel_sigma_scale = 1.0;
    double window_sec = 0.18;
    double half_peak_frac = 0.5;
    double max_width_sec = 0.18;
    double max_step_shift_z = 3.0;
};

struct RawTimeChunkDespikeCompactDeltaGateConfig {
    bool enabled = true;
    double window_sec = 0.12;
    double half_peak_frac = 0.5;
    double max_width_sec = 0.10;
    double max_step_shift_z = 3.0;
};

struct RawTimeChunkDespikeLocalResidualConfig {
    bool enabled = false;
    double window_sec = 0.25;
    double sigma_scale = 0.75;
    double delta_sigma_scale = 0.75;
    bool expand_with_filter = false;
    double event_padding_sec = 0.08;
    double high_score_event_override = 20.0;
    double max_added_flagged_fraction = 0.10;
    RawTimeChunkDespikeCompactRawGateConfig compact_raw_gate;
    RawTimeChunkDespikeCompactDeltaGateConfig compact_delta_gate;
};

struct RawTimeChunkDespikeConfig {
    bool enabled = false;
    double min_spike_sigma = 8.0;
    double time_constant_sec = 0.015;
    double window_size = 32.0;
    bool legacy_enabled = true;
    TimestreamSourceProtectionConfig source_protection;
    RawTimeChunkDespikeLocalResidualConfig local_residual;
};

struct RawTimeChunkDownsampleConfig {
    bool enabled = false;
    int factor = 1;
    double downsampled_freq_Hz = 0.0;
};

struct RawTimeChunkFilterNotchConfig {
    bool enabled = false;
    bool zero_phase = true;
    std::vector<double> freqs_Hz;
    std::vector<double> delta_f_Hz;
};

struct RawTimeChunkFilterEdgeGuardConfig {
    bool enabled = false;
    RawTimeChunkFilterEdgeGuardMode mode =
        RawTimeChunkFilterEdgeGuardMode::flag;
    RawTimeChunkFilterEdgeGuardCombine combine =
        RawTimeChunkFilterEdgeGuardCombine::sum;
    int min_samples = 0;
    int extra_samples = 0;
    int max_samples = 128;
    double iir_settle_attenuation = 0.01;
    bool apply_fir = true;
    bool apply_notch = true;
    bool apply_dynamic_notch = true;
    bool apply_iir_highpass = true;
    bool apply_downsample = true;
};

struct RawTimeChunkFilterConfig {
    bool enabled = false;
    double a_gibbs = 50.0;
    double freq_high_Hz = 16.0;
    double freq_low_Hz = 0.0;
    int n_terms = 32;
    RawTimeChunkFilterNotchConfig notch;
    RawTimeChunkFilterEdgeGuardConfig edge_guard;
};

struct RawTimeChunkIirFilterConfig {
    bool enabled = false;
    double freq_Hz = 0.1;
    int order = 1;
    bool zero_phase = false;
};

struct RawTimeChunkNetworkStepMaskConfig {
    bool enabled = false;
    double step_window_sec = 0.5;
    double step_score_thresh = 2.5;
    double min_good_frac = 0.8;
    int min_det_used = 32;
    double min_step_det_frac = 0.05;
    double min_alignment_frac = 0.5;
    double cluster_tol_sec = 0.25;
    double mask_half_width_sec = 0.5;
    double max_flagged_fraction = 0.30;
};

struct RawTimeChunkImpulsiveCaptureConfig {
    bool enabled = false;
    double min_good_frac = 0.8;
    double min_event_z = 6.0;
    double near_event_z = 4.0;
    int max_events_per_network = 3;
    double snippet_pre_window_sec = 0.25;
    double snippet_post_window_sec = 0.25;
};

struct RawTimeChunkImpulsiveCoincidenceConfig {
    bool enabled = false;
    double min_good_frac = 0.8;
    double event_score_thresh = 6.0;
    int min_det_used = 32;
    double min_impulsive_det_frac = 0.05;
    double min_alignment_frac = 0.5;
    int min_networks_aligned = 3;
    double high_score_override_thresh = 0.0;
    int high_score_min_networks_aligned = 0;
    double cluster_tol_sec = 0.03;
    double mask_pre_window_sec = 0.03;
    double mask_post_window_sec = 0.03;
    double max_flagged_fraction = 0.10;
};

struct RawTimeChunkFlaggingConfig {
    double delta_f_min_Hz = 60.e3;
    double lower_tod_inv_var_factor = 0.0;
    double upper_tod_inv_var_factor = 0.0;
    RawTimeChunkNetworkStepMaskConfig network_step_mask;
    RawTimeChunkImpulsiveCaptureConfig impulsive_capture;
    RawTimeChunkImpulsiveCoincidenceConfig impulsive_coincidence;
};

struct RawTimeChunkKernelConfig {
    bool enabled = false;
    std::string filepath;
    std::string type;
    double fwhm_arcsec = 0.0;
    std::vector<std::string> image_ext_names;
};

struct RawTimeChunkAltAzDestripeConfig {
    bool enabled = false;
    std::string grouping = "nw";
    bool fit_time_trend = true;
    bool fit_derivs = true;
    int min_samples = 64;
};

struct RawTimeChunkLineAuditConfig {
    bool enabled = false;
    double line_min_hz = 1.0;
    double line_max_hz = 60.0;
    double segment_sec = 4.0;
    double min_segment_sec = 2.0;
    double overlap_frac = 0.5;
    int continuum_radius_bins = 8;
    double prominence_thresh = 8.0;
    double cm_prominence_thresh = 6.0;
    double min_good_frac = 0.8;
    int min_windows = 2;
    int max_peaks_per_detector = 3;
    int max_det = 128;
    int min_det_for_network = 16;
    double cluster_tol_hz = 0.15;
    double notch_min_detector_frac = 0.10;
    int notch_min_detectors = 8;
    double notch_min_cm_prominence = 10.0;
    double detector_min_prominence = 12.0;
    double detector_min_line_power_frac = 0.10;
    double bad_detector_max_cluster_frac = 0.10;
    bool pre_filter_enabled = true;
    bool post_filter_enabled = false;
    bool post_filter_apply_shared_notches = false;
    bool post_filter_apply_detector_notches = false;
    int post_filter_apply_iterations = 1;
    double post_filter_line_min_hz =
        std::numeric_limits<double>::quiet_NaN();
    double post_filter_line_max_hz =
        std::numeric_limits<double>::quiet_NaN();
    bool ptc_model_protected_enabled = false;
    bool ptc_require_model_subtracted = true;
    bool ptc_apply_fixed_notches = false;
    bool ptc_apply_shared_notches = false;
    bool ptc_apply_detector_notches = false;
    int ptc_apply_iterations = 1;
    double ptc_line_min_hz = std::numeric_limits<double>::quiet_NaN();
    double ptc_line_max_hz = std::numeric_limits<double>::quiet_NaN();
    bool fixed_notch_enabled = false;
    std::vector<double> fixed_notch_freqs_hz;
    std::vector<double> fixed_notch_widths_hz{0.25};
    double fixed_notch_exclusion_half_width_hz = 0.25;
    bool apply_shared_notches = false;
    int apply_min_support_networks = 2;
    double apply_min_detector_frac = 0.90;
    double apply_min_common_mode_prominence = 150.0;
    double apply_width_scale = 1.5;
    double apply_min_width_hz = 0.25;
    double apply_max_width_hz = 1.50;
    int apply_max_notches = 3;
    double apply_cluster_tol_hz = 0.25;
    double detector_notch_min_prominence = 8.0;
    double detector_notch_min_line_power_frac = 0.0;
    int detector_notch_max_notches = 3;
    double detector_notch_width_scale = 1.0;
    double detector_notch_min_width_hz = 0.25;
    double detector_notch_max_width_hz = 1.50;
    int detector_notch_context_samples = 0;
};

struct RawTimeChunkConfig {
    RawTimeChunkDespikeConfig despike;
    RawTimeChunkDownsampleConfig downsample;
    RawTimeChunkFilterConfig filter;
    RawTimeChunkIirFilterConfig iir_filter;
    RawTimeChunkFlaggingConfig flagging;
    RawTimeChunkKernelConfig kernel;
    RawTimeChunkAltAzDestripeConfig altaz_destripe;
    RawTimeChunkLineAuditConfig line_audit;
    bool flux_calibration_enabled = false;
    bool extinction_correction_enabled = false;
};

struct ProcessedTimeChunkSecondPassLocalConfig {
    bool enabled = false;
    double min_spike_sigma = 8.0;
    double min_good_frac = 0.5;
    double baseline_window_sec = 0.25;
    double sigma_scale = 0.75;
    double delta_sigma_scale = 0.75;
    double raw_candidate_rel_sigma_scale = 1.0;
    double raw_window_sec = 0.18;
    double raw_half_peak_frac = 0.5;
    double raw_max_width_sec = 0.18;
    double delta_window_sec = 0.12;
    double delta_half_peak_frac = 0.5;
    double delta_max_width_sec = 0.10;
    double max_step_shift_z = 3.0;
    double high_score_event_override = 20.0;
    double merge_within_detector_sec = 0.08;
    double cluster_events_sec = 0.08;
    int min_cluster_detectors = 3;
    double high_score_cluster_override = 9.0;
    int max_auto_flag_clusters_per_network = 3;
    bool selective_busy_network_acceptance_enabled = true;
    TimestreamSourceProtectionConfig source_protection;
};

struct ProcessedTimeChunkStandardPcaConfig {
    bool enabled = true;
    double stddev_limit = 0.0;
    int n_calc = 64;
    std::map<std::string, std::vector<int>> n_eig_to_cut;
};

struct ProcessedTimeChunkCorrGroupingConfig {
    bool enabled = false;
    ProcessedTimeChunkCorrGroupingMetric metric =
        ProcessedTimeChunkCorrGroupingMetric::abs;
    double corr_min = 0.6;
    int min_overlap = 300;
    double min_good_frac = 0.8;
    int min_group_size = 10;
    int max_samples = 20000;
    bool clean_residual = true;
};

struct ProcessedTimeChunkNullModelConfig {
    bool enabled = false;
    int n_surrogates = 16;
    double quantile = 0.99;
    double min_good_frac = 0.8;
    int max_modes = 64;
    int max_samples = 20000;
    int seed = 12345;
    std::vector<std::string> grouping;
};

struct ProcessedTimeChunkMarchenkoPasturConfig {
    bool enabled = false;
    double min_good_frac = 0.8;
    int max_modes = 64;
    int max_samples = 20000;
    double band_low_Hz = 0.0;
    double band_high_Hz = 0.0;
    double clip_z = 12.0;
    double bulk_keep_frac = 0.8;
    int q_grid_size = 64;
    std::vector<std::string> grouping;
};

struct ProcessedTimeChunkAdaptiveSelectorConfig {
    bool enabled = false;
    double min_good_frac = 0.7;
    int max_det = 120;
    int max_samples = 1024;
    int max_pairs = 2000;
    int seed = 12345;
    double clip_z = 50.0;
    double low_weight = 1.0;
    double tail_weight = 0.0;
    double topmode_weight = 0.1;
    double reg_weight = 0.3;
    std::array<double, 2> low_band_Hz{0.05, 0.5};
    std::array<double, 2> mid_band_Hz{0.5, 2.0};
    std::vector<int> candidate_offsets{-2, 0, 2, 4};
    std::vector<std::string> grouping;
    bool log_candidates = false;
};

struct ProcessedTimeChunkCleanConfig {
    bool enabled = false;
    ProcessedTimeChunkCleanerMode active = ProcessedTimeChunkCleanerMode::none;
    std::vector<std::string> grouping;
    double mask_radius_arcsec = 0.0;
    double tau = 0.0;
    ProcessedTimeChunkStandardPcaConfig standard_pca;
    ProcessedTimeChunkCorrGroupingConfig corr_grouping;
    ProcessedTimeChunkNullModelConfig null_model;
    ProcessedTimeChunkMarchenkoPasturConfig marchenko_pastur;
    ProcessedTimeChunkAdaptiveSelectorConfig adaptive_selector;
};

struct ProcessedTimeChunkBusyRowSuppressionConfig {
    bool enabled = false;
    bool require_busy_veto = true;
    int min_candidate_clusters = 5;
    double min_max_unflagged_residual_z = 25.0;
    double factor = 0.0;
};

struct ProcessedTimeChunkWeightValidationConfig {
    bool enabled = false;
    int accumulation_iters = 1;
    int apply_start_iter = 1;
    int min_valid_scans = 1;
    double min_factor = 0.1;
    double unvalidated_factor = 1.0;
    bool require_fruitloops_model = true;
    bool transient_ratio_enabled = false;
    double ratio_power = 1.0;
    double transient_ratio_power = 1.0;
    bool upward_enabled = false;
    double upward_max_factor = 1.10;
    double upward_power = 1.0;
    double upward_min_base_factor = 0.95;
    bool upward_require_atmospheric = true;
    double upward_min_atmospheric_factor = 0.9;
    bool atmospheric_correlation_enabled = true;
    ProcessedTimeChunkWeightGrouping atmospheric_grouping =
        ProcessedTimeChunkWeightGrouping::array;
    int atmospheric_min_detectors = 8;
    double atmospheric_ref = 0.0;
    double atmospheric_span = 0.15;
    double atmospheric_power = 1.0;
    double min_good_frac = 0.5;
    int min_overlap = 200;
    int max_samples = 5000;
    bool high_weight_validation_enabled = true;
    bool high_weight_apply_caps = true;
    ProcessedTimeChunkWeightGrouping high_weight_grouping =
        ProcessedTimeChunkWeightGrouping::array;
    int high_weight_min_group_detectors = 20;
    double high_weight_log_robust_z = 6.0;
    double high_weight_max_median_factor = 8.0;
    double high_weight_cap_median_factor = 4.0;
    double high_weight_min_validated_factor = 0.95;
};

struct ProcessedTimeChunkWeightCorrPenaltyTermConfig {
    bool enabled = true;
    double ref = 0.05;
    double span = 0.15;
    double weight = 1.0;
};

struct ProcessedTimeChunkWeightCorrPenaltyBandConfig {
    bool enabled = false;
    double ref = 0.6;
    double span = 2.0;
    double weight = 0.5;
    std::array<double, 2> low_band_Hz{0.05, 0.5};
    std::array<double, 2> mid_band_Hz{0.5, 2.0};
};

struct ProcessedTimeChunkWeightCorrPenaltyConfig {
    bool enabled = false;
    double min_good_frac = 0.7;
    int min_overlap = 200;
    int max_samples = 20000;
    int max_pairs = 4000;
    int seed = 12345;
    double floor = 0.05;
    double exponent = 2.0;
    ProcessedTimeChunkWeightCorrPenaltyTermConfig pair_corr;
    ProcessedTimeChunkWeightCorrPenaltyTermConfig cm_el_corr{
        false, 0.05, 0.25, 0.5};
    ProcessedTimeChunkWeightCorrPenaltyBandConfig cm_low_mid_ratio;
};

struct ProcessedTimeChunkWeightingConfig {
    ProcessedTimeChunkWeightingType type =
        ProcessedTimeChunkWeightingType::full;
    double source_mask_radius_arcsec = 0.0;
    double hybrid_correction_min_factor = 0.5;
    double hybrid_correction_max_factor = 2.0;
    double median_map_weight_factor = 0.0;
    double lower_map_weight_factor = 0.0;
    double upper_map_weight_factor = 0.0;
    ProcessedTimeChunkWeightValidationConfig validation;
    ProcessedTimeChunkWeightCorrPenaltyConfig corr_penalty;
    ProcessedTimeChunkBusyRowSuppressionConfig busy_row_suppression;
};

struct ProcessedTimeChunkFlaggingConfig {
    double lower_tod_inv_var_factor = 0.0;
    double upper_tod_inv_var_factor = 0.0;
    ProcessedTimeChunkSecondPassLocalConfig second_pass_local;
};

struct ProcessedTimeChunkConfig {
    ProcessedTimeChunkCleanConfig clean;
    ProcessedTimeChunkWeightingConfig weighting;
    ProcessedTimeChunkFlaggingConfig flagging;
};

struct FruitLoopsWeightFeedbackConfig {
    bool enabled = false;
    FruitLoopsWeightFeedbackReference reference =
        FruitLoopsWeightFeedbackReference::p95;
    double low_relative_weight = 0.02;
    double high_relative_weight = 0.10;
};

struct TimestreamFruitLoopsConfig {
    bool enabled = false;
    bool save_all_iters = false;
    std::string path;
    std::string type;
    FruitLoopsMode mode = FruitLoopsMode::upper;
    double sig2noise_limit = 0.0;
    std::vector<double> array_flux_limit;
    double peak_fraction_limit = 0.0;
    double local_snr_floor = 0.0;
    double local_sigma_inner_radius_arcsec = 10.0;
    double local_sigma_outer_radius_arcsec = 35.0;
    double local_sigma_inner_fwhm = 1.5;
    double local_sigma_outer_fwhm = 4.0;
    double local_sigma_edge_guard_arcsec = 5.0;
    int local_sigma_min_pixels = 50;
    double adaptive_support_radius_arcsec = 12.0;
    double adaptive_support_radius_fwhm = 1.5;
    FruitLoopsWeightFeedbackConfig weight_feedback;
    double center_keep_radius_arcsec = 0.0;
    FruitLoopsInterpModeOverride interp_mode_override =
        FruitLoopsInterpModeOverride::automatic;
    bool legacy_center = false;
    bool recompute_weights_after_addback = false;
    int max_iters = 1;
};

struct TimestreamLearningMapPixelOutlierConfig {
    bool diagnostics_enabled = true;
    bool contributor_diagnostics_enabled = false;
    bool targeted_contributor_diagnostics_enabled = false;
    bool detector_exclusion_enabled = false;
    int top_n = 8;
    int targeted_contributor_max_pixels = 32;
    int detector_exclusion_min_pixels = 4;
    double min_abs_z = 8.0;
    double min_n_eff = 4.0;
    double source_radius_arcsec = 30.0;
};

struct TimestreamLearningBusyDetectorConfig {
    bool exclusion_enabled = true;
};

struct TimestreamLearningScanNetworkPathologyConfig {
    bool enabled = true;
    bool apply_pre_rtc = false;
    bool apply_pre_ptc = false;
    bool apply_pre_mapmaking = true;
    int min_candidate_clusters = 4;
    int min_candidate_events = 100;
    double min_max_residual_z = 25.0;
    int severe_candidate_events = 250;
    double severe_max_residual_z = 50.0;
    double max_new_flagged_fraction = 0.35;
};

struct TimestreamLearningConfig {
    bool enabled = false;
    bool diagnostics_enabled = true;
    int learn_iters = 2;
    int apply_start_iter = 2;
    int max_records_per_type = 200000;
    bool apply_sample_masks_enabled = true;
    double apply_max_new_flagged_fraction = 0.02;
    TimestreamLearningMapPixelOutlierConfig map_pixel_outlier;
    TimestreamLearningBusyDetectorConfig busy_detector;
    TimestreamLearningScanNetworkPathologyConfig scan_network_pathology;
};

struct TimestreamConfig {
    bool enabled = true;
    TodType type = TodType::xs;
    TimestreamOutputConfig output;
    TimestreamChunkingConfig chunking;
    RawTimeChunkConfig raw_time_chunk;
    ProcessedTimeChunkConfig processed_time_chunk;
    TimestreamFruitLoopsConfig fruit_loops;
    TimestreamLearningConfig learning;
};

inline ConfigPath append_config_path(ConfigPath path,
                                     std::initializer_list<std::string> suffix) {
    path.insert(path.end(), suffix.begin(), suffix.end());
    return path;
}

inline void validate(const TodStreamOutputConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.outer_context_samples, 0,
                  append_config_path(path, {"outer_context_samples"}), report);
    check_minimum(config.selection_n_uniform, 0,
                  append_config_path(path, {"selection", "n_uniform"}), report);
    check_minimum(config.selection_n_source_dense, 0,
                  append_config_path(path, {"selection", "n_source_dense"}), report);
    if (config.selection_mode == TodOutputSelectionMode::uniform_plus_source_crossing &&
        config.selection_n_uniform + config.selection_n_source_dense <= 0) {
        report.add_error(append_config_path(path, {"selection"}),
                         "uniform_plus_source_crossing requires at least one selected chunk");
    }
    for (const auto chunk : config.chunks_1based) {
        check_minimum(chunk, 1,
                      append_config_path(path, {"indices"}), report);
    }
}

inline void validate(const TimestreamChunkingConfig &config,
                     ValidationReport &report) {
    check_minimum(config.value, 0.0, {"timestream", "chunking", "value"}, report);
}

inline void validate(const TimestreamSourceProtectionConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.radius_arcsec, 0.0,
                  append_config_path(path, {"radius_arcsec"}), report);
}

inline void validate(const RawTimeChunkDespikeCompactRawGateConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.candidate_rel_sigma_scale, 0.0,
                  append_config_path(path, {"candidate_rel_sigma_scale"}),
                  report);
    check_minimum(config.window_sec, 0.0,
                  append_config_path(path, {"window_sec"}), report);
    check_minimum(config.half_peak_frac, 0.0,
                  append_config_path(path, {"half_peak_frac"}), report);
    check_maximum(config.half_peak_frac, 1.0,
                  append_config_path(path, {"half_peak_frac"}), report);
    check_minimum(config.max_width_sec, 0.0,
                  append_config_path(path, {"max_width_sec"}), report);
    check_minimum(config.max_step_shift_z, 0.0,
                  append_config_path(path, {"max_step_shift_z"}), report);
}

inline void validate(const RawTimeChunkDespikeCompactDeltaGateConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.window_sec, 0.0,
                  append_config_path(path, {"window_sec"}), report);
    check_minimum(config.half_peak_frac, 0.0,
                  append_config_path(path, {"half_peak_frac"}), report);
    check_maximum(config.half_peak_frac, 1.0,
                  append_config_path(path, {"half_peak_frac"}), report);
    check_minimum(config.max_width_sec, 0.0,
                  append_config_path(path, {"max_width_sec"}), report);
    check_minimum(config.max_step_shift_z, 0.0,
                  append_config_path(path, {"max_step_shift_z"}), report);
}

inline void validate(const RawTimeChunkDespikeLocalResidualConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{
        "timestream", "raw_time_chunk", "despike", "local_residual"};
    check_minimum(config.window_sec, 0.0,
                  append_config_path(path, {"window_sec"}), report);
    check_minimum(config.sigma_scale, 0.0,
                  append_config_path(path, {"sigma_scale"}), report);
    check_minimum(config.delta_sigma_scale, 0.0,
                  append_config_path(path, {"delta_sigma_scale"}), report);
    check_minimum(config.event_padding_sec, 0.0,
                  append_config_path(path, {"event_padding_sec"}), report);
    check_minimum(config.high_score_event_override, 0.0,
                  append_config_path(path, {"high_score_event_override"}),
                  report);
    check_minimum(config.max_added_flagged_fraction, 0.0,
                  append_config_path(path, {"max_added_flagged_fraction"}),
                  report);
    check_maximum(config.max_added_flagged_fraction, 1.0,
                  append_config_path(path, {"max_added_flagged_fraction"}),
                  report);
    validate(config.compact_raw_gate,
             append_config_path(path, {"compact_raw_gate"}), report);
    validate(config.compact_delta_gate,
             append_config_path(path, {"compact_delta_gate"}), report);
}

inline void validate(const RawTimeChunkDespikeConfig &config,
                     ValidationReport &report) {
    validate(config.source_protection,
             {"timestream", "raw_time_chunk", "despike", "source_protection"},
             report);
    validate(config.local_residual, report);
}

inline void validate(const RawTimeChunkDownsampleConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{"timestream", "raw_time_chunk", "downsample"};
    check_minimum(config.factor, 0, append_config_path(path, {"factor"}),
                  report);
    check_minimum(config.downsampled_freq_Hz, 0.0,
                  append_config_path(path, {"downsampled_freq_Hz"}), report);
}

inline void validate(const RawTimeChunkFilterNotchConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{"timestream", "raw_time_chunk", "filter", "notch"};
    if (!config.zero_phase) {
        report.add_error(append_config_path(path, {"zero_phase"}),
                         "must be true to avoid phase shifts");
    }
    if (config.freqs_Hz.empty()) {
        report.add_error(append_config_path(path, {"freqs_Hz"}),
                         "must contain at least one notch frequency");
    }
    if (config.delta_f_Hz.size() != 1 &&
        config.delta_f_Hz.size() != config.freqs_Hz.size()) {
        report.add_error(append_config_path(path, {"delta_f_Hz"}),
                         "must have length 1 or match freqs_Hz length");
    }
    for (const auto freq_Hz : config.freqs_Hz) {
        check_minimum(freq_Hz, 1e-12, append_config_path(path, {"freqs_Hz"}),
                      report);
    }
    for (const auto delta_f_Hz : config.delta_f_Hz) {
        check_minimum(delta_f_Hz, 1e-12,
                      append_config_path(path, {"delta_f_Hz"}), report);
    }
}

inline void validate(const RawTimeChunkFilterEdgeGuardConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "raw_time_chunk", "filter", "edge_guard"};
    check_minimum(config.min_samples, 0,
                  append_config_path(path, {"min_samples"}), report);
    check_minimum(config.extra_samples, 0,
                  append_config_path(path, {"extra_samples"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.iir_settle_attenuation, 0.0,
                  append_config_path(path, {"iir_settle_attenuation"}), report);
    check_maximum(config.iir_settle_attenuation, 1.0,
                  append_config_path(path, {"iir_settle_attenuation"}), report);
}

inline void validate(const RawTimeChunkFilterConfig &config,
                     ValidationReport &report) {
    if (config.enabled) {
        const ConfigPath path{"timestream", "raw_time_chunk", "filter"};
        check_minimum(config.a_gibbs, 0.0,
                      append_config_path(path, {"a_gibbs"}), report);
        check_minimum(config.freq_low_Hz, 0.0,
                      append_config_path(path, {"freq_low_Hz"}), report);
        check_minimum(config.freq_high_Hz, 0.0,
                      append_config_path(path, {"freq_high_Hz"}), report);
        if (config.freq_high_Hz < config.freq_low_Hz) {
            report.add_error(append_config_path(path, {"freq_high_Hz"}),
                             "must be greater than or equal to freq_low_Hz");
        }
        check_minimum(config.n_terms, 0, append_config_path(path, {"n_terms"}),
                      report);
        validate(config.notch, report);
    }
    validate(config.edge_guard, report);
}

inline void validate(const RawTimeChunkIirFilterConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{"timestream", "raw_time_chunk", "IIR_filter"};
    check_minimum(config.freq_Hz, 1e-12,
                  append_config_path(path, {"freq_Hz"}), report);
    check_minimum(config.order, 1, append_config_path(path, {"order"}), report);
    if (!config.zero_phase) {
        report.add_error(append_config_path(path, {"zero_phase"}),
                         "must be true to avoid phase shifts");
    }
}

inline void validate(const RawTimeChunkNetworkStepMaskConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "raw_time_chunk", "flagging", "network_step_mask"};
    check_minimum(config.step_window_sec, 0.01,
                  append_config_path(path, {"step_window_sec"}), report);
    check_minimum(config.step_score_thresh, 0.0,
                  append_config_path(path, {"step_score_thresh"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_det_used, 1,
                  append_config_path(path, {"min_det_used"}), report);
    check_minimum(config.min_step_det_frac, 0.0,
                  append_config_path(path, {"min_step_det_frac"}), report);
    check_maximum(config.min_step_det_frac, 1.0,
                  append_config_path(path, {"min_step_det_frac"}), report);
    check_minimum(config.min_alignment_frac, 0.0,
                  append_config_path(path, {"min_alignment_frac"}), report);
    check_maximum(config.min_alignment_frac, 1.0,
                  append_config_path(path, {"min_alignment_frac"}), report);
    check_minimum(config.cluster_tol_sec, 0.0,
                  append_config_path(path, {"cluster_tol_sec"}), report);
    check_minimum(config.mask_half_width_sec, 0.0,
                  append_config_path(path, {"mask_half_width_sec"}), report);
    check_minimum(config.max_flagged_fraction, 0.0,
                  append_config_path(path, {"max_flagged_fraction"}), report);
    check_maximum(config.max_flagged_fraction, 1.0,
                  append_config_path(path, {"max_flagged_fraction"}), report);
}

inline void validate(const RawTimeChunkImpulsiveCaptureConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "raw_time_chunk", "flagging", "impulsive_capture"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_event_z, 0.0,
                  append_config_path(path, {"min_event_z"}), report);
    check_minimum(config.near_event_z, 0.0,
                  append_config_path(path, {"near_event_z"}), report);
    check_minimum(config.max_events_per_network, 1,
                  append_config_path(path, {"max_events_per_network"}), report);
    check_minimum(config.snippet_pre_window_sec, 0.0,
                  append_config_path(path, {"snippet_pre_window_sec"}), report);
    check_minimum(config.snippet_post_window_sec, 0.0,
                  append_config_path(path, {"snippet_post_window_sec"}), report);
}

inline void validate(const RawTimeChunkImpulsiveCoincidenceConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "raw_time_chunk", "flagging",
        "impulsive_coincidence"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.event_score_thresh, 0.0,
                  append_config_path(path, {"event_score_thresh"}), report);
    check_minimum(config.min_det_used, 1,
                  append_config_path(path, {"min_det_used"}), report);
    check_minimum(config.min_impulsive_det_frac, 0.0,
                  append_config_path(path, {"min_impulsive_det_frac"}), report);
    check_maximum(config.min_impulsive_det_frac, 1.0,
                  append_config_path(path, {"min_impulsive_det_frac"}), report);
    check_minimum(config.min_alignment_frac, 0.0,
                  append_config_path(path, {"min_alignment_frac"}), report);
    check_maximum(config.min_alignment_frac, 1.0,
                  append_config_path(path, {"min_alignment_frac"}), report);
    check_minimum(config.min_networks_aligned, 1,
                  append_config_path(path, {"min_networks_aligned"}), report);
    check_minimum(config.high_score_override_thresh, 0.0,
                  append_config_path(path, {"high_score_override_thresh"}),
                  report);
    check_minimum(config.high_score_min_networks_aligned, 0,
                  append_config_path(path,
                                     {"high_score_min_networks_aligned"}),
                  report);
    check_minimum(config.cluster_tol_sec, 0.0,
                  append_config_path(path, {"cluster_tol_sec"}), report);
    check_minimum(config.mask_pre_window_sec, 0.0,
                  append_config_path(path, {"mask_pre_window_sec"}), report);
    check_minimum(config.mask_post_window_sec, 0.0,
                  append_config_path(path, {"mask_post_window_sec"}), report);
    check_minimum(config.max_flagged_fraction, 0.0,
                  append_config_path(path, {"max_flagged_fraction"}), report);
    check_maximum(config.max_flagged_fraction, 1.0,
                  append_config_path(path, {"max_flagged_fraction"}), report);
}

inline void validate(const RawTimeChunkFlaggingConfig &config,
                     ValidationReport &report) {
    validate(config.network_step_mask, report);
    validate(config.impulsive_capture, report);
    validate(config.impulsive_coincidence, report);
}

inline void validate(const RawTimeChunkKernelConfig &, ValidationReport &) {}

inline void validate(const RawTimeChunkAltAzDestripeConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    check_minimum(config.min_samples, 4,
                  {"timestream", "raw_time_chunk", "altaz_destripe",
                   "min_samples"},
                  report);
}

inline void validate_optional_minimum(const double value, const double minimum,
                                      const ConfigPath &path,
                                      ValidationReport &report) {
    if (std::isfinite(value)) {
        check_minimum(value, minimum, path, report);
    }
}

inline void validate(const RawTimeChunkLineAuditConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{"timestream", "raw_time_chunk", "line_audit"};
    check_minimum(config.line_min_hz, 0.0,
                  append_config_path(path, {"line_min_hz"}), report);
    check_minimum(config.line_max_hz, 0.0,
                  append_config_path(path, {"line_max_hz"}), report);
    check_minimum(config.segment_sec, 0.1,
                  append_config_path(path, {"segment_sec"}), report);
    check_minimum(config.min_segment_sec, 0.1,
                  append_config_path(path, {"min_segment_sec"}), report);
    check_minimum(config.overlap_frac, 0.0,
                  append_config_path(path, {"overlap_frac"}), report);
    check_maximum(config.overlap_frac, 0.95,
                  append_config_path(path, {"overlap_frac"}), report);
    check_minimum(config.continuum_radius_bins, 1,
                  append_config_path(path, {"continuum_radius_bins"}), report);
    check_minimum(config.prominence_thresh, 1.0,
                  append_config_path(path, {"prominence_thresh"}), report);
    check_minimum(config.cm_prominence_thresh, 1.0,
                  append_config_path(path, {"cm_prominence_thresh"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_windows, 1,
                  append_config_path(path, {"min_windows"}), report);
    check_minimum(config.max_peaks_per_detector, 1,
                  append_config_path(path, {"max_peaks_per_detector"}), report);
    check_minimum(config.max_det, 0, append_config_path(path, {"max_det"}),
                  report);
    check_minimum(config.min_det_for_network, 1,
                  append_config_path(path, {"min_det_for_network"}), report);
    check_minimum(config.cluster_tol_hz, 0.0,
                  append_config_path(path, {"cluster_tol_hz"}), report);
    check_minimum(config.notch_min_detector_frac, 0.0,
                  append_config_path(path, {"notch_min_detector_frac"}),
                  report);
    check_maximum(config.notch_min_detector_frac, 1.0,
                  append_config_path(path, {"notch_min_detector_frac"}),
                  report);
    check_minimum(config.notch_min_detectors, 1,
                  append_config_path(path, {"notch_min_detectors"}), report);
    check_minimum(config.notch_min_cm_prominence, 1.0,
                  append_config_path(path, {"notch_min_cm_prominence"}),
                  report);
    check_minimum(config.detector_min_prominence, 1.0,
                  append_config_path(path, {"detector_min_prominence"}),
                  report);
    check_minimum(config.detector_min_line_power_frac, 0.0,
                  append_config_path(path, {"detector_min_line_power_frac"}),
                  report);
    check_maximum(config.detector_min_line_power_frac, 1.0,
                  append_config_path(path, {"detector_min_line_power_frac"}),
                  report);
    check_minimum(config.bad_detector_max_cluster_frac, 0.0,
                  append_config_path(path, {"bad_detector_max_cluster_frac"}),
                  report);
    check_maximum(config.bad_detector_max_cluster_frac, 1.0,
                  append_config_path(path, {"bad_detector_max_cluster_frac"}),
                  report);
    check_minimum(config.post_filter_apply_iterations, 1,
                  append_config_path(path, {"post_filter_apply_iterations"}),
                  report);
    validate_optional_minimum(config.post_filter_line_min_hz, 0.0,
                              append_config_path(path,
                                                 {"post_filter_line_min_hz"}),
                              report);
    validate_optional_minimum(config.post_filter_line_max_hz, 0.0,
                              append_config_path(path,
                                                 {"post_filter_line_max_hz"}),
                              report);
    check_minimum(config.ptc_apply_iterations, 1,
                  append_config_path(path, {"ptc_apply_iterations"}), report);
    validate_optional_minimum(config.ptc_line_min_hz, 0.0,
                              append_config_path(path, {"ptc_line_min_hz"}),
                              report);
    validate_optional_minimum(config.ptc_line_max_hz, 0.0,
                              append_config_path(path, {"ptc_line_max_hz"}),
                              report);
    if (std::isfinite(config.ptc_line_min_hz) &&
        std::isfinite(config.ptc_line_max_hz) &&
        config.ptc_line_max_hz < config.ptc_line_min_hz) {
        report.add_error(append_config_path(path, {"ptc_line_max_hz"}),
                         "must be greater than or equal to ptc_line_min_hz");
    }
    if (config.fixed_notch_enabled) {
        if (config.fixed_notch_freqs_hz.empty()) {
            report.add_error(append_config_path(path, {"fixed_notch_freqs_hz"}),
                             "must contain at least one fixed notch when enabled");
        }
        if (config.fixed_notch_widths_hz.empty()) {
            report.add_error(append_config_path(path, {"fixed_notch_widths_hz"}),
                             "must contain at least one fixed notch width");
        }
        if (!config.fixed_notch_widths_hz.empty() &&
            config.fixed_notch_widths_hz.size() != 1 &&
            config.fixed_notch_widths_hz.size() !=
                config.fixed_notch_freqs_hz.size()) {
            report.add_error(append_config_path(path, {"fixed_notch_widths_hz"}),
                             "must have length 1 or match fixed_notch_freqs_hz");
        }
        for (const auto freq_hz : config.fixed_notch_freqs_hz) {
            if (!std::isfinite(freq_hz) || freq_hz <= 0.0) {
                report.add_error(
                    append_config_path(path, {"fixed_notch_freqs_hz"}),
                    "values must be finite and greater than 0");
            }
        }
        for (const auto width_hz : config.fixed_notch_widths_hz) {
            if (!std::isfinite(width_hz) || width_hz <= 0.0) {
                report.add_error(
                    append_config_path(path, {"fixed_notch_widths_hz"}),
                    "values must be finite and greater than 0");
            }
        }
    }
    check_minimum(config.fixed_notch_exclusion_half_width_hz, 0.0,
                  append_config_path(path,
                                     {"fixed_notch_exclusion_half_width_hz"}),
                  report);
    check_minimum(config.apply_min_support_networks, 1,
                  append_config_path(path, {"apply_min_support_networks"}),
                  report);
    check_minimum(config.apply_min_detector_frac, 0.0,
                  append_config_path(path, {"apply_min_detector_frac"}), report);
    check_maximum(config.apply_min_detector_frac, 1.0,
                  append_config_path(path, {"apply_min_detector_frac"}), report);
    check_minimum(config.apply_min_common_mode_prominence, 1.0,
                  append_config_path(path,
                                     {"apply_min_common_mode_prominence"}),
                  report);
    check_minimum(config.apply_width_scale, 0.01,
                  append_config_path(path, {"apply_width_scale"}), report);
    check_minimum(config.apply_min_width_hz, 0.0,
                  append_config_path(path, {"apply_min_width_hz"}), report);
    check_minimum(config.apply_max_width_hz, 0.0,
                  append_config_path(path, {"apply_max_width_hz"}), report);
    if (config.apply_max_width_hz < config.apply_min_width_hz) {
        report.add_error(append_config_path(path, {"apply_max_width_hz"}),
                         "must be greater than or equal to apply_min_width_hz");
    }
    check_minimum(config.apply_max_notches, 0,
                  append_config_path(path, {"apply_max_notches"}), report);
    check_minimum(config.apply_cluster_tol_hz, 0.0,
                  append_config_path(path, {"apply_cluster_tol_hz"}), report);
    check_minimum(config.detector_notch_min_prominence, 1.0,
                  append_config_path(path, {"detector_notch_min_prominence"}),
                  report);
    check_minimum(config.detector_notch_min_line_power_frac, 0.0,
                  append_config_path(path,
                                     {"detector_notch_min_line_power_frac"}),
                  report);
    check_maximum(config.detector_notch_min_line_power_frac, 1.0,
                  append_config_path(path,
                                     {"detector_notch_min_line_power_frac"}),
                  report);
    check_minimum(config.detector_notch_max_notches, 0,
                  append_config_path(path, {"detector_notch_max_notches"}),
                  report);
    check_minimum(config.detector_notch_width_scale, 0.01,
                  append_config_path(path, {"detector_notch_width_scale"}),
                  report);
    check_minimum(config.detector_notch_min_width_hz, 0.0,
                  append_config_path(path, {"detector_notch_min_width_hz"}),
                  report);
    check_minimum(config.detector_notch_max_width_hz, 0.0,
                  append_config_path(path, {"detector_notch_max_width_hz"}),
                  report);
    if (config.detector_notch_max_width_hz <
        config.detector_notch_min_width_hz) {
        report.add_error(
            append_config_path(path, {"detector_notch_max_width_hz"}),
            "must be greater than or equal to detector_notch_min_width_hz");
    }
    check_minimum(config.detector_notch_context_samples, 0,
                  append_config_path(path, {"detector_notch_context_samples"}),
                  report);
}

inline void validate(const RawTimeChunkConfig &config, ValidationReport &report) {
    validate(config.despike, report);
    validate(config.downsample, report);
    validate(config.filter, report);
    validate(config.iir_filter, report);
    validate(config.flagging, report);
    validate(config.kernel, report);
    validate(config.altaz_destripe, report);
    validate(config.line_audit, report);
    if (config.downsample.enabled && !config.filter.enabled) {
        report.add_error({"timestream", "raw_time_chunk", "downsample"},
                         "requires raw_time_chunk.filter.enabled=true");
    }
}

inline void validate(const ProcessedTimeChunkSecondPassLocalConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{
        "timestream", "processed_time_chunk", "flagging", "second_pass_local"};
    check_minimum(config.min_spike_sigma, 0.0,
                  append_config_path(path, {"min_spike_sigma"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.baseline_window_sec, 0.0,
                  append_config_path(path, {"baseline_window_sec"}), report);
    check_minimum(config.sigma_scale, 0.0,
                  append_config_path(path, {"sigma_scale"}), report);
    check_minimum(config.delta_sigma_scale, 0.0,
                  append_config_path(path, {"delta_sigma_scale"}), report);
    check_minimum(config.raw_candidate_rel_sigma_scale, 0.0,
                  append_config_path(path, {"raw_candidate_rel_sigma_scale"}),
                  report);
    check_minimum(config.raw_window_sec, 0.0,
                  append_config_path(path, {"raw_window_sec"}), report);
    check_minimum(config.raw_half_peak_frac, 0.0,
                  append_config_path(path, {"raw_half_peak_frac"}), report);
    check_minimum(config.raw_max_width_sec, 0.0,
                  append_config_path(path, {"raw_max_width_sec"}), report);
    check_minimum(config.delta_window_sec, 0.0,
                  append_config_path(path, {"delta_window_sec"}), report);
    check_minimum(config.delta_half_peak_frac, 0.0,
                  append_config_path(path, {"delta_half_peak_frac"}), report);
    check_minimum(config.delta_max_width_sec, 0.0,
                  append_config_path(path, {"delta_max_width_sec"}), report);
    check_minimum(config.max_step_shift_z, 0.0,
                  append_config_path(path, {"max_step_shift_z"}), report);
    check_minimum(config.high_score_event_override, 0.0,
                  append_config_path(path, {"high_score_event_override"}), report);
    check_minimum(config.merge_within_detector_sec, 0.0,
                  append_config_path(path, {"merge_within_detector_sec"}), report);
    check_minimum(config.cluster_events_sec, 0.0,
                  append_config_path(path, {"cluster_events_sec"}), report);
    check_minimum(config.min_cluster_detectors, 1,
                  append_config_path(path, {"min_cluster_detectors"}), report);
    check_minimum(config.high_score_cluster_override, 0.0,
                  append_config_path(path, {"high_score_cluster_override"}),
                  report);
    check_minimum(config.max_auto_flag_clusters_per_network, 1,
                  append_config_path(path,
                                     {"max_auto_flag_clusters_per_network"}),
                  report);
    validate(config.source_protection,
             {"timestream", "processed_time_chunk", "flagging",
              "second_pass_local", "source_protection"},
             report);
}

inline void validate(const ProcessedTimeChunkStandardPcaConfig &config,
                     ValidationReport &report) {
    check_minimum(config.n_calc, 0,
                  {"timestream", "processed_time_chunk", "clean",
                   "standard_pca", "n_calc"},
                  report);
}

inline void validate(const ProcessedTimeChunkCorrGroupingConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "clean", "corr_grouping"};
    check_minimum(config.corr_min, 0.0,
                  append_config_path(path, {"corr_min"}), report);
    check_maximum(config.corr_min, 1.0,
                  append_config_path(path, {"corr_min"}), report);
    check_minimum(config.min_overlap, 1,
                  append_config_path(path, {"min_overlap"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_group_size, 2,
                  append_config_path(path, {"min_group_size"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
}

inline void validate(const ProcessedTimeChunkNullModelConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "clean", "null_model"};
    check_minimum(config.n_surrogates, 4,
                  append_config_path(path, {"n_surrogates"}), report);
    check_minimum(config.quantile, 0.5,
                  append_config_path(path, {"quantile"}), report);
    check_maximum(config.quantile, 0.999999,
                  append_config_path(path, {"quantile"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.max_modes, 0,
                  append_config_path(path, {"max_modes"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.seed, 0, append_config_path(path, {"seed"}), report);
}

inline void validate(const ProcessedTimeChunkMarchenkoPasturConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "clean", "marchenko_pastur"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.max_modes, 0,
                  append_config_path(path, {"max_modes"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.band_low_Hz, 0.0,
                  append_config_path(path, {"band_low_Hz"}), report);
    check_minimum(config.band_high_Hz, 0.0,
                  append_config_path(path, {"band_high_Hz"}), report);
    check_minimum(config.bulk_keep_frac, 0.1,
                  append_config_path(path, {"bulk_keep_frac"}), report);
    check_maximum(config.bulk_keep_frac, 1.0,
                  append_config_path(path, {"bulk_keep_frac"}), report);
    check_minimum(config.q_grid_size, 8,
                  append_config_path(path, {"q_grid_size"}), report);
}

inline void validate(const ProcessedTimeChunkAdaptiveSelectorConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "clean", "adaptive_selector"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.max_det, 0, append_config_path(path, {"max_det"}),
                  report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.max_pairs, 0,
                  append_config_path(path, {"max_pairs"}), report);
    check_minimum(config.seed, 0, append_config_path(path, {"seed"}), report);
    check_minimum(config.low_weight, 0.0,
                  append_config_path(path, {"low_weight"}), report);
    check_minimum(config.tail_weight, 0.0,
                  append_config_path(path, {"tail_weight"}), report);
    check_minimum(config.topmode_weight, 0.0,
                  append_config_path(path, {"topmode_weight"}), report);
    check_minimum(config.reg_weight, 0.0,
                  append_config_path(path, {"reg_weight"}), report);
    check_minimum(config.low_band_Hz[0], 0.0,
                  append_config_path(path, {"low_band_Hz"}), report);
    if (config.low_band_Hz[1] <= config.low_band_Hz[0]) {
        report.add_error(append_config_path(path, {"low_band_Hz"}),
                         "must be [fmin, fmax] with fmax greater than fmin");
    }
    check_minimum(config.mid_band_Hz[0], 0.0,
                  append_config_path(path, {"mid_band_Hz"}), report);
    if (config.mid_band_Hz[1] <= config.mid_band_Hz[0]) {
        report.add_error(append_config_path(path, {"mid_band_Hz"}),
                         "must be [fmin, fmax] with fmax greater than fmin");
    }
}

inline void validate(const ProcessedTimeChunkCleanConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    validate(config.standard_pca, report);
    validate(config.corr_grouping, report);
    validate(config.null_model, report);
    validate(config.marchenko_pastur, report);
    validate(config.adaptive_selector, report);
}

inline void validate(const ProcessedTimeChunkBusyRowSuppressionConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "weighting",
        "busy_row_suppression"};
    check_minimum(config.min_candidate_clusters, 0,
                  append_config_path(path, {"min_candidate_clusters"}), report);
    check_minimum(config.min_max_unflagged_residual_z, 0.0,
                  append_config_path(path, {"min_max_unflagged_residual_z"}),
                  report);
    check_minimum(config.factor, 0.0, append_config_path(path, {"factor"}),
                  report);
    check_maximum(config.factor, 1.0, append_config_path(path, {"factor"}),
                  report);
}

inline void validate(const ProcessedTimeChunkWeightValidationConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "weighting", "validation"};
    check_minimum(config.accumulation_iters, 1,
                  append_config_path(path, {"accumulation_iters"}), report);
    check_minimum(config.apply_start_iter, 0,
                  append_config_path(path, {"apply_start_iter"}), report);
    check_minimum(config.min_valid_scans, 1,
                  append_config_path(path, {"min_valid_scans"}), report);
    check_minimum(config.min_factor, 0.0,
                  append_config_path(path, {"min_factor"}), report);
    check_maximum(config.min_factor, 1.0,
                  append_config_path(path, {"min_factor"}), report);
    check_minimum(config.unvalidated_factor, 0.0,
                  append_config_path(path, {"unvalidated_factor"}), report);
    check_maximum(config.unvalidated_factor, 1.0,
                  append_config_path(path, {"unvalidated_factor"}), report);
    check_minimum(config.ratio_power, 0.0,
                  append_config_path(path, {"ratio_power"}), report);
    check_minimum(config.transient_ratio_power, 0.0,
                  append_config_path(path, {"transient_ratio_power"}), report);
    check_minimum(config.upward_max_factor, 1.0,
                  append_config_path(path, {"upward_max_factor"}), report);
    check_minimum(config.upward_power, 0.0,
                  append_config_path(path, {"upward_power"}), report);
    check_minimum(config.upward_min_base_factor, 0.0,
                  append_config_path(path, {"upward_min_base_factor"}), report);
    check_maximum(config.upward_min_base_factor, 1.0,
                  append_config_path(path, {"upward_min_base_factor"}), report);
    check_minimum(config.upward_min_atmospheric_factor, 0.0,
                  append_config_path(path, {"upward_min_atmospheric_factor"}),
                  report);
    check_maximum(config.upward_min_atmospheric_factor, 1.0,
                  append_config_path(path, {"upward_min_atmospheric_factor"}),
                  report);
    check_minimum(config.atmospheric_min_detectors, 2,
                  append_config_path(path, {"atmospheric_min_detectors"}),
                  report);
    check_minimum(config.atmospheric_ref, 0.0,
                  append_config_path(path, {"atmospheric_ref"}), report);
    check_maximum(config.atmospheric_ref, 1.0,
                  append_config_path(path, {"atmospheric_ref"}), report);
    check_minimum(config.atmospheric_span, 1e-12,
                  append_config_path(path, {"atmospheric_span"}), report);
    check_minimum(config.atmospheric_power, 0.0,
                  append_config_path(path, {"atmospheric_power"}), report);
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_overlap, 2,
                  append_config_path(path, {"min_overlap"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.high_weight_min_group_detectors, 2,
                  append_config_path(path,
                                     {"high_weight_min_group_detectors"}),
                  report);
    check_minimum(config.high_weight_log_robust_z, 0.0,
                  append_config_path(path, {"high_weight_log_robust_z"}),
                  report);
    check_minimum(config.high_weight_max_median_factor, 1.0,
                  append_config_path(path,
                                     {"high_weight_max_median_factor"}),
                  report);
    check_minimum(config.high_weight_cap_median_factor, 1.0,
                  append_config_path(path,
                                     {"high_weight_cap_median_factor"}),
                  report);
    check_minimum(config.high_weight_min_validated_factor, 0.0,
                  append_config_path(path,
                                     {"high_weight_min_validated_factor"}),
                  report);
}

inline void validate(const ProcessedTimeChunkWeightCorrPenaltyTermConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.span, 1e-12, append_config_path(path, {"span"}),
                  report);
    check_minimum(config.weight, 0.0, append_config_path(path, {"weight"}),
                  report);
}

inline void validate(const ProcessedTimeChunkWeightCorrPenaltyBandConfig &config,
                     const ConfigPath &path,
                     ValidationReport &report) {
    check_minimum(config.span, 1e-12, append_config_path(path, {"span"}),
                  report);
    check_minimum(config.weight, 0.0, append_config_path(path, {"weight"}),
                  report);
    check_minimum(config.low_band_Hz[0], 0.0,
                  append_config_path(path, {"low_band_Hz"}), report);
    if (config.low_band_Hz[1] <= config.low_band_Hz[0]) {
        report.add_error(append_config_path(path, {"low_band_Hz"}),
                         "must be [fmin, fmax] with fmax greater than fmin");
    }
    check_minimum(config.mid_band_Hz[0], 0.0,
                  append_config_path(path, {"mid_band_Hz"}), report);
    if (config.mid_band_Hz[1] <= config.mid_band_Hz[0]) {
        report.add_error(append_config_path(path, {"mid_band_Hz"}),
                         "must be [fmin, fmax] with fmax greater than fmin");
    }
}

inline void validate(const ProcessedTimeChunkWeightCorrPenaltyConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{
        "timestream", "processed_time_chunk", "weighting", "corr_penalty"};
    check_minimum(config.min_good_frac, 0.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_maximum(config.min_good_frac, 1.0,
                  append_config_path(path, {"min_good_frac"}), report);
    check_minimum(config.min_overlap, 2,
                  append_config_path(path, {"min_overlap"}), report);
    check_minimum(config.max_samples, 0,
                  append_config_path(path, {"max_samples"}), report);
    check_minimum(config.max_pairs, 0,
                  append_config_path(path, {"max_pairs"}), report);
    check_minimum(config.seed, 0, append_config_path(path, {"seed"}), report);
    check_minimum(config.floor, 0.0, append_config_path(path, {"floor"}),
                  report);
    check_maximum(config.floor, 1.0, append_config_path(path, {"floor"}),
                  report);
    check_minimum(config.exponent, 0.0,
                  append_config_path(path, {"exponent"}), report);
    validate(config.pair_corr, append_config_path(path, {"pair_corr"}), report);
    validate(config.cm_el_corr, append_config_path(path, {"cm_el_corr"}),
             report);
    validate(config.cm_low_mid_ratio,
             append_config_path(path, {"cm_low_mid_ratio"}), report);
}

inline void validate(const ProcessedTimeChunkWeightingConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{"timestream", "processed_time_chunk", "weighting"};
    check_minimum(config.source_mask_radius_arcsec, 0.0,
                  append_config_path(path, {"source_mask_radius_arcsec"}),
                  report);
    check_minimum(config.hybrid_correction_min_factor, 0.0,
                  append_config_path(path, {"hybrid_correction_min_factor"}),
                  report);
    check_minimum(config.hybrid_correction_max_factor, 0.0,
                  append_config_path(path, {"hybrid_correction_max_factor"}),
                  report);
    if (config.hybrid_correction_max_factor <
        config.hybrid_correction_min_factor) {
        report.add_error(
            append_config_path(path, {"hybrid_correction_max_factor"}),
            "must be greater than or equal to hybrid_correction_min_factor");
    }
    validate(config.validation, report);
    validate(config.corr_penalty, report);
    validate(config.busy_row_suppression, report);
}

inline void validate(const ProcessedTimeChunkFlaggingConfig &config,
                     ValidationReport &report) {
    validate(config.second_pass_local, report);
}

inline void validate(const ProcessedTimeChunkConfig &config,
                     ValidationReport &report) {
    validate(config.clean, report);
    validate(config.weighting, report);
    validate(config.flagging, report);
}

inline void validate(const FruitLoopsWeightFeedbackConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{"timestream", "fruit_loops", "weight_feedback"};
    check_minimum(config.low_relative_weight, 0.0,
                  append_config_path(path, {"low_relative_weight"}), report);
    check_minimum(config.high_relative_weight, 0.0,
                  append_config_path(path, {"high_relative_weight"}), report);
    if (config.enabled &&
        config.high_relative_weight <= config.low_relative_weight) {
        report.add_error(append_config_path(path, {"high_relative_weight"}),
                         "must be greater than low_relative_weight when enabled");
    }
}

inline void validate(const TimestreamFruitLoopsConfig &config,
                     ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    const ConfigPath path{"timestream", "fruit_loops"};
    check_minimum(config.peak_fraction_limit, 0.0,
                  append_config_path(path, {"peak_fraction_limit"}), report);
    check_minimum(config.local_snr_floor, 0.0,
                  append_config_path(path, {"local_snr_floor"}), report);
    check_minimum(config.local_sigma_inner_radius_arcsec, 0.0,
                  append_config_path(path, {"local_sigma_inner_radius_arcsec"}),
                  report);
    check_minimum(config.local_sigma_outer_radius_arcsec, 0.0,
                  append_config_path(path, {"local_sigma_outer_radius_arcsec"}),
                  report);
    check_minimum(config.local_sigma_inner_fwhm, 0.0,
                  append_config_path(path, {"local_sigma_inner_fwhm"}), report);
    check_minimum(config.local_sigma_outer_fwhm, 0.0,
                  append_config_path(path, {"local_sigma_outer_fwhm"}), report);
    check_minimum(config.local_sigma_edge_guard_arcsec, 0.0,
                  append_config_path(path, {"local_sigma_edge_guard_arcsec"}),
                  report);
    check_minimum(config.local_sigma_min_pixels, 1,
                  append_config_path(path, {"local_sigma_min_pixels"}), report);
    check_minimum(config.adaptive_support_radius_arcsec, 0.0,
                  append_config_path(path, {"adaptive_support_radius_arcsec"}),
                  report);
    check_minimum(config.adaptive_support_radius_fwhm, 0.0,
                  append_config_path(path, {"adaptive_support_radius_fwhm"}),
                  report);
    validate(config.weight_feedback, report);
    check_minimum(config.center_keep_radius_arcsec, 0.0,
                  append_config_path(path, {"center_keep_radius_arcsec"}),
                  report);
    check_minimum(config.max_iters, 0,
                  append_config_path(path, {"max_iters"}), report);
}

inline void validate(const TimestreamLearningMapPixelOutlierConfig &config,
                     ValidationReport &report) {
    check_minimum(config.top_n, 0,
                  {"timestream", "learning", "map_pixel_outlier_top_n"},
                  report);
    check_minimum(config.targeted_contributor_max_pixels, 0,
                  {"timestream", "learning",
                   "map_pixel_outlier_targeted_contributor_max_pixels"},
                  report);
    check_minimum(config.detector_exclusion_min_pixels, 1,
                  {"timestream", "learning",
                   "map_pixel_outlier_detector_exclusion_min_pixels"},
                  report);
    check_minimum(config.min_abs_z, 0.0,
                  {"timestream", "learning", "map_pixel_outlier_min_abs_z"},
                  report);
    check_minimum(config.min_n_eff, 0.0,
                  {"timestream", "learning", "map_pixel_outlier_min_n_eff"},
                  report);
    check_minimum(config.source_radius_arcsec, 0.0,
                  {"timestream", "learning",
                   "map_pixel_outlier_source_radius_arcsec"},
                  report);
}

inline void validate(const TimestreamLearningBusyDetectorConfig &,
                     ValidationReport &) {}

inline void validate(const TimestreamLearningScanNetworkPathologyConfig &config,
                     ValidationReport &report) {
    check_minimum(config.min_candidate_clusters, 0,
                  {"timestream", "learning",
                   "scan_network_pathology_min_candidate_clusters"},
                  report);
    check_minimum(config.min_candidate_events, 0,
                  {"timestream", "learning",
                   "scan_network_pathology_min_candidate_events"},
                  report);
    check_minimum(config.min_max_residual_z, 0.0,
                  {"timestream", "learning",
                   "scan_network_pathology_min_max_residual_z"},
                  report);
    check_minimum(config.severe_candidate_events, 0,
                  {"timestream", "learning",
                   "scan_network_pathology_severe_candidate_events"},
                  report);
    check_minimum(config.severe_max_residual_z, 0.0,
                  {"timestream", "learning",
                   "scan_network_pathology_severe_max_residual_z"},
                  report);
    check_minimum(config.max_new_flagged_fraction, 0.0,
                  {"timestream", "learning",
                   "scan_network_pathology_max_new_flagged_fraction"},
                  report);
}

inline void validate(const TimestreamLearningConfig &config,
                     ValidationReport &report) {
    check_minimum(config.learn_iters, 0,
                  {"timestream", "learning", "learn_iters"}, report);
    check_minimum(config.apply_start_iter, 0,
                  {"timestream", "learning", "apply_start_iter"}, report);
    check_minimum(config.max_records_per_type, 0,
                  {"timestream", "learning", "max_records_per_type"}, report);
    check_minimum(config.apply_max_new_flagged_fraction, 0.0,
                  {"timestream", "learning", "apply_max_new_flagged_fraction"},
                  report);
    validate(config.map_pixel_outlier, report);
    validate(config.busy_detector, report);
    validate(config.scan_network_pathology, report);
}

inline void validate(const TimestreamConfig &config, ValidationReport &report) {
    if (!config.enabled) {
        report.add_error({"timestream", "enabled"},
                         "false is not supported by the current pipeline");
    }
    validate(config.output.raw_time_chunk,
             {"timestream", "raw_time_chunk", "output"}, report);
    validate(config.output.processed_time_chunk,
             {"timestream", "processed_time_chunk", "output"}, report);
    validate(config.chunking, report);
    validate(config.raw_time_chunk, report);
    validate(config.processed_time_chunk, report);
    validate(config.fruit_loops, report);
    validate(config.learning, report);
}

}  // namespace citlali::config
