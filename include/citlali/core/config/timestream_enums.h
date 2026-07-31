#pragma once

#include <optional>
#include <string_view>

namespace citlali::config {

enum class TodType {
    xs,
    rs,
    is,
    qs
};

enum class PolarimetryGrouping {
    frequency_group,
    detector_location
};

enum class PolarimetryHwprPolicy {
    automatic,
    ignore,
    require
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

enum class FruitLoopsSourceCenterMode {
    automatic,
    header,
    peak,
    map_center
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

enum class AuxiliaryMeasuredChannelCalibrationPolicy {
    native,
    primary_equivalent,
    sky_equivalent
};

inline constexpr std::string_view corr_network_processed_cleaner_grouping() {
    return "corr_nw";
}

std::optional<TodType> parse_tod_type(std::string_view value);
std::optional<PolarimetryGrouping> parse_polarimetry_grouping(
    std::string_view value);
std::optional<PolarimetryHwprPolicy> parse_polarimetry_hwpr_policy(
    std::string_view value);
std::optional<TodOutputStream> parse_tod_output_stream(
    std::string_view value);
std::optional<TodOutputType> parse_tod_output_type(std::string_view value);
std::optional<TodStreamOutputMode> parse_tod_stream_output_mode(
    std::string_view value);
std::optional<TodOutputSelectionMode> parse_tod_output_selection_mode(
    std::string_view value);
std::optional<RawTimeChunkFilterEdgeGuardMode>
parse_raw_filter_edge_guard_mode(std::string_view value);
std::optional<RawTimeChunkFilterEdgeGuardCombine>
parse_raw_filter_edge_guard_combine(std::string_view value);
std::optional<ProcessedTimeChunkWeightingType> parse_processed_weighting_type(
    std::string_view value);
std::optional<ProcessedTimeChunkWeightGrouping>
parse_processed_weight_grouping(std::string_view value);
std::optional<ProcessedTimeChunkCleanerMode>
parse_processed_cleaner_mode(std::string_view value);
std::optional<ProcessedTimeChunkCorrGroupingMetric>
parse_processed_corr_grouping_metric(std::string_view value);
std::optional<FruitLoopsMode> parse_fruit_loops_mode(std::string_view value);
std::optional<FruitLoopsSourceCenterMode>
parse_fruit_loops_source_center_mode(std::string_view value);
std::optional<FruitLoopsWeightFeedbackReference>
parse_fruit_loops_weight_feedback_reference(std::string_view value);
std::optional<FruitLoopsInterpModeOverride>
parse_fruit_loops_interp_mode_override(std::string_view value);
std::optional<AuxiliaryMeasuredChannelCalibrationPolicy>
parse_auxiliary_measured_channel_calibration_policy(std::string_view value);

std::string_view to_string(TodType value);
std::string_view to_string(PolarimetryGrouping value);
std::string_view to_string(PolarimetryHwprPolicy value);
std::string_view to_string(TodOutputStream value);
std::string_view to_string(TodOutputType value);
std::string_view to_string(TodStreamOutputMode value);
std::string_view to_string(TodOutputSelectionMode value);
std::string_view to_string(RawTimeChunkFilterEdgeGuardMode value);
std::string_view to_string(RawTimeChunkFilterEdgeGuardCombine value);
std::string_view to_string(ProcessedTimeChunkWeightingType value);
std::string_view to_string(ProcessedTimeChunkWeightGrouping value);
std::string_view to_string(ProcessedTimeChunkCleanerMode value);
std::string_view to_string(ProcessedTimeChunkCorrGroupingMetric value);
std::string_view to_string(FruitLoopsMode value);
std::string_view to_string(FruitLoopsSourceCenterMode value);
std::string_view to_string(FruitLoopsWeightFeedbackReference value);
std::string_view to_string(FruitLoopsInterpModeOverride value);
std::string_view to_string(
    AuxiliaryMeasuredChannelCalibrationPolicy value);

inline bool is_tod_output_stream(TodOutputStream value,
                                 TodOutputStream stream) {
    return value == stream;
}

inline bool is_tod_output_selection_mode(
    std::string_view value, TodOutputSelectionMode mode) {
    return value == to_string(mode);
}

inline bool is_tod_type(TodType value, TodType type) {
    return value == type;
}

inline bool is_tod_type(std::string_view value, TodType type) {
    return value == to_string(type);
}

inline bool is_xs_tod_type(TodType value) {
    return is_tod_type(value, TodType::xs);
}

inline bool is_xs_tod_type(std::string_view value) {
    return is_tod_type(value, TodType::xs);
}

inline bool is_rs_tod_type(TodType value) {
    return is_tod_type(value, TodType::rs);
}

inline bool is_rs_tod_type(std::string_view value) {
    return is_tod_type(value, TodType::rs);
}

inline bool is_is_tod_type(TodType value) {
    return is_tod_type(value, TodType::is);
}

inline bool is_is_tod_type(std::string_view value) {
    return is_tod_type(value, TodType::is);
}

inline bool is_qs_tod_type(TodType value) {
    return is_tod_type(value, TodType::qs);
}

inline bool is_qs_tod_type(std::string_view value) {
    return is_tod_type(value, TodType::qs);
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

inline bool is_raw_filter_edge_guard_mode(
    RawTimeChunkFilterEdgeGuardMode value,
    RawTimeChunkFilterEdgeGuardMode mode) {
    return value == mode;
}

inline bool is_raw_filter_edge_guard_mode(
    std::string_view value, RawTimeChunkFilterEdgeGuardMode mode) {
    return value == to_string(mode);
}

inline bool is_flag_raw_filter_edge_guard_mode(
    RawTimeChunkFilterEdgeGuardMode value) {
    return is_raw_filter_edge_guard_mode(
        value, RawTimeChunkFilterEdgeGuardMode::flag);
}

inline bool is_flag_raw_filter_edge_guard_mode(std::string_view value) {
    return is_raw_filter_edge_guard_mode(
        value, RawTimeChunkFilterEdgeGuardMode::flag);
}

inline bool is_none_raw_filter_edge_guard_mode(
    RawTimeChunkFilterEdgeGuardMode value) {
    return is_raw_filter_edge_guard_mode(
        value, RawTimeChunkFilterEdgeGuardMode::none);
}

inline bool is_none_raw_filter_edge_guard_mode(std::string_view value) {
    return is_raw_filter_edge_guard_mode(
        value, RawTimeChunkFilterEdgeGuardMode::none);
}

inline bool is_raw_filter_edge_guard_combine(
    RawTimeChunkFilterEdgeGuardCombine value,
    RawTimeChunkFilterEdgeGuardCombine combine) {
    return value == combine;
}

inline bool is_raw_filter_edge_guard_combine(
    std::string_view value, RawTimeChunkFilterEdgeGuardCombine combine) {
    return value == to_string(combine);
}

inline bool is_sum_raw_filter_edge_guard_combine(
    RawTimeChunkFilterEdgeGuardCombine value) {
    return is_raw_filter_edge_guard_combine(
        value, RawTimeChunkFilterEdgeGuardCombine::sum);
}

inline bool is_sum_raw_filter_edge_guard_combine(std::string_view value) {
    return is_raw_filter_edge_guard_combine(
        value, RawTimeChunkFilterEdgeGuardCombine::sum);
}

inline bool is_max_raw_filter_edge_guard_combine(
    RawTimeChunkFilterEdgeGuardCombine value) {
    return is_raw_filter_edge_guard_combine(
        value, RawTimeChunkFilterEdgeGuardCombine::max);
}

inline bool is_max_raw_filter_edge_guard_combine(std::string_view value) {
    return is_raw_filter_edge_guard_combine(
        value, RawTimeChunkFilterEdgeGuardCombine::max);
}

inline bool is_processed_weighting_type(
    ProcessedTimeChunkWeightingType value,
    ProcessedTimeChunkWeightingType type) {
    return value == type;
}

inline bool is_processed_weighting_type(
    std::string_view value, ProcessedTimeChunkWeightingType type) {
    return value == to_string(type);
}

inline bool is_full_processed_weighting_type(
    ProcessedTimeChunkWeightingType value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::full);
}

inline bool is_full_processed_weighting_type(std::string_view value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::full);
}

inline bool is_approximate_processed_weighting_type(
    ProcessedTimeChunkWeightingType value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::approximate);
}

inline bool is_approximate_processed_weighting_type(std::string_view value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::approximate);
}

inline bool is_hybrid_processed_weighting_type(
    ProcessedTimeChunkWeightingType value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::hybrid);
}

inline bool is_hybrid_processed_weighting_type(std::string_view value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::hybrid);
}

inline bool is_validated_processed_weighting_type(
    ProcessedTimeChunkWeightingType value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::validated);
}

inline bool is_validated_processed_weighting_type(std::string_view value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::validated);
}

inline bool is_constant_processed_weighting_type(
    ProcessedTimeChunkWeightingType value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::constant);
}

inline bool is_constant_processed_weighting_type(std::string_view value) {
    return is_processed_weighting_type(
        value, ProcessedTimeChunkWeightingType::constant);
}

inline bool is_processed_weight_grouping(
    ProcessedTimeChunkWeightGrouping value,
    ProcessedTimeChunkWeightGrouping grouping) {
    return value == grouping;
}

inline bool is_processed_weight_grouping(
    std::string_view value, ProcessedTimeChunkWeightGrouping grouping) {
    return value == to_string(grouping);
}

inline bool is_array_processed_weight_grouping(
    ProcessedTimeChunkWeightGrouping value) {
    return is_processed_weight_grouping(
        value, ProcessedTimeChunkWeightGrouping::array);
}

inline bool is_array_processed_weight_grouping(std::string_view value) {
    return is_processed_weight_grouping(
        value, ProcessedTimeChunkWeightGrouping::array);
}

inline bool is_network_processed_weight_grouping(
    ProcessedTimeChunkWeightGrouping value) {
    return is_processed_weight_grouping(
        value, ProcessedTimeChunkWeightGrouping::network);
}

inline bool is_network_processed_weight_grouping(std::string_view value) {
    return is_processed_weight_grouping(
        value, ProcessedTimeChunkWeightGrouping::network);
}

inline bool is_all_processed_weight_grouping(
    ProcessedTimeChunkWeightGrouping value) {
    return is_processed_weight_grouping(
        value, ProcessedTimeChunkWeightGrouping::all);
}

inline bool is_all_processed_weight_grouping(std::string_view value) {
    return is_processed_weight_grouping(
        value, ProcessedTimeChunkWeightGrouping::all);
}

inline bool is_corr_network_processed_cleaner_grouping(
    std::string_view value) {
    return value == corr_network_processed_cleaner_grouping();
}

inline bool is_processed_corr_grouping_metric(
    ProcessedTimeChunkCorrGroupingMetric value,
    ProcessedTimeChunkCorrGroupingMetric metric) {
    return value == metric;
}

inline bool is_processed_corr_grouping_metric(
    std::string_view value, ProcessedTimeChunkCorrGroupingMetric metric) {
    return value == to_string(metric);
}

inline bool is_abs_processed_corr_grouping_metric(
    ProcessedTimeChunkCorrGroupingMetric value) {
    return is_processed_corr_grouping_metric(
        value, ProcessedTimeChunkCorrGroupingMetric::abs);
}

inline bool is_abs_processed_corr_grouping_metric(std::string_view value) {
    return is_processed_corr_grouping_metric(
        value, ProcessedTimeChunkCorrGroupingMetric::abs);
}

inline bool is_signed_processed_corr_grouping_metric(
    ProcessedTimeChunkCorrGroupingMetric value) {
    return is_processed_corr_grouping_metric(
        value, ProcessedTimeChunkCorrGroupingMetric::signed_metric);
}

inline bool is_signed_processed_corr_grouping_metric(std::string_view value) {
    return is_processed_corr_grouping_metric(
        value, ProcessedTimeChunkCorrGroupingMetric::signed_metric);
}

inline constexpr std::string_view fruit_loops_coadd_type() {
    return "coadd";
}

inline constexpr std::string_view fruit_loops_coadded_type_alias() {
    return "coadded";
}

inline constexpr std::string_view fruit_loops_obsnum_raw_type() {
    return "obsnum/raw";
}

inline constexpr std::string_view fruit_loops_obsnum_filtered_type() {
    return "obsnum/filtered";
}

inline constexpr std::string_view fruit_loops_coadd_raw_type() {
    return "coadd/raw";
}

inline constexpr std::string_view fruit_loops_coadd_filtered_type() {
    return "coadd/filtered";
}

inline std::string_view canonical_fruit_loops_type(std::string_view value) {
    if (value == fruit_loops_coadded_type_alias()) {
        return fruit_loops_coadd_type();
    }
    return value;
}

inline bool is_obsnum_raw_fruit_loops_type(std::string_view value) {
    return value == fruit_loops_obsnum_raw_type();
}

inline bool is_obsnum_filtered_fruit_loops_type(std::string_view value) {
    return value == fruit_loops_obsnum_filtered_type();
}

inline bool is_coadd_raw_fruit_loops_type(std::string_view value) {
    return value == fruit_loops_coadd_raw_type();
}

inline bool is_coadd_filtered_fruit_loops_type(std::string_view value) {
    return value == fruit_loops_coadd_filtered_type();
}

inline bool is_filtered_fruit_loops_type(std::string_view value) {
    return is_obsnum_filtered_fruit_loops_type(value) ||
           is_coadd_filtered_fruit_loops_type(value);
}

inline bool is_fruit_loops_mode(FruitLoopsMode value,
                                FruitLoopsMode mode) {
    return value == mode;
}

inline bool is_fruit_loops_mode(std::string_view value,
                                FruitLoopsMode mode) {
    return value == to_string(mode);
}

inline bool is_upper_fruit_loops_mode(FruitLoopsMode value) {
    return is_fruit_loops_mode(value, FruitLoopsMode::upper);
}

inline bool is_upper_fruit_loops_mode(std::string_view value) {
    return is_fruit_loops_mode(value, FruitLoopsMode::upper);
}

inline bool is_lower_fruit_loops_mode(FruitLoopsMode value) {
    return is_fruit_loops_mode(value, FruitLoopsMode::lower);
}

inline bool is_lower_fruit_loops_mode(std::string_view value) {
    return is_fruit_loops_mode(value, FruitLoopsMode::lower);
}

inline bool is_both_fruit_loops_mode(FruitLoopsMode value) {
    return is_fruit_loops_mode(value, FruitLoopsMode::both);
}

inline bool is_both_fruit_loops_mode(std::string_view value) {
    return is_fruit_loops_mode(value, FruitLoopsMode::both);
}

inline bool is_fruit_loops_weight_feedback_reference(
    FruitLoopsWeightFeedbackReference value,
    FruitLoopsWeightFeedbackReference reference) {
    return value == reference;
}

inline bool is_fruit_loops_weight_feedback_reference(
    std::string_view value, FruitLoopsWeightFeedbackReference reference) {
    return value == to_string(reference);
}

inline bool is_p95_fruit_loops_weight_feedback_reference(
    std::string_view value) {
    return is_fruit_loops_weight_feedback_reference(
        value, FruitLoopsWeightFeedbackReference::p95);
}

inline bool is_p90_fruit_loops_weight_feedback_reference(
    std::string_view value) {
    return is_fruit_loops_weight_feedback_reference(
        value, FruitLoopsWeightFeedbackReference::p90);
}

inline bool is_p99_fruit_loops_weight_feedback_reference(
    std::string_view value) {
    return is_fruit_loops_weight_feedback_reference(
        value, FruitLoopsWeightFeedbackReference::p99);
}

inline bool is_median_fruit_loops_weight_feedback_reference(
    std::string_view value) {
    return is_fruit_loops_weight_feedback_reference(
        value, FruitLoopsWeightFeedbackReference::median);
}

inline bool is_p50_fruit_loops_weight_feedback_reference(
    std::string_view value) {
    return is_fruit_loops_weight_feedback_reference(
        value, FruitLoopsWeightFeedbackReference::p50);
}

inline bool is_max_fruit_loops_weight_feedback_reference(
    std::string_view value) {
    return is_fruit_loops_weight_feedback_reference(
        value, FruitLoopsWeightFeedbackReference::max);
}

inline bool is_peak_fruit_loops_weight_feedback_reference(
    std::string_view value) {
    return is_fruit_loops_weight_feedback_reference(
        value, FruitLoopsWeightFeedbackReference::peak);
}

inline bool is_indices_tod_output_selection_mode(
    TodOutputSelectionMode value) {
    return value == TodOutputSelectionMode::indices;
}

inline bool is_all_tod_output_selection_mode(TodOutputSelectionMode value) {
    return value == TodOutputSelectionMode::all;
}

inline bool is_all_tod_output_selection_mode(std::string_view value) {
    return is_tod_output_selection_mode(value, TodOutputSelectionMode::all);
}

inline bool is_uniform_source_tod_output_selection_mode(
    TodOutputSelectionMode value) {
    return value ==
           TodOutputSelectionMode::uniform_plus_source_crossing;
}

inline bool is_fruit_loops_interp_mode(
    FruitLoopsInterpModeOverride value, FruitLoopsInterpModeOverride mode) {
    return value == mode;
}

inline bool is_fruit_loops_interp_mode(
    std::string_view value, FruitLoopsInterpModeOverride mode) {
    return value == to_string(mode);
}

inline constexpr std::string_view
legacy_nearest_fruit_loops_interp_mode_override() {
    return "legacy_nearest";
}

inline std::string_view canonical_fruit_loops_interp_mode_override(
    std::string_view value) {
    if (value == legacy_nearest_fruit_loops_interp_mode_override()) {
        return to_string(FruitLoopsInterpModeOverride::trunc);
    }
    return value;
}

inline bool is_fruit_loops_auto_interp_mode(
    FruitLoopsInterpModeOverride value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::automatic);
}

inline bool is_fruit_loops_auto_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::automatic);
}

inline bool is_fruit_loops_nearest_interp_mode(
    FruitLoopsInterpModeOverride value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::nearest);
}

inline bool is_fruit_loops_nearest_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::nearest);
}

inline bool is_fruit_loops_bilinear_interp_mode(
    FruitLoopsInterpModeOverride value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::bilinear);
}

inline bool is_fruit_loops_bilinear_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::bilinear);
}

inline bool is_fruit_loops_jinc_interp_mode(
    FruitLoopsInterpModeOverride value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::jinc);
}

inline bool is_fruit_loops_jinc_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::jinc);
}

inline bool is_fruit_loops_trunc_interp_mode(
    FruitLoopsInterpModeOverride value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::trunc);
}

inline bool is_fruit_loops_trunc_interp_mode(std::string_view value) {
    return is_fruit_loops_interp_mode(
        value, FruitLoopsInterpModeOverride::trunc);
}

}  // namespace citlali::config
