#pragma once

#include <citlali/core/config/enum_parser.h>

#include <array>
#include <optional>
#include <string_view>

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

enum class AuxiliaryMeasuredChannelCalibrationPolicy {
    native,
    primary_equivalent,
    sky_equivalent
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

inline constexpr std::string_view corr_network_processed_cleaner_grouping() {
    return "corr_nw";
}

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

inline constexpr std::array<EnumName<AuxiliaryMeasuredChannelCalibrationPolicy>, 3>
    auxiliary_measured_channel_calibration_policy_names{{
        {AuxiliaryMeasuredChannelCalibrationPolicy::native, "native"},
        {AuxiliaryMeasuredChannelCalibrationPolicy::primary_equivalent,
         "primary_equivalent"},
        {AuxiliaryMeasuredChannelCalibrationPolicy::sky_equivalent,
         "sky_equivalent"},
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

inline std::optional<AuxiliaryMeasuredChannelCalibrationPolicy>
parse_auxiliary_measured_channel_calibration_policy(std::string_view value) {
    return parse_enum(value, auxiliary_measured_channel_calibration_policy_names);
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

inline std::string_view to_string(
    AuxiliaryMeasuredChannelCalibrationPolicy value) {
    return enum_name(value, auxiliary_measured_channel_calibration_policy_names);
}

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
