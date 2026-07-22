#include <citlali/core/config/enum_parser.h>
#include <citlali/core/config/timestream_enums.h>

#include <array>

namespace citlali::config {
namespace {

inline constexpr std::array<EnumName<TodType>, 4> tod_type_names{{
    {TodType::xs, "xs"},
    {TodType::rs, "rs"},
    {TodType::is, "is"},
    {TodType::qs, "qs"},
}};

inline constexpr std::array<EnumName<PolarimetryGrouping>, 2>
    polarimetry_grouping_names{{
        {PolarimetryGrouping::frequency_group, "fg"},
        {PolarimetryGrouping::detector_location, "loc"},
    }};

inline constexpr std::array<EnumName<PolarimetryHwprPolicy>, 3>
    polarimetry_hwpr_policy_names{{
        {PolarimetryHwprPolicy::automatic, "auto"},
        {PolarimetryHwprPolicy::ignore, "true"},
        {PolarimetryHwprPolicy::require, "false"},
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

inline constexpr std::array<EnumName<FruitLoopsSourceCenterMode>, 4>
    fruit_loops_source_center_mode_names{{
        {FruitLoopsSourceCenterMode::automatic, "auto"},
        {FruitLoopsSourceCenterMode::header, "header"},
        {FruitLoopsSourceCenterMode::peak, "peak"},
        {FruitLoopsSourceCenterMode::map_center, "map_center"},
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

}  // namespace

std::optional<TodType> parse_tod_type(std::string_view value) {
    return parse_enum(value, tod_type_names);
}

std::optional<PolarimetryGrouping> parse_polarimetry_grouping(
    std::string_view value) {
    return parse_enum(value, polarimetry_grouping_names);
}

std::optional<PolarimetryHwprPolicy> parse_polarimetry_hwpr_policy(
    std::string_view value) {
    return parse_enum(value, polarimetry_hwpr_policy_names);
}

std::optional<TodOutputStream> parse_tod_output_stream(
    std::string_view value) {
    return parse_enum(value, tod_output_stream_names);
}

std::optional<TodOutputType> parse_tod_output_type(std::string_view value) {
    return parse_enum(value, tod_output_type_names);
}

std::optional<TodStreamOutputMode> parse_tod_stream_output_mode(
    std::string_view value) {
    return parse_enum(value, tod_stream_output_mode_names);
}

std::optional<TodOutputSelectionMode> parse_tod_output_selection_mode(
    std::string_view value) {
    return parse_enum(value, tod_output_selection_mode_names);
}

std::optional<RawTimeChunkFilterEdgeGuardMode>
parse_raw_filter_edge_guard_mode(std::string_view value) {
    return parse_enum(value, raw_filter_edge_guard_mode_names);
}

std::optional<RawTimeChunkFilterEdgeGuardCombine>
parse_raw_filter_edge_guard_combine(std::string_view value) {
    return parse_enum(value, raw_filter_edge_guard_combine_names);
}

std::optional<ProcessedTimeChunkWeightingType> parse_processed_weighting_type(
    std::string_view value) {
    return parse_enum(value, processed_weighting_type_names);
}

std::optional<ProcessedTimeChunkWeightGrouping>
parse_processed_weight_grouping(std::string_view value) {
    return parse_enum(value, processed_weight_grouping_names);
}

std::optional<ProcessedTimeChunkCleanerMode>
parse_processed_cleaner_mode(std::string_view value) {
    return parse_enum(value, processed_cleaner_mode_names);
}

std::optional<ProcessedTimeChunkCorrGroupingMetric>
parse_processed_corr_grouping_metric(std::string_view value) {
    return parse_enum(value, processed_corr_grouping_metric_names);
}

std::optional<FruitLoopsMode> parse_fruit_loops_mode(
    std::string_view value) {
    return parse_enum(value, fruit_loops_mode_names);
}

std::optional<FruitLoopsSourceCenterMode>
parse_fruit_loops_source_center_mode(std::string_view value) {
    return parse_enum(value, fruit_loops_source_center_mode_names);
}

std::optional<FruitLoopsWeightFeedbackReference>
parse_fruit_loops_weight_feedback_reference(std::string_view value) {
    return parse_enum(value, fruit_loops_weight_feedback_reference_names);
}

std::optional<FruitLoopsInterpModeOverride>
parse_fruit_loops_interp_mode_override(std::string_view value) {
    return parse_enum(value, fruit_loops_interp_mode_override_names);
}

std::optional<AuxiliaryMeasuredChannelCalibrationPolicy>
parse_auxiliary_measured_channel_calibration_policy(std::string_view value) {
    return parse_enum(value, auxiliary_measured_channel_calibration_policy_names);
}

std::string_view to_string(TodType value) {
    return enum_name(value, tod_type_names);
}

std::string_view to_string(PolarimetryGrouping value) {
    return enum_name(value, polarimetry_grouping_names);
}

std::string_view to_string(PolarimetryHwprPolicy value) {
    return enum_name(value, polarimetry_hwpr_policy_names);
}

std::string_view to_string(TodOutputStream value) {
    return enum_name(value, tod_output_stream_names);
}

std::string_view to_string(TodOutputType value) {
    return enum_name(value, tod_output_type_names);
}

std::string_view to_string(TodStreamOutputMode value) {
    return enum_name(value, tod_stream_output_mode_names);
}

std::string_view to_string(TodOutputSelectionMode value) {
    return enum_name(value, tod_output_selection_mode_names);
}

std::string_view to_string(RawTimeChunkFilterEdgeGuardMode value) {
    return enum_name(value, raw_filter_edge_guard_mode_names);
}

std::string_view to_string(RawTimeChunkFilterEdgeGuardCombine value) {
    return enum_name(value, raw_filter_edge_guard_combine_names);
}

std::string_view to_string(ProcessedTimeChunkWeightingType value) {
    return enum_name(value, processed_weighting_type_names);
}

std::string_view to_string(ProcessedTimeChunkWeightGrouping value) {
    return enum_name(value, processed_weight_grouping_names);
}

std::string_view to_string(ProcessedTimeChunkCleanerMode value) {
    return enum_name(value, processed_cleaner_mode_names);
}

std::string_view to_string(ProcessedTimeChunkCorrGroupingMetric value) {
    return enum_name(value, processed_corr_grouping_metric_names);
}

std::string_view to_string(FruitLoopsMode value) {
    return enum_name(value, fruit_loops_mode_names);
}

std::string_view to_string(FruitLoopsSourceCenterMode value) {
    return enum_name(value, fruit_loops_source_center_mode_names);
}

std::string_view to_string(FruitLoopsWeightFeedbackReference value) {
    return enum_name(value, fruit_loops_weight_feedback_reference_names);
}

std::string_view to_string(FruitLoopsInterpModeOverride value) {
    return enum_name(value, fruit_loops_interp_mode_override_names);
}

std::string_view to_string(
    AuxiliaryMeasuredChannelCalibrationPolicy value) {
    return enum_name(value, auxiliary_measured_channel_calibration_policy_names);
}

}  // namespace citlali::config
