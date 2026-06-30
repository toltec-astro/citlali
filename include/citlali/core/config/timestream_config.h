#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/enum_parser.h>

#include <array>
#include <initializer_list>
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

inline constexpr std::array<EnumName<TodType>, 4> tod_type_names{{
    {TodType::xs, "xs"},
    {TodType::rs, "rs"},
    {TodType::is, "is"},
    {TodType::qs, "qs"},
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

inline std::optional<TodType> parse_tod_type(std::string_view value) {
    return parse_enum(value, tod_type_names);
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

inline std::string_view to_string(TodType value) {
    return enum_name(value, tod_type_names);
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

struct RawTimeChunkConfig {
    RawTimeChunkDespikeConfig despike;
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

struct ProcessedTimeChunkFlaggingConfig {
    ProcessedTimeChunkSecondPassLocalConfig second_pass_local;
};

struct ProcessedTimeChunkConfig {
    ProcessedTimeChunkFlaggingConfig flagging;
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

inline void validate(const RawTimeChunkConfig &config, ValidationReport &report) {
    validate(config.despike, report);
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

inline void validate(const ProcessedTimeChunkFlaggingConfig &config,
                     ValidationReport &report) {
    validate(config.second_pass_local, report);
}

inline void validate(const ProcessedTimeChunkConfig &config,
                     ValidationReport &report) {
    validate(config.flagging, report);
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
    validate(config.learning, report);
}

}  // namespace citlali::config
