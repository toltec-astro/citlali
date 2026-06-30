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

struct TimestreamConfig {
    bool enabled = true;
    TodType type = TodType::xs;
    TimestreamOutputConfig output;
    TimestreamChunkingConfig chunking;
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
}

}  // namespace citlali::config
