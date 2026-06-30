#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/enum_parser.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>

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

inline std::optional<TodType> parse_tod_type(std::string_view value) {
    return parse_enum(value, tod_type_names);
}

inline std::optional<TodOutputType> parse_tod_output_type(std::string_view value) {
    return parse_enum(value, tod_output_type_names);
}

inline std::string_view to_string(TodType value) {
    return enum_name(value, tod_type_names);
}

inline std::string_view to_string(TodOutputType value) {
    return enum_name(value, tod_output_type_names);
}

struct TimestreamOutputConfig {
    bool raw_time_chunk_enabled = false;
    bool processed_time_chunk_enabled = false;
    TodOutputType type = TodOutputType::none;
    std::string subdir_name;
    bool write_eigenvalues = false;
};

struct TimestreamConfig {
    bool enabled = true;
    TodType type = TodType::xs;
    TimestreamOutputConfig output;
};

inline void validate(const TimestreamConfig &config, ValidationReport &report) {
    if (!config.enabled) {
        report.add_error({"timestream", "enabled"},
                         "false is not supported by the current pipeline");
    }
}

}  // namespace citlali::config
