#pragma once

// Included by tod_output_selection.h inside namespace citlali::pipeline.

inline std::optional<citlali::config::TodOutputType> requested_tod_output_type(
    bool raw_time_chunk_enabled, bool processed_time_chunk_enabled) {
    if (raw_time_chunk_enabled && processed_time_chunk_enabled) {
        return citlali::config::TodOutputType::both;
    }
    if (raw_time_chunk_enabled) {
        return citlali::config::TodOutputType::rtc;
    }
    if (processed_time_chunk_enabled) {
        return citlali::config::TodOutputType::ptc;
    }
    return std::nullopt;
}

inline void apply_tod_output_mode_flags(const std::string &mode,
                                        bool &mini, bool &outer) {
    mini = (mode == "mini" || mode == "mini_outer");
    outer = (mode == "full_outer" || mode == "mini_outer");
}
