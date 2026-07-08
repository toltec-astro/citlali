#pragma once

// Included by tod_output_selection.h inside namespace citlali::pipeline.

inline std::optional<citlali::config::TodOutputType> requested_tod_output_type(
    bool raw_time_chunk_enabled, bool processed_time_chunk_enabled) {
    const auto output_type = citlali::config::enabled_tod_output_type(
        raw_time_chunk_enabled, processed_time_chunk_enabled);
    if (citlali::config::is_tod_output_enabled(output_type)) {
        return output_type;
    }
    return std::nullopt;
}

inline void apply_tod_output_mode_flags(
    citlali::config::TodStreamOutputMode mode,
                                        bool &mini, bool &outer) {
    mini = citlali::config::is_mini_tod_stream_output_mode(mode);
    outer = citlali::config::is_outer_tod_stream_output_mode(mode);
}
