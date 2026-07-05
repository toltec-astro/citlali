#pragma once

// Included by tod_output_selection.h inside namespace citlali::pipeline.

inline std::optional<std::string> requested_tod_output_type_name(
    bool raw_time_chunk_enabled, bool processed_time_chunk_enabled) {
    if (raw_time_chunk_enabled && processed_time_chunk_enabled) {
        return "both";
    }
    if (raw_time_chunk_enabled) {
        return "rtc";
    }
    if (processed_time_chunk_enabled) {
        return "ptc";
    }
    return std::nullopt;
}

inline void apply_tod_output_mode_flags(const std::string &mode,
                                        bool &mini, bool &outer) {
    mini = (mode == "mini" || mode == "mini_outer");
    outer = (mode == "full_outer" || mode == "mini_outer");
}

inline void align_legacy_tod_output_selection(
    bool raw_time_chunk_enabled, bool processed_time_chunk_enabled,
    bool raw_chunk_select_enabled, bool processed_chunk_select_enabled,
    const std::vector<Eigen::Index> &raw_output_chunks,
    const std::vector<Eigen::Index> &processed_output_chunks,
    bool &legacy_chunk_select_enabled,
    std::vector<Eigen::Index> &legacy_output_chunks) {
    if (raw_time_chunk_enabled) {
        legacy_chunk_select_enabled = raw_chunk_select_enabled;
        legacy_output_chunks = raw_output_chunks;
    }
    else if (processed_time_chunk_enabled) {
        legacy_chunk_select_enabled = processed_chunk_select_enabled;
        legacy_output_chunks = processed_output_chunks;
    }
    else {
        legacy_chunk_select_enabled = false;
        legacy_output_chunks.clear();
    }
}

