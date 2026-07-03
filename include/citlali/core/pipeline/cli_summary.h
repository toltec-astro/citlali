#pragma once

#include <string>

namespace citlali::pipeline {

inline const char *tod_output_mode_label(bool mini_output) {
    return mini_output ? "mini" : "full";
}

inline const char *tod_outer_mode_suffix(bool outer_output) {
    return outer_output ? "_outer" : "";
}

inline double physical_memory_gb(long long physical_memory_kb) {
    return static_cast<double>(physical_memory_kb) / 1e7;
}

inline bool should_report_rtc_tod_output(const std::string &tod_output_type) {
    return tod_output_type == "rtc" || tod_output_type == "both";
}

inline bool should_report_ptc_tod_output(const std::string &tod_output_type) {
    return tod_output_type == "ptc" || tod_output_type == "both";
}

template <class Logger>
void log_diagnostics_sidecar_summary(const Logger &logger) {
    logger->info("RTC diagnostics sidecar output: standard");
    logger->info("PTC diagnostics sidecar output: standard");
    logger->info("Map diagnostics sidecar output: standard");
}

template <class Logger>
void log_physical_memory_summary(const Logger &logger,
                                 long long physical_memory_kb) {
    if (physical_memory_kb >= 0) {
        logger->info("physical memory used {:.2f} GB",
                     physical_memory_gb(physical_memory_kb));
    }
    else {
        logger->debug("physical memory used unavailable on this platform");
    }
}

template <class Logger>
void log_rtc_tod_output_summary(const Logger &logger,
                                long long n_output_scans,
                                bool mini_output, bool outer_output) {
    logger->info("RTC TOD output scans: {}", n_output_scans);
    logger->info("RTC TOD output mode: {}{}",
                 tod_output_mode_label(mini_output),
                 tod_outer_mode_suffix(outer_output));
}

template <class Logger>
void log_ptc_tod_output_summary(const Logger &logger,
                                long long n_output_scans,
                                bool mini_output) {
    logger->info("PTC TOD output scans: {}", n_output_scans);
    logger->info("PTC TOD output mode: {}", tod_output_mode_label(mini_output));
}

template <class MapBuffer>
double map_buffer_memory_gb(const MapBuffer &mb) {
    return 8 * mb.n_rows * mb.n_cols *
           (mb.signal.size() + mb.weight.size() + mb.kernel.size() +
            mb.coverage.size() + mb.grid_weight.size()) /
           1e9;
}

template <class MapBuffer>
double noise_buffer_memory_gb(const MapBuffer &mb) {
    return 8 * mb.n_rows * mb.n_cols * mb.noise.size() * mb.n_noise / 1e9;
}

}  // namespace citlali::pipeline
