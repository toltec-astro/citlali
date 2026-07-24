#pragma once

#include <citlali/core/config/timestream_config.h>

#include <string>

namespace citlali::pipeline {

inline const char *tod_output_mode_label(bool mini_output) {
    return mini_output ? "mini" : "full";
}

inline const char *tod_outer_mode_suffix(bool outer_output) {
    return outer_output ? "_outer" : "";
}

inline double physical_memory_gb(long long physical_memory_kb) {
    constexpr double kibibytes_per_gibibyte = 1024.0 * 1024.0;
    return static_cast<double>(physical_memory_kb) / kibibytes_per_gibibyte;
}

inline bool should_report_rtc_tod_output(
    citlali::config::TodOutputType tod_output_type) {
    return citlali::config::tod_output_includes_rtc(tod_output_type);
}

inline bool should_report_ptc_tod_output(
    citlali::config::TodOutputType tod_output_type) {
    return citlali::config::tod_output_includes_ptc(tod_output_type);
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

template <class Logger, class Obsnum, class MapBuffer>
void log_reduction_map_summary(const Logger &logger, const Obsnum &obsnum,
                               const MapBuffer &mb,
                               bool run_polarization) {
    logger->info("reduction info");
    logger->info("obsnum: {}", obsnum);
    logger->info("map buffer rows: {}", mb.n_rows);
    logger->info("map buffer cols: {}", mb.n_cols);
    logger->info("number of maps: {}", mb.signal.size());
    logger->info("map units: {}", mb.sig_unit);
    logger->info("polarized reduction: {}", run_polarization);
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

template <class Logger, class MapBuffer>
double log_observation_map_memory_summary(const Logger &logger,
                                          const MapBuffer &mb) {
    const double size_gb = map_buffer_memory_gb(mb);
    logger->info("estimated size of map buffer {:.2f} GB", size_gb);
    return size_gb;
}

template <class Logger, class MapBuffer>
double log_coadd_map_memory_summary(const Logger &logger,
                                    const MapBuffer &mb) {
    logger->info("coadd map buffer rows: {}", mb.n_rows);
    logger->info("coadd map buffer cols: {}", mb.n_cols);
    const double size_gb = map_buffer_memory_gb(mb);
    logger->info("estimated size of coadd buffer {:.2f} GB", size_gb);
    return size_gb;
}

template <class Logger, class MapBuffer>
double log_noise_map_memory_summary(const Logger &logger,
                                    const MapBuffer &mb,
                                    const char *buffer_label) {
    logger->info("{} map buffer noise maps: {}", buffer_label, mb.n_noise);
    const double size_gb = noise_buffer_memory_gb(mb);
    logger->info("estimated size of noise buffer {:.2f} GB", size_gb);
    return size_gb;
}

template <class Logger, class ObsMapBuffer, class CoaddMapBuffer>
double log_map_memory_summary(const Logger &logger, const ObsMapBuffer &omb,
                              const CoaddMapBuffer &cmb, bool run_coadd,
                              bool run_noise) {
    double total_gb = log_observation_map_memory_summary(logger, omb);
    if (run_coadd) {
        total_gb += log_coadd_map_memory_summary(logger, cmb);
        if (run_noise) {
            total_gb += log_noise_map_memory_summary(logger, cmb, "coadd");
        }
    }
    else if (run_noise) {
        total_gb += log_noise_map_memory_summary(logger, omb, "observation");
    }
    return total_gb;
}

template <class Logger>
void log_tod_output_selection_summary(
    const Logger &logger, citlali::config::TodOutputType tod_output_type,
    long long n_rtc_output_scans, bool rtc_mini_output,
    bool rtc_outer_output, long long n_ptc_output_scans,
    bool ptc_mini_output) {
    if (should_report_rtc_tod_output(tod_output_type)) {
        log_rtc_tod_output_summary(
            logger, n_rtc_output_scans, rtc_mini_output, rtc_outer_output);
    }
    if (should_report_ptc_tod_output(tod_output_type)) {
        log_ptc_tod_output_summary(
            logger, n_ptc_output_scans, ptc_mini_output);
    }
}

}  // namespace citlali::pipeline
