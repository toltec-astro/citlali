#pragma once

#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class MapBuffer, class Logger>
void calculate_map_psd_with_log(MapBuffer &map_buffer,
                                StageProfileCollector &stage_profile,
                                const Logger &logger,
                                const char *log_message) {
    (void)stage_profile;
    logger->info("{}", log_message);
    const auto profile_scope =
        profile_stage("map.diagnostics.psd", logger, log_message);
    map_buffer.calc_map_psd();
}

template <class MapBuffer, class Logger>
void calculate_map_psd_with_log(MapBuffer &map_buffer, const Logger &logger,
                                const char *log_message) {
    calculate_map_psd_with_log(
        map_buffer, stage_profile_collector(), logger, log_message);
}

template <class MapBuffer, class Logger>
void calculate_map_histogram_with_log(MapBuffer &map_buffer,
    StageProfileCollector &stage_profile, const Logger &logger,
    const char *log_message) {
    (void)stage_profile;
    logger->info("{}", log_message);
    const auto profile_scope =
        profile_stage("map.diagnostics.histogram", logger, log_message);
    map_buffer.calc_map_hist();
}

template <class MapBuffer, class Logger>
void calculate_map_histogram_with_log(MapBuffer &map_buffer,
                                      const Logger &logger,
                                      const char *log_message) {
    calculate_map_histogram_with_log(
        map_buffer, stage_profile_collector(), logger, log_message);
}

template <class MapBuffer>
void calculate_map_median_statistics(MapBuffer &map_buffer) {
    map_buffer.calc_median_err();
    map_buffer.calc_median_rms();
}

template <class MapBuffer, class Logger>
void calculate_map_diagnostics(MapBuffer &map_buffer,
                               StageProfileCollector &stage_profile,
                               const Logger &logger,
                               const char *psd_log_message,
                               const char *histogram_log_message) {
    calculate_map_psd_with_log(
        map_buffer, stage_profile, logger, psd_log_message);
    calculate_map_histogram_with_log(
        map_buffer, stage_profile, logger, histogram_log_message);
    const auto profile_scope =
        profile_stage("map.diagnostics.median_statistics", logger);
    calculate_map_median_statistics(map_buffer);
}

// Temporary adapter for engine-internal callers that have not yet received
// the run-owned collector.
template <class MapBuffer, class Logger>
void calculate_map_diagnostics(MapBuffer &map_buffer, const Logger &logger,
                               const char *psd_log_message,
                               const char *histogram_log_message) {
    calculate_map_diagnostics(
        map_buffer, stage_profile_collector(), logger, psd_log_message,
        histogram_log_message);
}

}  // namespace citlali::pipeline
