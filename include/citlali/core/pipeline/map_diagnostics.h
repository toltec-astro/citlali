#pragma once

namespace citlali::pipeline {

template <class MapBuffer, class Logger>
void calculate_map_psd_with_log(MapBuffer &map_buffer, const Logger &logger,
                                const char *log_message) {
    logger->info("{}", log_message);
    map_buffer.calc_map_psd();
}

template <class MapBuffer, class Logger>
void calculate_map_histogram_with_log(MapBuffer &map_buffer,
                                      const Logger &logger,
                                      const char *log_message) {
    logger->info("{}", log_message);
    map_buffer.calc_map_hist();
}

template <class MapBuffer>
void calculate_map_median_statistics(MapBuffer &map_buffer) {
    map_buffer.calc_median_err();
    map_buffer.calc_median_rms();
}

template <class MapBuffer, class Logger>
void calculate_map_diagnostics(MapBuffer &map_buffer, const Logger &logger,
                               const char *psd_log_message,
                               const char *histogram_log_message) {
    calculate_map_psd_with_log(map_buffer, logger, psd_log_message);
    calculate_map_histogram_with_log(
        map_buffer, logger, histogram_log_message);
    calculate_map_median_statistics(map_buffer);
}

}  // namespace citlali::pipeline
