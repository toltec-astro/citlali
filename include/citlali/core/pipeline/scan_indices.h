#pragma once

namespace citlali::pipeline {

template <class Logger>
void log_scan_index_calculation(const Logger &logger) {
    logger->info("calculating scan indices");
}

template <class Engine>
void calculate_telescope_scan_indices(Engine &engine) {
    engine.telescope.calc_scan_indices(timestream_config(engine).chunking);
}

template <class Engine, class Logger>
void calculate_scan_indices(Engine &engine, const Logger &logger) {
    log_scan_index_calculation(logger);
    calculate_telescope_scan_indices(engine);
}

template <class Engine, class Logger>
void calculate_scan_indices_if_needed(Engine &engine, bool should_calculate,
                                      const Logger &logger) {
    if (!should_calculate) {
        return;
    }

    calculate_scan_indices(engine, logger);
}

template <class Engine, class Logger>
void calculate_reduction_observation_scan_indices_if_needed(
    Engine &engine, bool should_calculate, const Logger &logger) {
    calculate_scan_indices_if_needed(engine, should_calculate, logger);
}

}  // namespace citlali::pipeline
