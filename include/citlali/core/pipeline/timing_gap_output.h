#pragma once

#include <citlali/core/pipeline/timing_gap_log_file.h>

#include <fstream>

namespace citlali::pipeline {

template <class Engine>
bool has_timing_gaps(const Engine &engine) {
    return engine.gaps.size() > 0;
}

template <class Engine>
bool should_write_timing_gaps_log(const Engine &engine) {
    return engine.verbose_mode;
}

template <class Engine, class Logger>
void warn_timing_gaps_found(const Engine &engine, const Logger &logger) {
    logger->warn("gaps found in obnsum {} data file timing!",
                 engine.obsnum);
}

template <class Engine, class Logger>
void record_timing_gaps_if_needed(const Engine &engine, const Logger &logger) {
    if (has_timing_gaps(engine)) {
        warn_timing_gaps_found(engine, logger);
        if (should_write_timing_gaps_log(engine)) {
            logger->debug("writing gaps.log file");
            std::ofstream f;
            f.open(gaps_log_filepath(engine.obsnum_dir_name));
            write_timing_gaps_log_header(f);
            for (auto const &[key, val] : engine.gaps) {
                log_timing_gap_entry(key, val, logger);
                write_timing_gap_entry(f, key, val);
            }
            f.close();
        }
    }
}

template <class Engine, class Logger>
void record_reduction_observation_timing_gaps_if_needed(
    const Engine &engine, const Logger &logger) {
    record_timing_gaps_if_needed(engine, logger);
}

}  // namespace citlali::pipeline
