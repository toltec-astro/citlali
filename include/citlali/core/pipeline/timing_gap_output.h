#pragma once

#include <citlali/core/pipeline/timing_gap_log_file.h>
#include <citlali/core/pipeline/timing_gap_policy.h>

namespace citlali::pipeline {

template <class Engine, class Logger>
void record_timing_gaps_if_needed(const Engine &engine, const Logger &logger) {
    if (has_timing_gaps(engine)) {
        warn_timing_gaps_found(engine, logger);
        if (should_write_timing_gaps_log(engine)) {
            write_timing_gaps_log_file(engine, logger);
        }
    }
}

template <class Engine, class Logger>
void record_reduction_observation_timing_gaps_if_needed(
    const Engine &engine, const Logger &logger) {
    record_timing_gaps_if_needed(engine, logger);
}

}  // namespace citlali::pipeline
