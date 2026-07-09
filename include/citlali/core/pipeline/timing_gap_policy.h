#pragma once

#include <citlali/core/pipeline/runtime_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool has_timing_gaps(const Engine &engine) {
    return engine.gaps.size() > 0;
}

template <class Engine>
bool should_write_timing_gaps_log(const Engine &engine) {
    return verbose_runtime_enabled(engine);
}

template <class Engine, class Logger>
void warn_timing_gaps_found(const Engine &engine, const Logger &logger) {
    logger->warn("gaps found in obnsum {} data file timing!",
                 engine.obsnum);
}

}  // namespace citlali::pipeline
