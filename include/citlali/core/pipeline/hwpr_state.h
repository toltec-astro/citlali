#pragma once

namespace citlali::pipeline {

template <class Engine>
void disable_hwpr_loading(Engine &engine) {
    engine.calib.run_hwpr = false;
}

template <class Engine, class Logger>
void log_hwpr_ignored_if_needed(const Engine &engine, const Logger &logger) {
    if (!engine.calib.run_hwpr) {
        logger->info("ignoring hwpr");
    }
}

}  // namespace citlali::pipeline
