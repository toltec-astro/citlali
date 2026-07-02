#pragma once

#include <citlali/core/pipeline/hwpr_policy.h>

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

template <class Engine, class Logger>
void load_hwpr_file(Engine &engine, const std::string &filepath,
                    const Logger &logger) {
    logger->info("getting hwpr file {}", filepath);
    engine.calib.get_hwpr(filepath, engine.telescope.sim_obs);
}

template <class Engine, class RawObs, class Logger>
void load_hwpr_data_if_requested(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger) {
    if (should_load_hwpr_for_polarization(engine)) {
        std::string hwpr_filepath;
        if (should_use_raw_hwpr_data(engine, rawobs)) {
            hwpr_filepath = hwpr_data_filepath(rawobs);
            if (is_valid_hwpr_filepath(hwpr_filepath)) {
                load_hwpr_file(engine, hwpr_filepath, logger);
            }
            else {
                disable_hwpr_loading(engine);
            }
        }
        else {
            disable_hwpr_loading(engine);
        }
        log_hwpr_ignored_if_needed(engine, logger);
    }
}

template <class Engine, class RawObs, class Logger>
void load_reduction_observation_hwpr_data_if_requested(
    Engine &engine, const RawObs &rawobs, const Logger &logger) {
    load_hwpr_data_if_requested(engine, rawobs, logger);
}

}  // namespace citlali::pipeline
