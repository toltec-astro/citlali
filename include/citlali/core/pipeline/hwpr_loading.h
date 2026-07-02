#pragma once

#include <string>

namespace citlali::pipeline {

template <class Engine>
bool should_load_hwpr_for_polarization(const Engine &engine) {
    return engine.rtcproc.run_polarization;
}

template <class Engine, class RawObs, class Logger>
void load_hwpr_data_if_requested(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger) {
    if (engine.rtcproc.run_polarization) {
        std::string hwpr_filepath;
        if (rawobs.hwpdata().has_value() && engine.calib.ignore_hwpr != "true") {
            hwpr_filepath = rawobs.hwpdata()->filepath();
            if (hwpr_filepath != "null") {
                logger->info("getting hwpr file {}", hwpr_filepath);
                engine.calib.get_hwpr(hwpr_filepath, engine.telescope.sim_obs);
            }
            else {
                engine.calib.run_hwpr = false;
            }
        }
        else {
            engine.calib.run_hwpr = false;
        }
        if (!engine.calib.run_hwpr) {
            logger->info("ignoring hwpr");
        }
    }
}

template <class Engine, class RawObs, class Logger>
void load_reduction_observation_hwpr_data_if_requested(
    Engine &engine, const RawObs &rawobs, const Logger &logger) {
    load_hwpr_data_if_requested(engine, rawobs, logger);
}

}  // namespace citlali::pipeline
