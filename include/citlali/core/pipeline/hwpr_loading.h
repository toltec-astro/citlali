#pragma once

#include <string>

namespace citlali::pipeline {

template <class Engine>
bool should_load_hwpr_for_polarization(const Engine &engine) {
    return engine.rtcproc.run_polarization;
}

template <class Engine>
bool is_hwpr_ignored_by_config(const Engine &engine) {
    return engine.calib.ignore_hwpr == "true";
}

template <class RawObs>
bool has_hwpr_data(const RawObs &rawobs) {
    return rawobs.hwpdata().has_value();
}

template <class Engine, class RawObs>
bool should_use_raw_hwpr_data(const Engine &engine, const RawObs &rawobs) {
    return has_hwpr_data(rawobs) && !is_hwpr_ignored_by_config(engine);
}

template <class Engine, class RawObs, class Logger>
void load_hwpr_data_if_requested(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger) {
    if (should_load_hwpr_for_polarization(engine)) {
        std::string hwpr_filepath;
        if (should_use_raw_hwpr_data(engine, rawobs)) {
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
