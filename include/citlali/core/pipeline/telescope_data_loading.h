#pragma once

#include <citlali/core/pipeline/map_center_override.h>
#include <citlali/core/pipeline/simulated_observation_indices.h>

#include <string>

namespace citlali::pipeline {

template <class RawObs>
std::string telescope_data_filepath(const RawObs &rawobs) {
    return rawobs.teldata().filepath();
}

template <class Engine>
bool should_align_telescope_timestreams(const Engine &engine) {
    return !engine.telescope.sim_obs;
}

template <class Engine>
bool should_interpolate_over_timing_gaps(const Engine &engine) {
    return engine.interp_over_gaps;
}

template <class Engine, class Logger>
void load_telescope_data_file(Engine &engine, const std::string &filepath,
                              const Logger &logger) {
    logger->info("getting telescope file {}", filepath);
    engine.telescope.get_tel_data(filepath);
}

template <class TodProc, class RawObs>
void align_telescope_timestreams_over_gaps(TodProc &todproc,
                                           const RawObs &rawobs) {
    todproc.align_timestreams_gaps(rawobs);
}

template <class TodProc, class RawObs>
void align_telescope_timestreams_direct(TodProc &todproc,
                                        const RawObs &rawobs) {
    todproc.align_timestreams(rawobs);
}

template <class TodProc, class RawObs, class Logger>
void load_and_align_telescope_data(TodProc &todproc, const RawObs &rawobs,
                                   const Logger &logger) {
    auto &engine = todproc.engine();

    auto tel_path = telescope_data_filepath(rawobs);
    load_telescope_data_file(engine, tel_path, logger);

    overwrite_map_center_if_configured(engine, logger);

    if (should_align_telescope_timestreams(engine)) {
        logger->info("aligning timestreams");
        if (should_interpolate_over_timing_gaps(engine)) {
            align_telescope_timestreams_over_gaps(todproc, rawobs);
        }
        else {
            todproc.align_timestreams(rawobs);
        }
    }
    else {
        reset_simulated_observation_indices(engine, rawobs);
    }
}

}  // namespace citlali::pipeline
