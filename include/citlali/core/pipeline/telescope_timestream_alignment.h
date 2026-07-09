#pragma once

#include <citlali/core/pipeline/simulated_observation_indices.h>

namespace citlali::pipeline {

template <class Engine>
bool should_align_telescope_timestreams(const Engine &engine) {
    return !engine.telescope.sim_obs;
}

template <class Engine>
bool should_interpolate_over_timing_gaps(const Engine &engine) {
    return citlali::config::timing_gap_interpolation_active(
        engine.typed_config.runtime);
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
void align_telescope_timestreams(TodProc &todproc, const RawObs &rawobs,
                                 const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("aligning timestreams");
    if (should_interpolate_over_timing_gaps(engine)) {
        align_telescope_timestreams_over_gaps(todproc, rawobs);
    }
    else {
        align_telescope_timestreams_direct(todproc, rawobs);
    }
}

template <class Engine, class RawObs>
void reset_simulated_telescope_indices(Engine &engine, const RawObs &rawobs) {
    reset_simulated_observation_indices(engine, rawobs);
}

}  // namespace citlali::pipeline
