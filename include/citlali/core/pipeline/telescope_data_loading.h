#pragma once

#include <citlali/core/pipeline/map_center_override.h>
#include <citlali/core/pipeline/simulated_observation_indices.h>

namespace citlali::pipeline {

template <class TodProc, class RawObs, class Logger>
void load_and_align_telescope_data(TodProc &todproc, const RawObs &rawobs,
                                   const Logger &logger) {
    auto &engine = todproc.engine();

    auto tel_path = rawobs.teldata().filepath();
    logger->info("getting telescope file {}", tel_path);
    engine.telescope.get_tel_data(tel_path);

    overwrite_map_center_if_configured(engine, logger);

    if (!engine.telescope.sim_obs) {
        logger->info("aligning timestreams");
        if (engine.interp_over_gaps) {
            todproc.align_timestreams_gaps(rawobs);
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
