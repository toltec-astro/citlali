#pragma once

#include <citlali/core/pipeline/telescope_data_loading.h>

namespace citlali::pipeline {

template <class Engine, class Logger>
void calculate_tangent_plane_pointing(Engine &engine, const Logger &logger) {
    logger->info("calculating tangent plane pointing");
    engine.telescope.calc_tan_pointing();
}

template <class TodProc, class Logger>
void interpolate_pointing_offsets(TodProc &todproc, const Logger &logger) {
    logger->info("calculating pointing offsets");
    todproc.interp_pointing();
}

template <class TodProc, class Logger>
void calculate_telescope_pointing(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    calculate_tangent_plane_pointing(engine, logger);
    interpolate_pointing_offsets(todproc, logger);
}

template <class TodProc, class RawObs, class Logger>
void load_and_point_telescope_data_if_needed(TodProc &todproc,
                                             const RawObs &rawobs,
                                             bool should_load,
                                             const Logger &logger) {
    if (!should_load) {
        return;
    }

    load_and_align_telescope_data(todproc, rawobs, logger);
    calculate_telescope_pointing(todproc, logger);
}

template <class TodProc, class RawObs, class Logger>
void load_and_point_reduction_observation_telescope_data_if_needed(
    TodProc &todproc, const RawObs &rawobs, bool should_load,
    const Logger &logger) {
    load_and_point_telescope_data_if_needed(
        todproc, rawobs, should_load, logger);
}

}  // namespace citlali::pipeline
