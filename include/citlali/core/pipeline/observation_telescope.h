#pragma once

#include <citlali/core/pipeline/map_center_override.h>
#include <citlali/core/pipeline/simulated_observation_indices.h>
#include <citlali/core/pipeline/telescope_data_loading.h>

#include <string>

namespace citlali::pipeline {

template <class TodProc, class Logger>
void calculate_telescope_pointing(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("calculating tangent plane pointing");
    engine.telescope.calc_tan_pointing();

    logger->info("calculating pointing offsets");
    todproc.interp_pointing();
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

template <class Engine, class Logger>
void calculate_scan_indices(Engine &engine, const Logger &logger) {
    logger->info("calculating scan indices");
    engine.telescope.calc_scan_indices();
}

template <class Engine, class Logger>
void calculate_scan_indices_if_needed(Engine &engine, bool should_calculate,
                                      const Logger &logger) {
    if (!should_calculate) {
        return;
    }

    calculate_scan_indices(engine, logger);
}

}  // namespace citlali::pipeline
