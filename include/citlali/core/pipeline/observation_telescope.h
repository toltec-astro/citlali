#pragma once

#include <citlali/core/pipeline/map_center_override.h>
#include <citlali/core/pipeline/simulated_observation_indices.h>

#include <string>

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
