#pragma once

#include <string>

namespace citlali::pipeline {

inline double degrees_to_radians(double degrees) {
    constexpr double deg_to_rad = 0.017453292519943295769;
    return degrees * deg_to_rad;
}

template <class Engine, class RawObs>
void reset_simulated_observation_indices(Engine &engine,
                                         const RawObs &rawobs) {
    engine.start_indices.clear();
    engine.end_indices.clear();

    for (const auto &data_item : rawobs.kidsdata()) {
        (void)data_item;
        engine.start_indices.push_back(0);
        engine.start_indices.push_back(0);
    }

    if (engine.calib.run_hwpr) {
        engine.hwpr_start_indices = 0;
        engine.hwpr_end_indices = 0;
    }
}

template <class Engine, class Logger>
void overwrite_map_center_if_configured(Engine &engine, const Logger &logger) {
    if (engine.omb.crval_config[0] != 0 && engine.omb.crval_config[1] != 0) {
        logger->info("overwriting map center to ({}, {})",
                     engine.omb.crval_config[0], engine.omb.crval_config[1]);
        const double map_center_ra_rad =
            degrees_to_radians(engine.omb.crval_config[0]);
        const double map_center_dec_rad =
            degrees_to_radians(engine.omb.crval_config[1]);
        engine.telescope.tel_header["Header.Source.Ra"].setConstant(
            map_center_ra_rad);
        engine.telescope.tel_header["Header.Source.Dec"].setConstant(
            map_center_dec_rad);
    }
}

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
