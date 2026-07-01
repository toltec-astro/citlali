#pragma once

#include <citlali/core/pipeline/observation_preflight.h>

#include <cstddef>

namespace citlali::pipeline {

template <class TodProc, class MapExtents, class MapCoords, class Logger>
void calculate_initial_observation_map_dimensions(TodProc &todproc,
                                                 MapExtents &map_extents,
                                                 MapCoords &map_coords,
                                                 const Logger &logger) {
    auto &engine = todproc.engine();

    if (!engine.run_mapmaking) {
        return;
    }

    logger->info("calculating number of maps");
    todproc.calc_map_num();
    logger->info("calculating obs map dimensions");
    todproc.calc_omb_size(map_extents, map_coords);
}

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class Logger>
bool prepare_initial_observation_setup(TodProc &todproc, const RawObs &rawobs,
                                       const RawObsKidsMeta &rawobs_kids_meta,
                                       MapExtents &map_extents,
                                       MapCoords &map_coords,
                                       const Logger &logger) {
    auto &engine = todproc.engine();

    configure_observation_calibration<IsBeammap>(todproc, rawobs, logger);
    if (!apply_flxscale_correction(engine, rawobs, logger)) {
        return false;
    }

    check_observation_inputs(todproc, rawobs, logger);
    update_sample_rate_from_rawobs_meta(engine, rawobs_kids_meta, logger);
    load_and_align_telescope_data(todproc, rawobs, logger);
    calculate_telescope_pointing(todproc, logger);
    calculate_scan_indices(engine, logger);
    calculate_initial_observation_map_dimensions(
        todproc, map_extents, map_coords, logger);
    return true;
}

template <bool IsBeammap, class KidsDataProc, class TodProc,
          class CitlaliConfig, class RawObs, class MapExtents,
          class MapCoords, class Logger>
bool prepare_initial_observation(
    TodProc &todproc, CitlaliConfig &citlali_config, const RawObs &rawobs,
    MapExtents &map_extents, MapCoords &map_coords, const Logger &logger) {
    auto kidsproc = make_kids_data_proc<KidsDataProc>(citlali_config);
    auto rawobs_kids_meta = load_rawobs_kids_meta(kidsproc, rawobs, logger);

    return prepare_initial_observation_setup<IsBeammap>(
        todproc, rawobs, rawobs_kids_meta, map_extents, map_coords, logger);
}

template <bool IsBeammap, class KidsDataProc, class TodProc,
          class IOCoordinator, class CitlaliConfig, class MapExtents,
          class MapCoords, class Logger>
bool prepare_initial_observations(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    MapExtents &map_extents, MapCoords &map_coords, const Logger &logger) {
    logger->info("starting initial loop through input obs");
    std::size_t observation_index = 0;
    for (const auto &rawobs : co.inputs()) {
        logger->info("starting setup of observation {}/{}",
                     observation_index + 1, co.n_inputs());
        if (!prepare_initial_observation<IsBeammap, KidsDataProc>(
                todproc, citlali_config, rawobs, map_extents, map_coords,
                logger)) {
            return false;
        }
        ++observation_index;
    }
    return true;
}

template <class TodProc, class MapCoords, class Logger>
void calculate_initial_coadd_map_dimensions(TodProc &todproc,
                                            MapCoords &map_coords,
                                            const Logger &logger) {
    auto &engine = todproc.engine();

    if (!engine.run_coadd) {
        return;
    }

    logger->info("calculating cmb dimensions");
    todproc.calc_cmb_size(map_coords);
}

template <bool IsBeammap, class KidsDataProc, class TodProc,
          class IOCoordinator, class CitlaliConfig, class MapExtents,
          class MapCoords, class Logger>
bool prepare_initial_reduction_geometry(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    MapExtents &map_extents, MapCoords &map_coords, const Logger &logger) {
    if (!prepare_initial_observations<IsBeammap, KidsDataProc>(
            todproc, co, citlali_config, map_extents, map_coords, logger)) {
        return false;
    }

    calculate_initial_coadd_map_dimensions(todproc, map_coords, logger);
    return true;
}

}  // namespace citlali::pipeline
