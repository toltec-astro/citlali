#pragma once

#include <citlali/core/pipeline/initial_observation_setup.h>
#include <citlali/core/pipeline/kids_metadata.h>

#include <cstddef>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
