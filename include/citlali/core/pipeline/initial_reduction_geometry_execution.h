#pragma once

#include <citlali/core/pipeline/initial_coadd_map_dimensions.h>
#include <citlali/core/pipeline/initial_observation_loop.h>

namespace citlali::pipeline {

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
