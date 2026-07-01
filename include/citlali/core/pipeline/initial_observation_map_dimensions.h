#pragma once

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

}  // namespace citlali::pipeline
