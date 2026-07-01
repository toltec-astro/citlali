#pragma once

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
