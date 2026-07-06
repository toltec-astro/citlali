#pragma once

#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class TodProc, class MapCoords, class Logger>
void calculate_initial_coadd_map_dimensions(TodProc &todproc,
                                            MapCoords &map_coords,
                                            const Logger &logger) {
    auto &engine = todproc.engine();

    if (!coadd_outputs_enabled(engine)) {
        return;
    }

    logger->info("calculating cmb dimensions");
    todproc.calc_cmb_size(map_coords);
}

}  // namespace citlali::pipeline
