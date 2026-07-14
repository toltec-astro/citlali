#pragma once

#include <citlali/core/config/mapmaking_config.h>

namespace citlali::pipeline {

template <class ReductionConfig>
void normalize_beammap_iterations_if_mapmaking_disabled(
    ReductionConfig &reduction_config) {
    if (citlali::config::mapmaking_active(reduction_config.mapmaking)) {
        return;
    }
    // We don't need to do iterations if no maps are made.
    reduction_config.beammap.iteration.max_iterations = 1;
}

}  // namespace citlali::pipeline
