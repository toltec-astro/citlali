#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/post_processing_config.h>

namespace citlali::pipeline {

template <class ReductionConfig>
void disable_map_products_if_mapmaking_disabled(
    ReductionConfig &reduction_config) {
    if (citlali::config::mapmaking_active(reduction_config.mapmaking)) {
        return;
    }
    citlali::config::set_noise_maps_enabled(reduction_config.noise, false);
    citlali::config::set_map_filtering_enabled(
        reduction_config.post_processing, false);
    citlali::config::set_source_finding_enabled(
        reduction_config.post_processing, false);
    citlali::config::set_source_fitting_active(
        reduction_config.post_processing, false);
    // We don't need to do iterations if no maps are made.
    reduction_config.beammap.iteration.max_iterations = 1;
}

}  // namespace citlali::pipeline
