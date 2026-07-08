#pragma once

#include <citlali/core/config/coadd_config.h>
#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/post_processing_config.h>

namespace citlali::engine_detail {

template <class ReductionConfig>
void disable_map_products_if_mapmaking_disabled(
    bool &run_coadd, bool &run_noise, bool &run_map_filter,
    bool &run_source_finder, ReductionConfig &typed_config,
    int &beammap_iter_max) {
    if (citlali::config::mapmaking_active(typed_config.mapmaking)) {
        return;
    }
    run_coadd = false;
    run_noise = false;
    run_map_filter = false;
    run_source_finder = false;
    citlali::config::set_coadd_enabled(typed_config.coadd, false);
    citlali::config::set_noise_maps_enabled(typed_config.noise, false);
    citlali::config::set_map_filtering_enabled(
        typed_config.post_processing, false);
    citlali::config::set_source_finding_enabled(
        typed_config.post_processing, false);
    citlali::config::set_source_fitting_active(
        typed_config.post_processing, false);
    // We don't need to do iterations if no maps are made.
    beammap_iter_max = 1;
    typed_config.beammap.iteration.max_iterations = 1;
}

}  // namespace citlali::engine_detail
