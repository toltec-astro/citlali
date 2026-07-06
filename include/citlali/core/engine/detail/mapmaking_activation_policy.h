#pragma once

namespace citlali::engine_detail {

template <class ReductionConfig>
void disable_map_products_if_mapmaking_disabled(
    bool run_mapmaking, bool &run_coadd, bool &run_noise,
    bool &run_map_filter, bool &run_source_finder,
    ReductionConfig &typed_config, int &beammap_iter_max) {
    if (run_mapmaking) {
        return;
    }
    run_coadd = false;
    run_noise = false;
    run_map_filter = false;
    run_source_finder = false;
    typed_config.coadd.enabled = false;
    typed_config.noise.enabled = false;
    typed_config.post_processing.map_filtering_enabled = false;
    typed_config.post_processing.map_filtering.enabled = false;
    typed_config.post_processing.source_finding_enabled = false;
    typed_config.post_processing.source_finding.enabled = false;
    typed_config.post_processing.source_fitting.active = false;
    // We don't need to do iterations if no maps are made.
    beammap_iter_max = 1;
    typed_config.beammap.iteration.max_iterations = 1;
}

}  // namespace citlali::engine_detail
