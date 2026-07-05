#pragma once

namespace citlali::engine_detail {

template <class CoaddConfig, class NoiseConfig, class PostProcessingConfig,
          class BeammapConfig>
void disable_map_products_if_mapmaking_disabled(
    bool run_mapmaking, bool &run_coadd, bool &run_noise,
    bool &run_map_filter, bool &run_source_finder,
    CoaddConfig &typed_coadd_config, NoiseConfig &typed_noise_config,
    PostProcessingConfig &typed_post_processing_config, int &beammap_iter_max,
    BeammapConfig &typed_beammap_config) {
    if (run_mapmaking) {
        return;
    }
    run_coadd = false;
    run_noise = false;
    run_map_filter = false;
    run_source_finder = false;
    typed_coadd_config.enabled = false;
    typed_noise_config.enabled = false;
    typed_post_processing_config.map_filtering_enabled = false;
    typed_post_processing_config.map_filtering.enabled = false;
    typed_post_processing_config.source_finding_enabled = false;
    typed_post_processing_config.source_finding.enabled = false;
    typed_post_processing_config.source_fitting.active = false;
    // We don't need to do iterations if no maps are made.
    beammap_iter_max = 1;
    typed_beammap_config.iteration.max_iterations = 1;
}

}  // namespace citlali::engine_detail
