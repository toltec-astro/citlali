#pragma once

#include <citlali/core/config/post_processing_config.h>

namespace citlali::pipeline {

template <class ObservationMapBuffer, class CoaddMapBuffer>
void adapt_source_finding_config_one_way(
    const citlali::config::SourceFindingConfig &config,
    double arcsec_to_rad, bool coadd_enabled,
    ObservationMapBuffer &observation_maps, CoaddMapBuffer &coadd_maps) {
    observation_maps.source_sigma = config.source_sigma;
    observation_maps.source_window_rad =
        config.source_window_arcsec * arcsec_to_rad;
    observation_maps.source_finder_mode = config.mode;

    if (!coadd_enabled) {
        return;
    }
    coadd_maps.source_sigma = config.source_sigma;
    coadd_maps.source_window_rad =
        config.source_window_arcsec * arcsec_to_rad;
    coadd_maps.source_finder_mode = config.mode;
}

}  // namespace citlali::pipeline
