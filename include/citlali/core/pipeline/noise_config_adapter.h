#pragma once

#include <citlali/core/config/noise_config.h>

namespace citlali::pipeline {

template <class ObservationMapBlock, class CoaddMapBlock>
void adapt_noise_config_one_way(
    const citlali::config::NoiseConfig &effective,
    bool coadd_enabled, ObservationMapBlock &observation_maps,
    CoaddMapBlock &coadd_maps) {
    const auto count = effective.enabled ? effective.n_noise_maps : 0;
    observation_maps.n_noise = count;
    observation_maps.randomize_dets = effective.randomize_dets;
    coadd_maps.n_noise = coadd_enabled ? count : 0;
    coadd_maps.randomize_dets = effective.randomize_dets;
}

}  // namespace citlali::pipeline
