#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/pointing_offsets_config_read.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_astrometry_config(CT &config) {
    auto &astrometry_config = citlali::pipeline::astrometry_config(*this);
    astrometry_config = citlali::config::AstrometryConfig{};
    citlali::pipeline::read_pointing_offsets_config(
        config, pointing_offsets, astrometry_config, logger);
}
