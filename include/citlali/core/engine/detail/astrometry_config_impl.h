#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/pointing_offsets_config_read.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <utility>

template<typename CT>
void Engine::get_astrometry_config(CT &config) {
    auto &astrometry_config = citlali::pipeline::astrometry_config(*this);
    auto observation =
        citlali::pipeline::read_astrometry_config(config, logger);
    citlali::pipeline::require_valid_astrometry_config(observation, logger);
    citlali::pipeline::install_astrometry_config(
        std::move(observation), astrometry_config, pointing_offsets);
}
