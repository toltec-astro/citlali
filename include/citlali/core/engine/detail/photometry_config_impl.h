#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_source_flux_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    auto &photometry =
        citlali::pipeline::beammap_photometry_config(*this);
    auto observation =
        citlali::pipeline::read_beammap_photometry_config(config);

    if (citlali::pipeline::runtime_reduction_type(*this) ==
        citlali::config::ReductionType::beammap) {
        citlali::pipeline::require_valid_beammap_source_fluxes(
            observation, toltec_io.array_name_map, logger);
    }
    citlali::pipeline::install_beammap_photometry_config(
        std::move(observation), photometry, source_flux_mJy_beam,
        source_flux_MJy_Sr);
}
