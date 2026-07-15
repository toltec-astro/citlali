#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_source_flux_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    auto &source_config = citlali::pipeline::beammap_config(*this).source;
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    auto observation =
        citlali::pipeline::read_beammap_source_observation_config(
            config, config_diag);

    if (citlali::pipeline::runtime_reduction_type(*this) ==
        citlali::config::ReductionType::beammap) {
        citlali::pipeline::require_valid_beammap_source_fluxes(
            observation, toltec_io.array_name_map, logger);
    }
    citlali::pipeline::install_beammap_source_observation_config(
        std::move(observation), source_config, source_flux_mJy_beam,
        source_flux_MJy_Sr);
}
