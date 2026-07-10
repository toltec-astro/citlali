#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_source_flux_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    auto &source_config = citlali::pipeline::beammap_config(*this).source;
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    source_config = citlali::config::BeammapSourceConfig{};

    citlali::pipeline::read_beammap_source_identity_config(
        config, source_config, config_diag);
    citlali::pipeline::read_beammap_source_fluxes(
        config, source_flux_mJy_beam, source_config);

    if (citlali::pipeline::runtime_config(*this).reduction_type ==
            citlali::config::ReductionType::beammap &&
        !citlali::pipeline::validate_beammap_source_fluxes(
            source_flux_mJy_beam, toltec_io.array_name_map, logger)) {
        std::exit(EXIT_FAILURE);
    }
}
