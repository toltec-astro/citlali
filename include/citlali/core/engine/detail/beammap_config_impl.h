#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_config_loading.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_beammap_config(CT &config) {
    logger->info("getting beammap config options");
    auto &beammap_config = citlali::pipeline::beammap_config(*this);
    auto &beammap_plan = citlali::pipeline::beammap_plan(*this);
    auto &config_diag = citlali::pipeline::config_diagnostics(*this);
    const auto read_result =
        citlali::pipeline::read_beammap_request_config(
            config, config_diag, toltec_io.array_name_map.size());
    beammap_plan.reset_from_request(
        read_result.request, read_result.presence,
        citlali::config::mapmaking_active(
            citlali::pipeline::mapmaking_config(*this)));
    citlali::pipeline::log_beammap_effective_resolution(
        beammap_plan, logger);
    citlali::pipeline::install_beammap_effective_compatibility_config(
        beammap_plan, beammap_config);
    citlali::pipeline::sync_beammap_map_fitter(
        beammap_plan.effective().fitting, map_fitter);
}
