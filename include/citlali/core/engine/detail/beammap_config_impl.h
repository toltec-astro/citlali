#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/beammap_config_loading.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_beammap_config(CT &config) {
    logger->info("getting beammap config options");
    auto &beammap_config = citlali::pipeline::beammap_config(*this);
    auto &config_diag = config_diagnostics;
    const auto beammap_core_config =
        citlali::pipeline::read_beammap_core_config(
            config, config_diag, logger);

    const auto beammap_fitting_config =
        citlali::pipeline::read_beammap_fitting_config(
            config, config_diag);

    const auto beammap_scan_band_mask_config =
        citlali::pipeline::read_beammap_scan_band_mask_config(
            config, config_diag);

    const auto beammap_split_fits_config =
        citlali::pipeline::read_beammap_split_fits_config(
            config, config_diag, logger);
    citlali::pipeline::sync_beammap_map_fitter(
        beammap_fitting_config, map_fitter);

    const auto beammap_priors_config =
        citlali::pipeline::read_beammap_priors_config(
            config, config_diag, logger);

    const auto beammap_flagging_config =
        citlali::pipeline::read_beammap_flagging_config(
            config, config_diag, toltec_io.array_name_map.size());

    const auto beammap_sensitivity_config =
        citlali::pipeline::read_beammap_sensitivity_config(
            config, config_diag.invalid_keys);

    const auto beammap_detector_tod_output_config =
        citlali::pipeline::read_beammap_detector_tod_output_config(
            config, config_diag);

    citlali::pipeline::apply_beammap_typed_config(
        beammap_config, beammap_core_config, beammap_fitting_config,
        beammap_scan_band_mask_config, beammap_split_fits_config,
        beammap_priors_config, beammap_detector_tod_output_config,
        beammap_flagging_config, beammap_sensitivity_config);
}
