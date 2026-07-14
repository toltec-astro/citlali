#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/pointing_config_read.h>
#include <citlali/core/pipeline/pointing_config_adapter.h>
#include <citlali/core/pipeline/pointing_config_logging.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/output_policy.h>

template<typename CT>
void Engine::get_pointing_config(CT &config) {
    logger->info("getting pointing config options");
    auto &pointing_plan = citlali::pipeline::pointing_plan(*this);
    pointing_plan = {};
    auto &pointing_request =
        citlali::pipeline::reduction_config(*this).pointing;
    pointing_request = citlali::config::PointingConfig{};
    auto &diagnostics = citlali::pipeline::config_diagnostics(*this);

    double default_header_max_radius_arcsec = 0.0;
    if (citlali::config::is_standard_pointing_source_strategy(
            pointing_request.source_strategy) &&
        std::isfinite(map_fitter.fitting_region_pix) && map_fitter.fitting_region_pix > 0.0 &&
        std::isfinite(omb.pixel_size_rad) && omb.pixel_size_rad > 0.0) {
        default_header_max_radius_arcsec =
            map_fitter.fitting_region_pix * omb.pixel_size_rad * RAD_TO_ASEC;
    }

    const auto request_presence =
        citlali::pipeline::read_pointing_request_config(
            config, pointing_request, diagnostics);
    pointing_plan.reset_from_request(
        pointing_request, request_presence,
        citlali::pipeline::mapmaking_plan(*this).effective.enabled,
        citlali::pipeline::map_filter_outputs_enabled(*this),
        citlali::pipeline::coadd_outputs_enabled(*this),
        default_header_max_radius_arcsec);
    citlali::pipeline::adapt_pointing_config_one_way(
        pointing_plan.effective, ptcproc);

    citlali::pipeline::log_pointing_config(
        pointing_plan.effective,
        citlali::pipeline::fruit_loops_config(*this),
        logger);
}
