#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/pointing_config_read.h>
#include <citlali/core/pipeline/pointing_config_logging.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_pointing_config(CT &config) {
    logger->info("getting pointing config options");
    auto &pointing_config = citlali::pipeline::pointing_config(*this);
    pointing_config = citlali::config::PointingConfig{};
    auto &diagnostics = citlali::pipeline::config_diagnostics(*this);

    double default_header_max_radius_arcsec = 0.0;
    if (citlali::config::is_standard_pointing_source_strategy(
            pointing_config.source_strategy) &&
        std::isfinite(map_fitter.fitting_region_pix) && map_fitter.fitting_region_pix > 0.0 &&
        std::isfinite(omb.pixel_size_rad) && omb.pixel_size_rad > 0.0) {
        default_header_max_radius_arcsec =
            map_fitter.fitting_region_pix * omb.pixel_size_rad * RAD_TO_ASEC;
    }

    citlali::pipeline::read_pointing_source_strategy_config(
        config, default_header_max_radius_arcsec, pointing_config,
        diagnostics);

    ptcproc.fruit_loops_source_center_mode =
        std::string(citlali::config::to_string(
            pointing_config.fruitloops_center_mode));
    ptcproc.fruit_loops_header_center_max_radius_arcsec =
        pointing_config.header_max_radius_arcsec;
    ptcproc.fruit_loops_header_center_require_coverage =
        pointing_config.header_require_coverage;

    citlali::pipeline::log_pointing_config(
        pointing_config, ptcproc, logger);
}
