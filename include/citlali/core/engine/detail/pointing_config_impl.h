#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/engine/detail/pointing_config_logging.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

template<typename CT>
void Engine::get_pointing_config(CT &config) {
    logger->info("getting pointing config options");
    auto &pointing_config = citlali::pipeline::pointing_config(*this);
    pointing_config = citlali::config::PointingConfig{};
    auto &diagnostics = config_diagnostics;

    std::string pointing_source_strategy = "standard";
    citlali::engine_detail::read_optional_parsed_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","mode"},
        pointing_source_strategy, pointing_config.source_strategy,
        citlali::config::parse_pointing_source_strategy, diagnostics,
        {"standard", "psf_preserve"});

    bool pointing_fit_gaussian_enabled =
        citlali::config::is_standard_pointing_source_strategy(
            pointing_config.source_strategy);
    pointing_config.fit_gaussian = pointing_fit_gaussian_enabled;
    citlali::engine_detail::read_optional_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","fit_gaussian"},
        pointing_fit_gaussian_enabled, pointing_config.fit_gaussian,
        diagnostics);

    std::string pointing_fruitloops_center_mode =
        citlali::config::is_psf_preserve_pointing_source_strategy(
            pointing_config.source_strategy) ? "map_center" : "auto";
    if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
            pointing_fruitloops_center_mode)) {
        pointing_config.fruitloops_center_mode = *parsed;
    }
    citlali::engine_detail::read_optional_parsed_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","fruitloops_center_mode"},
        pointing_fruitloops_center_mode,
        pointing_config.fruitloops_center_mode,
        citlali::config::parse_fruit_loops_center_mode, diagnostics,
        {"auto", "header", "peak", "map_center"});

    double pointing_header_center_max_radius_arcsec = 0.0;
    if (citlali::config::is_standard_pointing_source_strategy(
            pointing_config.source_strategy) &&
        std::isfinite(map_fitter.fitting_region_pix) && map_fitter.fitting_region_pix > 0.0 &&
        std::isfinite(omb.pixel_size_rad) && omb.pixel_size_rad > 0.0) {
        pointing_header_center_max_radius_arcsec =
            map_fitter.fitting_region_pix * omb.pixel_size_rad * RAD_TO_ASEC;
    }
    pointing_config.header_max_radius_arcsec =
        pointing_header_center_max_radius_arcsec;
    citlali::engine_detail::read_optional_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","header_max_radius_arcsec"},
        pointing_header_center_max_radius_arcsec,
        pointing_config.header_max_radius_arcsec, diagnostics, {}, {0.0});

    bool pointing_header_center_require_coverage = true;
    pointing_config.header_require_coverage =
        pointing_header_center_require_coverage;
    citlali::engine_detail::read_optional_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","header_require_coverage"},
        pointing_header_center_require_coverage,
        pointing_config.header_require_coverage, diagnostics);

    ptcproc.fruit_loops_source_center_mode =
        std::string(citlali::config::to_string(
            pointing_config.fruitloops_center_mode));
    ptcproc.fruit_loops_header_center_max_radius_arcsec =
        pointing_config.header_max_radius_arcsec;
    ptcproc.fruit_loops_header_center_require_coverage =
        pointing_config.header_require_coverage;

    citlali::engine_detail::log_pointing_config(
        pointing_config, ptcproc, logger);
}
