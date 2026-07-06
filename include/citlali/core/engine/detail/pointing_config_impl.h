#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/engine/detail/pointing_config_logging.h>

template<typename CT>
void Engine::get_pointing_config(CT &config) {
    logger->info("getting pointing config options");
    auto &pointing_config = typed_config.pointing;
    pointing_config = citlali::config::PointingConfig{};

    pointing_source_strategy = "standard";
    citlali::engine_detail::read_optional_parsed_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","mode"},
        pointing_source_strategy, pointing_config.source_strategy,
        citlali::config::parse_pointing_source_strategy, missing_keys,
        invalid_keys, {"standard", "psf_preserve"});

    pointing_fit_gaussian_enabled = (pointing_source_strategy == "standard");
    pointing_config.fit_gaussian = pointing_fit_gaussian_enabled;
    citlali::engine_detail::read_optional_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","fit_gaussian"},
        pointing_fit_gaussian_enabled, pointing_config.fit_gaussian,
        missing_keys, invalid_keys);

    pointing_fruitloops_center_mode =
        (pointing_source_strategy == "psf_preserve") ? "map_center" : "auto";
    if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
            pointing_fruitloops_center_mode)) {
        pointing_config.fruitloops_center_mode = *parsed;
    }
    citlali::engine_detail::read_optional_parsed_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","fruitloops_center_mode"},
        pointing_fruitloops_center_mode,
        pointing_config.fruitloops_center_mode,
        citlali::config::parse_fruit_loops_center_mode, missing_keys,
        invalid_keys, {"auto", "header", "peak", "map_center"});

    pointing_header_center_max_radius_arcsec = 0.0;
    if (pointing_source_strategy == "standard" &&
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
        pointing_config.header_max_radius_arcsec, missing_keys, invalid_keys,
        {}, {0.0});

    pointing_header_center_require_coverage = true;
    pointing_config.header_require_coverage =
        pointing_header_center_require_coverage;
    citlali::engine_detail::read_optional_mirrored_config_value(
        config, std::tuple{"pointing","source_strategy","header_require_coverage"},
        pointing_header_center_require_coverage,
        pointing_config.header_require_coverage, missing_keys, invalid_keys);

    ptcproc.fruit_loops_source_center_mode = pointing_fruitloops_center_mode;
    ptcproc.fruit_loops_header_center_max_radius_arcsec =
        pointing_header_center_max_radius_arcsec;
    ptcproc.fruit_loops_header_center_require_coverage =
        pointing_header_center_require_coverage;

    citlali::engine_detail::log_pointing_config(
        pointing_source_strategy, pointing_fit_gaussian_enabled,
        pointing_fruitloops_center_mode,
        pointing_header_center_max_radius_arcsec,
        pointing_header_center_require_coverage, ptcproc, logger);
}
