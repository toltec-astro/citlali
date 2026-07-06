#pragma once

// Engine config loading implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/engine/detail/pointing_config_logging.h>

template<typename CT>
void Engine::get_pointing_config(CT &config) {
    logger->info("getting pointing config options");
    typed_config.pointing = citlali::config::PointingConfig{};

    auto parsed_cleanly = [&](std::size_t missing_before, std::size_t invalid_before) {
        return citlali::engine_detail::config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before);
    };

    pointing_source_strategy = "standard";
    if (config.template has_typed<std::string>(std::tuple{"pointing","source_strategy","mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_source_strategy, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","mode"},
                         {"standard", "psf_preserve"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_pointing_source_strategy(
                    pointing_source_strategy)) {
                typed_config.pointing.source_strategy = *parsed;
            }
        }
    }

    pointing_fit_gaussian_enabled = (pointing_source_strategy == "standard");
    typed_config.pointing.fit_gaussian = pointing_fit_gaussian_enabled;
    if (config.template has_typed<bool>(std::tuple{"pointing","source_strategy","fit_gaussian"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_fit_gaussian_enabled, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","fit_gaussian"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_config.pointing.fit_gaussian = pointing_fit_gaussian_enabled;
        }
    }

    pointing_fruitloops_center_mode =
        (pointing_source_strategy == "psf_preserve") ? "map_center" : "auto";
    if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
            pointing_fruitloops_center_mode)) {
        typed_config.pointing.fruitloops_center_mode = *parsed;
    }
    if (config.template has_typed<std::string>(std::tuple{"pointing","source_strategy","fruitloops_center_mode"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_fruitloops_center_mode, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","fruitloops_center_mode"},
                         {"auto", "header", "peak", "map_center"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
                    pointing_fruitloops_center_mode)) {
                typed_config.pointing.fruitloops_center_mode = *parsed;
            }
        }
    }

    pointing_header_center_max_radius_arcsec = 0.0;
    if (pointing_source_strategy == "standard" &&
        std::isfinite(map_fitter.fitting_region_pix) && map_fitter.fitting_region_pix > 0.0 &&
        std::isfinite(omb.pixel_size_rad) && omb.pixel_size_rad > 0.0) {
        pointing_header_center_max_radius_arcsec =
            map_fitter.fitting_region_pix * omb.pixel_size_rad * RAD_TO_ASEC;
    }
    typed_config.pointing.header_max_radius_arcsec =
        pointing_header_center_max_radius_arcsec;
    if (config.template has_typed<double>(std::tuple{"pointing","source_strategy","header_max_radius_arcsec"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_header_center_max_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","header_max_radius_arcsec"},
                         {}, {0.0});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_config.pointing.header_max_radius_arcsec =
                pointing_header_center_max_radius_arcsec;
        }
    }

    pointing_header_center_require_coverage = true;
    typed_config.pointing.header_require_coverage =
        pointing_header_center_require_coverage;
    if (config.template has_typed<bool>(std::tuple{"pointing","source_strategy","header_require_coverage"})) {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        get_config_value(config, pointing_header_center_require_coverage, missing_keys, invalid_keys,
                         std::tuple{"pointing","source_strategy","header_require_coverage"});
        if (parsed_cleanly(missing_before, invalid_before)) {
            typed_config.pointing.header_require_coverage =
                pointing_header_center_require_coverage;
        }
    }

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
