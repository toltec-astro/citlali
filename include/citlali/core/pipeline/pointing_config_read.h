#pragma once

#include <citlali/core/config/pointing_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_pointing_source_strategy_config(
    Config &config, double default_header_max_radius_arcsec,
    citlali::config::PointingConfig &pointing_config,
    Diagnostics &diagnostics) {
    std::string pointing_source_strategy = "standard";
    read_optional_parsed_mirrored_config_value(
        config, std::tuple{"pointing", "source_strategy", "mode"},
        pointing_source_strategy, pointing_config.source_strategy,
        citlali::config::parse_pointing_source_strategy, diagnostics,
        {"standard", "psf_preserve"});

    bool pointing_fit_gaussian_enabled =
        citlali::config::is_standard_pointing_source_strategy(
            pointing_config.source_strategy);
    pointing_config.fit_gaussian = pointing_fit_gaussian_enabled;
    read_optional_mirrored_config_value(
        config, std::tuple{"pointing", "source_strategy", "fit_gaussian"},
        pointing_fit_gaussian_enabled, pointing_config.fit_gaussian,
        diagnostics);

    std::string pointing_fruitloops_center_mode =
        citlali::config::is_psf_preserve_pointing_source_strategy(
            pointing_config.source_strategy)
            ? "map_center"
            : "auto";
    if (auto parsed = citlali::config::parse_fruit_loops_center_mode(
            pointing_fruitloops_center_mode)) {
        pointing_config.fruitloops_center_mode = *parsed;
    }
    read_optional_parsed_mirrored_config_value(
        config,
        std::tuple{"pointing", "source_strategy", "fruitloops_center_mode"},
        pointing_fruitloops_center_mode,
        pointing_config.fruitloops_center_mode,
        citlali::config::parse_fruit_loops_center_mode, diagnostics,
        {"auto", "header", "peak", "map_center"});

    double pointing_header_center_max_radius_arcsec =
        citlali::config::is_standard_pointing_source_strategy(
            pointing_config.source_strategy)
            ? default_header_max_radius_arcsec
            : 0.0;
    pointing_config.header_max_radius_arcsec =
        pointing_header_center_max_radius_arcsec;
    read_optional_mirrored_config_value(
        config,
        std::tuple{"pointing", "source_strategy",
                   "header_max_radius_arcsec"},
        pointing_header_center_max_radius_arcsec,
        pointing_config.header_max_radius_arcsec, diagnostics, {}, {0.0});

    bool pointing_header_center_require_coverage = true;
    pointing_config.header_require_coverage =
        pointing_header_center_require_coverage;
    read_optional_mirrored_config_value(
        config,
        std::tuple{"pointing", "source_strategy",
                   "header_require_coverage"},
        pointing_header_center_require_coverage,
        pointing_config.header_require_coverage, diagnostics);
}

}  // namespace citlali::pipeline
