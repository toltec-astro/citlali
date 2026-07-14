#pragma once

#include <citlali/core/config/pointing_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>
#include <citlali/core/pipeline/pointing_execution_plan.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Key, class Enum, class Parser,
          class Diagnostics>
bool read_optional_pointing_enum(
    Config &config, const Key &key, Enum &target, Parser parser,
    Diagnostics &diagnostics, std::vector<std::string> accepted_values) {
    if (!config.template has_typed<std::string>(key)) {
        return false;
    }
    std::string value{citlali::config::to_string(target)};
    read_config_value_if_clean(
        config, key, value,
        [&target, &parser, &key, &diagnostics](const auto &parsed_value) {
            if (auto parsed = parser(parsed_value)) {
                target = *parsed;
            } else {
                add_invalid_config_key(
                    key, diagnostics.invalid_key_paths());
            }
        },
        diagnostics, std::move(accepted_values));
    return true;
}

template <class Config, class Diagnostics>
PointingRequestPresence read_pointing_request_config(
    Config &config, citlali::config::PointingConfig &request,
    Diagnostics &diagnostics) {
    PointingRequestPresence presence;

    const auto strategy_key =
        std::tuple{"pointing", "source_strategy", "mode"};
    presence.source_strategy = read_optional_pointing_enum(
        config, strategy_key, request.source_strategy,
        citlali::config::parse_pointing_source_strategy, diagnostics,
        {"standard", "psf_preserve"});

    request.fit_gaussian =
        citlali::config::is_standard_pointing_source_strategy(
            request.source_strategy);
    const auto fit_key =
        std::tuple{"pointing", "source_strategy", "fit_gaussian"};
    presence.fit_gaussian = config.template has_typed<bool>(fit_key);
    read_optional_config_value(
        config, request.fit_gaussian, diagnostics, fit_key);

    request.fruitloops_center_mode =
        citlali::config::is_psf_preserve_pointing_source_strategy(
            request.source_strategy)
            ? citlali::config::FruitLoopsCenterMode::map_center
            : citlali::config::FruitLoopsCenterMode::automatic;
    const auto center_key = std::tuple{
        "pointing", "source_strategy", "fruitloops_center_mode"};
    presence.fruitloops_center_mode = read_optional_pointing_enum(
        config, center_key, request.fruitloops_center_mode,
        citlali::config::parse_fruit_loops_center_mode, diagnostics,
        {"auto", "header", "peak", "map_center"});

    request.header_max_radius_arcsec = 0.0;
    const auto radius_key = std::tuple{
        "pointing", "source_strategy", "header_max_radius_arcsec"};
    presence.header_max_radius_arcsec =
        config.template has_typed<double>(radius_key);
    read_optional_config_value(
        config, request.header_max_radius_arcsec, diagnostics,
        radius_key, {}, {0.0});

    request.header_require_coverage = true;
    const auto coverage_key = std::tuple{
        "pointing", "source_strategy", "header_require_coverage"};
    presence.header_require_coverage =
        config.template has_typed<bool>(coverage_key);
    read_optional_config_value(
        config, request.header_require_coverage, diagnostics,
        coverage_key);

    return presence;
}

}  // namespace citlali::pipeline
