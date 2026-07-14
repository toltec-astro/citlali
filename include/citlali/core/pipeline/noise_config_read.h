#pragma once

#include <citlali/core/config/noise_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <tuple>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_noise_request_config(
    Config &config, citlali::config::NoiseConfig &request,
    Diagnostics &diagnostics) {
    read_config_value(
        config, request.enabled, diagnostics,
        std::tuple{"noise_maps", "enabled"});
    read_config_value(
        config, request.n_noise_maps, diagnostics,
        std::tuple{"noise_maps", "n_noise_maps"}, {}, {0}, {});
    read_config_value(
        config, request.randomize_dets, diagnostics,
        std::tuple{"noise_maps", "randomize_dets"});

    request.write_realizations = false;
    read_optional_config_value(
        config, request.write_realizations, diagnostics,
        std::tuple{"noise_maps", "write_realizations"});

    request.products_enabled = request.enabled;
    read_optional_config_value(
        config, request.products_enabled, diagnostics,
        std::tuple{"noise_maps", "products", "enabled"});

    request.apply_empirical_weights = request.enabled;
    read_optional_config_value(
        config, request.apply_empirical_weights, diagnostics,
        std::tuple{
            "noise_maps", "products", "apply_empirical_weights"});
}

}  // namespace citlali::pipeline
