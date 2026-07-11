#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_fruit_loops_core_config(
    Config &config,
    citlali::config::TimestreamFruitLoopsConfig &typed_config,
    Diagnostics &diagnostics) {
    bool enabled = typed_config.enabled;
    read_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "enabled"},
        enabled, typed_config.enabled, diagnostics);
    if (!typed_config.enabled) {
        return;
    }

    bool save_all_iters = typed_config.save_all_iters;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "save_all_iters"},
        save_all_iters, typed_config.save_all_iters, diagnostics);

    std::string path = typed_config.path;
    read_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "path"},
        path, typed_config.path, diagnostics);

    std::string type = typed_config.type;
    read_config_value_if_clean(
        config, std::tuple{"timestream", "fruit_loops", "type"}, type,
        [&typed_config](const std::string &value) {
            typed_config.type = std::string{
                citlali::config::canonical_fruit_loops_type(value)};
        },
        diagnostics);

    std::string mode{citlali::config::to_string(typed_config.mode)};
    read_parsed_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "mode"}, mode,
        typed_config.mode, citlali::config::parse_fruit_loops_mode,
        diagnostics, {"upper", "lower", "both"});

    double sig2noise_limit = typed_config.sig2noise_limit;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "sig2noise_limit"},
        sig2noise_limit, typed_config.sig2noise_limit, diagnostics);

    auto array_flux_limit = typed_config.array_flux_limit;
    read_mirrored_config_value(
        config,
        std::tuple{"timestream", "fruit_loops", "array_flux_limit"},
        array_flux_limit, typed_config.array_flux_limit, diagnostics);

    int max_iters = typed_config.max_iters;
    read_mirrored_config_value(
        config, std::tuple{"timestream", "fruit_loops", "max_iters"},
        max_iters, typed_config.max_iters, diagnostics);
}

}  // namespace citlali::pipeline
