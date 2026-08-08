#pragma once

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <cmath>
#include <tuple>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
citlali::config::CalibrationConfig read_calibration_config(
    Config &config, Diagnostics &diagnostics) {
    citlali::config::CalibrationConfig result;
    const auto key =
        std::tuple{"calibration", "reference_spectral_index_alpha"};
    const auto node = config.get_node(key);
    if (!node.IsDefined()) {
        return result;
    }
    if (!config.template has_typed<double>(key)) {
        add_invalid_config_key(key, diagnostics.invalid_key_paths());
        return result;
    }
    double alpha = 0.0;
    read_config_value_if_clean(
        config, key, alpha,
        [&result](double value) {
            result.reference.spectral_index_alpha = value;
        },
        diagnostics, {-1.0, 0.0, 2.0, 4.0});
    return result;
}

}  // namespace citlali::pipeline
