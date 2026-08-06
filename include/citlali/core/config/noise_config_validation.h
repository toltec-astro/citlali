#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/noise_config.h>

namespace citlali::config {

inline void validate(const NoiseConfig &config, ValidationReport &report) {
    check_minimum(
        config.n_noise_maps, config.enabled ? 1 : 0,
        {"noise_maps", "n_noise_maps"}, report);
}

}  // namespace citlali::config
