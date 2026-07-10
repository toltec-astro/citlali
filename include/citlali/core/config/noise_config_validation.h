#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/noise_config.h>

namespace citlali::config {

inline void validate(const NoiseConfig &config, ValidationReport &report) {
    if (config.enabled) {
        check_minimum(config.n_noise_maps, 0, {"noise_maps", "n_noise_maps"},
                      report);
    }
}

}  // namespace citlali::config
