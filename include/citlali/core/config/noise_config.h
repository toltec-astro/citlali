#pragma once

#include <citlali/core/config/config_error.h>

namespace citlali::config {

struct NoiseConfig {
    bool enabled = false;
    int n_noise_maps = 1;
    bool randomize_dets = true;
    bool write_realizations = false;
    bool products_enabled = true;
    bool apply_empirical_weights = true;
};

inline void validate(const NoiseConfig &config, ValidationReport &report) {
    check_minimum(config.n_noise_maps, 1, {"noise_maps", "n_noise_maps"}, report);
}

}  // namespace citlali::config
