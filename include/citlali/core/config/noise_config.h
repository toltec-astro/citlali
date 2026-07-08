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

inline void set_noise_maps_enabled(NoiseConfig &config, bool enabled) {
    config.enabled = enabled;
}

inline bool noise_maps_active(const NoiseConfig &config) {
    return config.enabled;
}

inline bool noise_realization_outputs_active(const NoiseConfig &config) {
    return config.write_realizations;
}

inline bool noise_product_outputs_active(const NoiseConfig &config) {
    return config.products_enabled;
}

inline bool empirical_noise_weights_active(const NoiseConfig &config) {
    return config.apply_empirical_weights;
}

inline void validate(const NoiseConfig &config, ValidationReport &report) {
    if (config.enabled) {
        check_minimum(config.n_noise_maps, 0, {"noise_maps", "n_noise_maps"}, report);
    }
}

}  // namespace citlali::config
