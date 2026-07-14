#pragma once

namespace citlali::config {

struct NoiseConfig {
    bool enabled = false;
    int n_noise_maps = 1;
    bool randomize_dets = true;
    bool write_realizations = false;
    bool products_enabled = true;
    bool apply_empirical_weights = true;
};

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

}  // namespace citlali::config
