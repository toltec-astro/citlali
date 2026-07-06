#pragma once

#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool unfiltered_noise_products_apply_empirical_weights(
    const Engine &engine) {
    return empirical_noise_weights_enabled(engine);
}

template <class Engine>
bool filtered_noise_products_apply_empirical_weights(const Engine &engine) {
    return empirical_noise_weights_enabled(engine) ||
           engine.wiener_filter.normalize_error;
}

}  // namespace citlali::pipeline
