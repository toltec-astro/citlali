#pragma once

namespace citlali::pipeline {

template <class Engine>
bool unfiltered_noise_products_apply_empirical_weights(
    const Engine &engine) {
    return engine.apply_empirical_noise_weights;
}

template <class Engine>
bool filtered_noise_products_apply_empirical_weights(const Engine &engine) {
    return engine.apply_empirical_noise_weights ||
           engine.wiener_filter.normalize_error;
}

}  // namespace citlali::pipeline
