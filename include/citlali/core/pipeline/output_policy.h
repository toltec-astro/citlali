#pragma once

namespace citlali::pipeline {

template <class Engine>
bool mapmaking_enabled(const Engine &engine) {
    return engine.typed_config.mapmaking.enabled;
}

template <class Engine>
bool coadd_enabled(const Engine &engine) {
    return engine.typed_config.coadd.enabled;
}

template <class Engine>
bool noise_maps_enabled(const Engine &engine) {
    return engine.typed_config.noise.enabled;
}

template <class Engine>
bool noise_realization_outputs_enabled(const Engine &engine) {
    return engine.typed_config.noise.write_realizations;
}

template <class Engine>
bool noise_product_outputs_enabled(const Engine &engine) {
    return engine.typed_config.noise.products_enabled;
}

template <class Engine>
bool empirical_noise_weights_enabled(const Engine &engine) {
    return engine.typed_config.noise.apply_empirical_weights;
}

template <class Engine>
bool empirical_weight_calibration_enabled(const Engine &engine) {
    return noise_product_outputs_enabled(engine) &&
           noise_maps_enabled(engine) &&
           empirical_noise_weights_enabled(engine);
}

template <class Engine>
bool map_filter_enabled(const Engine &engine) {
    return engine.typed_config.post_processing.map_filtering.enabled;
}

template <class Engine>
bool source_finding_enabled(const Engine &engine) {
    return engine.typed_config.post_processing.source_finding.enabled;
}

template <class Engine>
bool mapmaking_outputs_enabled(const Engine &engine) {
    return mapmaking_enabled(engine);
}

template <class Engine>
bool coadd_outputs_enabled(const Engine &engine) {
    return coadd_enabled(engine);
}

template <class Engine>
bool map_filter_outputs_enabled(const Engine &engine) {
    return map_filter_enabled(engine);
}

template <class Engine>
bool source_finding_outputs_enabled(const Engine &engine) {
    return source_finding_enabled(engine);
}

template <class Engine>
bool should_write_filtered_outputs(const Engine &engine) {
    return map_filter_outputs_enabled(engine);
}

template <class Engine>
bool filtered_maps_written_during_filtering(const Engine &engine) {
    return engine.write_filtered_maps_partial;
}

template <class Engine>
bool should_calculate_filtered_noise_products(const Engine &engine) {
    return noise_product_outputs_enabled(engine) &&
           noise_maps_enabled(engine) &&
           !filtered_maps_written_during_filtering(engine);
}

template <class Engine>
bool should_find_filtered_sources(const Engine &engine) {
    return source_finding_outputs_enabled(engine);
}

template <class Engine>
bool should_write_iteration_coadd_outputs(const Engine &engine) {
    return coadd_outputs_enabled(engine);
}

}  // namespace citlali::pipeline
