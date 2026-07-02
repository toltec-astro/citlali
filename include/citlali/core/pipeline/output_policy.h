#pragma once

namespace citlali::pipeline {

template <class Engine>
bool should_write_filtered_outputs(const Engine &engine) {
    return engine.run_map_filter;
}

template <class Engine>
bool filtered_maps_written_during_filtering(const Engine &engine) {
    return engine.write_filtered_maps_partial;
}

template <class Engine>
bool should_calculate_filtered_noise_products(const Engine &engine) {
    return engine.run_noise_products &&
           engine.run_noise &&
           !filtered_maps_written_during_filtering(engine);
}

template <class Engine>
bool should_find_filtered_sources(const Engine &engine) {
    return engine.run_source_finder;
}

template <class Engine>
bool should_write_iteration_coadd_outputs(const Engine &engine) {
    return engine.run_coadd;
}

}  // namespace citlali::pipeline
