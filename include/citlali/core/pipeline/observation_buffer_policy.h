#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/output_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool should_allocate_observation_map_buffers(const Engine &engine) {
    return mapmaking_outputs_enabled(engine);
}

template <class Engine>
void configure_observation_pixel_contribution_targets(Engine &engine) {
    engine.configure_map_pixel_contribution_targets(engine.omb, "raw_obs");
}

template <class Engine>
bool should_allocate_observation_noise_maps(const Engine &engine) {
    // Noise realizations are observation-owned until the whole normalized map
    // bundle passes SCI-MAP-001 coadd admission. This keeps signal, kernel, and
    // realizations on one accepted map-operator boundary.
    return noise_maps_enabled(engine);
}

}  // namespace citlali::pipeline
