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
    return noise_maps_enabled(engine) &&
           (!coadd_outputs_enabled(engine) ||
            mapmaking_config(engine).method ==
                citlali::config::MapMethod::jinc);
}

}  // namespace citlali::pipeline
