#pragma once

#include <citlali/core/config/mapmaking_config.h>

namespace citlali::pipeline {

template <class Engine>
bool should_allocate_observation_map_buffers(const Engine &engine) {
    return engine.run_mapmaking;
}

template <class Engine>
void configure_observation_pixel_contribution_targets(Engine &engine) {
    engine.configure_map_pixel_contribution_targets(engine.omb, "raw_obs");
}

template <class Engine>
bool should_allocate_observation_noise_maps(const Engine &engine) {
    return engine.run_noise &&
           (!engine.run_coadd ||
            engine.typed_config.mapmaking.method ==
                citlali::config::MapMethod::jinc);
}

}  // namespace citlali::pipeline
