#pragma once

#include <citlali/core/pipeline/initial_fruit_loop_map_loading.h>
#include <citlali/core/pipeline/previous_fruit_loop_map_loading.h>

namespace citlali::pipeline {

template <bool IsBeammap, class Engine, class Logger>
void load_observation_fruit_loop_maps_if_needed(Engine &engine,
                                                const Logger &logger) {
    if constexpr (!IsBeammap) {
        load_initial_fruit_loop_maps_if_requested(engine);
        load_previous_fruit_loop_maps_if_needed(engine, logger);
    }
}

}  // namespace citlali::pipeline
