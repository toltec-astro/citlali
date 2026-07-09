#pragma once

#include <citlali/core/config/config_value.h>
#include <citlali/core/pipeline/fruit_loop_map_io.h>
#include <citlali/core/pipeline/fruit_loop_paths.h>

#include <string>

namespace citlali::pipeline {

template <class Engine>
bool should_load_initial_fruit_loop_maps(const Engine &engine) {
    return engine.ptcproc.run_fruit_loops &&
           engine.iteration.fruit_iter == 0 &&
           citlali::config::has_config_value(
               engine.ptcproc.fruit_loops_path);
}

template <class Engine>
std::string initial_fruit_loop_map_dir(const Engine &engine) {
    return fruit_loop_map_dir(engine.ptcproc.fruit_loops_path,
                              engine.ptcproc.fruit_loops_type,
                              engine.omb.obsnums.back());
}

template <class Engine>
void load_initial_fruit_loop_maps_if_requested(Engine &engine) {
    if (should_load_initial_fruit_loop_maps(engine)) {
        const auto fruit_dir = initial_fruit_loop_map_dir(engine);
        load_fruit_loop_maps(engine, fruit_dir);
    }
}

}  // namespace citlali::pipeline
