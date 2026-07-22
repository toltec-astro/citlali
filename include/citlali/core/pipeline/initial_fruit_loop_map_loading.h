#pragma once

#include <citlali/core/config/config_value.h>
#include <citlali/core/pipeline/fruit_loop_map_io.h>
#include <citlali/core/pipeline/fruit_loop_paths.h>
#include <citlali/core/pipeline/fruit_loop_restart_lifecycle.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <string>

namespace citlali::pipeline {

template <class Engine>
bool should_load_initial_fruit_loop_maps(const Engine &engine) {
    if (first_restarted_iteration(engine)) {
        return true;
    }
    return fruit_loops_config(engine).enabled &&
           engine.iteration.fruit_iter == 0 &&
           citlali::config::has_config_value(
               fruit_loops_config(engine).path);
}

template <class Engine>
std::string initial_fruit_loop_map_dir(const Engine &engine) {
    const auto *restart = fruit_loop_restart_resolution(engine);
    const auto &base_dir = first_restarted_iteration(engine)
        ? restart->source_reduction_dir
        : fruit_loops_config(engine).path;
    return fruit_loop_map_dir(base_dir,
                              fruit_loops_config(engine).type,
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
