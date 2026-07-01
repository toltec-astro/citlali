#pragma once

#include <citlali/core/pipeline/fruit_loop_map_io.h>
#include <citlali/core/pipeline/fruit_loop_paths.h>

#include <string>

namespace citlali::pipeline {

template <class Engine>
bool should_load_initial_fruit_loop_maps(const Engine &engine) {
    return engine.ptcproc.run_fruit_loops &&
           engine.fruit_iter == 0 &&
           engine.ptcproc.fruit_loops_path != "null";
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

template <class Engine>
bool should_load_previous_fruit_loop_maps(const Engine &engine) {
    return engine.fruit_iter > 0;
}

template <class Engine>
std::string saved_previous_fruit_loop_map_dir(const Engine &engine) {
    return previous_fruit_loop_map_dir(
        engine.output_dir, engine.redu_dir_num,
        engine.ptcproc.fruit_loops_type,
        engine.omb.obsnums.back());
}

template <class Engine>
std::string current_previous_fruit_loop_map_dir(const Engine &engine) {
    return fruit_loop_map_dir(engine.redu_dir_name,
                              engine.ptcproc.fruit_loops_type,
                              engine.omb.obsnums.back());
}

template <class Engine, class Logger>
std::string previous_fruit_loop_map_dir(const Engine &engine,
                                        const Logger &logger) {
    if (engine.ptcproc.save_all_iters) {
        return saved_previous_fruit_loop_map_dir(engine);
    }

    logger->info("loading previous iter maps for fruit loops iteration {}",
                 engine.fruit_iter);
    return current_previous_fruit_loop_map_dir(engine);
}

template <class Engine, class Logger>
void load_previous_fruit_loop_maps(Engine &engine,
                                   const std::string &fruit_dir,
                                   const Logger &logger) {
    logger->info("reading in {} for fruit loops iteration {}", fruit_dir,
                 engine.fruit_iter);
    load_fruit_loop_maps(engine, fruit_dir);
}

template <class Engine, class Logger>
void load_previous_fruit_loop_maps_if_needed(Engine &engine,
                                             const Logger &logger) {
    if (should_load_previous_fruit_loop_maps(engine)) {
        const auto fruit_dir = previous_fruit_loop_map_dir(engine, logger);
        load_previous_fruit_loop_maps(engine, fruit_dir, logger);
    }
}

template <bool IsBeammap, class Engine, class Logger>
void load_observation_fruit_loop_maps_if_needed(Engine &engine,
                                                const Logger &logger) {
    if constexpr (!IsBeammap) {
        load_initial_fruit_loop_maps_if_requested(engine);
        load_previous_fruit_loop_maps_if_needed(engine, logger);
    }
}

}  // namespace citlali::pipeline
