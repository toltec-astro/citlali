#pragma once

#include <citlali/core/pipeline/beammap_provenance_lifecycle.h>
#include <citlali/core/pipeline/fruit_loop_iteration_state.h>
#include <citlali/core/pipeline/fruit_loop_iteration_policy.h>
#include <citlali/core/pipeline/fruit_loop_restart_lifecycle.h>
#include <citlali/core/pipeline/iteration_buffers.h>
#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/iteration_output_layout.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/pointing_provenance_lifecycle.h>
#include <citlali/core/pipeline/post_processing_provenance_lifecycle.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class TodProc, class ConfigFilepaths, class Logger>
void begin_reduction_iteration(TodProc &todproc,
                               const ConfigFilepaths &config_filepaths,
                               StageProfileCollector &stage_profile,
                               const Logger &logger) {
    auto &engine = todproc.engine();

    begin_fruit_loop_iteration(engine, logger);
    begin_mapmaking_iteration_if_available(engine);
    begin_pointing_iteration_if_available(engine);
    begin_post_processing_iteration_if_available(engine);
    begin_beammap_run_if_available(engine);
    prepare_iteration_output_layout_if_needed(todproc, config_filepaths,
                                              stage_profile, logger);
    prepare_iteration_observation_buffers(todproc, logger);
}

template <class Engine, class Logger>
void initialize_reduction_iterations(Engine &engine,
                                     ReductionIterationState &state,
                                     const Logger &logger) {
    engine.iteration.fruit_iter = 0;
    reset_reduction_iteration_state(state);
    configure_fruit_loop_iteration_policy(engine, logger);
    initialize_fruit_loop_restart_if_requested(engine, state, logger);
}

template <class Engine, class Logger>
void initialize_reduction_iterations(Engine &engine,
                                     bool &fruit_loops_converged,
                                     const Logger &logger) {
    engine.iteration.fruit_iter = 0;
    fruit_loops_converged = false;
    configure_fruit_loop_iteration_policy(engine, logger);
}

}  // namespace citlali::pipeline
