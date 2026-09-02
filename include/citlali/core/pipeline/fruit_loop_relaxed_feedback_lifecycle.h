#pragma once

#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/timestream/fruit_loop_relaxed_feedback_state.h>

namespace citlali::pipeline {

template <class Engine, class Logger>
void update_fruit_loop_relaxed_feedback_state_if_needed(
    Engine &engine, const Logger &logger) {
    const auto &config = fruit_loops_config(engine);
    if (!config.enabled || !config.relaxation_experiment_enabled) {
        return;
    }
    if constexpr (requires {
                      engine.ptcproc.fruit_loop_relaxed_feedback_state;
                      engine.omb;
                      engine.observation_identity.obsnum;
                      engine.iteration.fruit_iter;
                  }) {
        citlali::fruit::update_fruit_loop_relaxed_feedback_state(
            engine.ptcproc.fruit_loop_relaxed_feedback_state, engine.omb,
            engine.observation_identity.obsnum,
            engine.iteration.fruit_iter, config.relaxation_alpha);
        const auto &state =
            engine.ptcproc.fruit_loop_relaxed_feedback_state;
        logger->info(
            "EL-F1 feedback state: method={} alpha={:.2f} observation={} completed_iteration={} maps={} rows={} cols={}",
            state.method_id, state.alpha, state.observation_id,
            state.completed_iteration, state.map_count, state.n_rows,
            state.n_cols);
    }
    else {
        throw citlali::error::runtime(
            "EL-F1 relaxed feedback state is unavailable for this engine");
    }
}

}  // namespace citlali::pipeline
