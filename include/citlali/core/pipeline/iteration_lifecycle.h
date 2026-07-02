#pragma once

#include <citlali/core/pipeline/reduction_learning_lifecycle.h>

namespace citlali::pipeline {

struct ReductionIterationState {
    bool fruit_loops_converged = false;
};

inline void reset_reduction_iteration_state(ReductionIterationState &state) {
    state.fruit_loops_converged = false;
}

template <class Engine>
bool fruit_loop_iteration_pending(const Engine &engine,
                                  bool fruit_loops_converged) {
    return (engine.fruit_iter < engine.ptcproc.fruit_loops_iters) &&
           !fruit_loops_converged;
}

template <class Engine>
bool fruit_loop_iteration_pending(const Engine &engine,
                                  const ReductionIterationState &state) {
    return fruit_loop_iteration_pending(engine,
                                        state.fruit_loops_converged);
}

template <class Engine>
bool should_log_fruit_loop_iteration_start(const Engine &engine) {
    return engine.ptcproc.run_fruit_loops;
}

template <class Engine>
void begin_iteration_weight_validation(Engine &engine) {
    engine.ptcproc.begin_weight_validation_iteration(engine.fruit_iter);
}

template <class Engine>
void finalize_iteration_weight_validation(Engine &engine) {
    engine.ptcproc.finalize_weight_validation_iteration(engine.fruit_iter);
}

template <class Engine, class Logger>
void begin_fruit_loop_iteration(Engine &engine, const Logger &logger) {
    if (should_log_fruit_loop_iteration_start(engine)) {
        logger->info("starting fruit loops iteration {}", engine.fruit_iter);
    }
    begin_iteration_weight_validation(engine);

    begin_reduction_learning_iteration(engine);
    log_reduction_learning_iteration_if_needed(engine, logger, "begin");
}

template <class Engine, class Logger>
void finalize_fruit_loop_iteration(Engine &engine, const Logger &logger) {
    finalize_iteration_weight_validation(engine);
    finalize_reduction_learning_iteration(engine);
    log_reduction_learning_iteration_if_needed(engine, logger, "finalize");
    engine.write_learning_summary();
}

template <class TodProc, class Logger>
void make_reduction_iteration_index_file(TodProc &todproc,
                                         const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("making index files");
    todproc.make_index_file(engine.redu_dir_name);
}

template <class Engine>
void advance_fruit_loop_iteration(Engine &engine) {
    engine.fruit_iter++;
}

template <class TodProc, class Logger>
void finalize_iteration_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    finalize_fruit_loop_iteration(engine, logger);

    make_reduction_iteration_index_file(todproc, logger);
    advance_fruit_loop_iteration(engine);
}

}  // namespace citlali::pipeline
