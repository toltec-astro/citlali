#pragma once

#include <citlali/core/pipeline/reduction_learning_lifecycle.h>
#include <citlali/core/pipeline/weight_validation_lifecycle.h>

namespace citlali::pipeline {

template <class Engine>
bool should_log_fruit_loop_iteration_start(const Engine &engine) {
    return engine.ptcproc.run_fruit_loops;
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
    todproc.make_index_file(engine.output_paths.redu_dir_name);
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
