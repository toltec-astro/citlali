#pragma once

namespace citlali::pipeline {

template <class Engine>
bool fruit_loop_iteration_pending(const Engine &engine,
                                  bool fruit_loops_converged) {
    return (engine.fruit_iter < engine.ptcproc.fruit_loops_iters) &&
           !fruit_loops_converged;
}

template <class Engine, class Logger>
void begin_fruit_loop_iteration(Engine &engine, const Logger &logger) {
    if (engine.ptcproc.run_fruit_loops) {
        logger->info("starting fruit loops iteration {}", engine.fruit_iter);
    }
    engine.ptcproc.begin_weight_validation_iteration(engine.fruit_iter);

    const bool learning_source_model_available =
        engine.ptcproc.run_fruit_loops &&
        (engine.fruit_iter > 0 || engine.ptcproc.fruit_loops_path != "null");
    engine.reduction_learning.begin_iteration(
        engine.fruit_iter, learning_source_model_available, engine.redu_type);
    if (engine.reduction_learning.is_enabled() &&
        engine.reduction_learning.diagnostics_enabled()) {
        logger->info("reduction learning begin: {}",
                     engine.reduction_learning.summary_string());
    }
}

template <class Engine, class Logger>
void finalize_fruit_loop_iteration(Engine &engine, const Logger &logger) {
    engine.ptcproc.finalize_weight_validation_iteration(engine.fruit_iter);
    engine.reduction_learning.finalize_iteration(engine.fruit_iter);
    if (engine.reduction_learning.is_enabled() &&
        engine.reduction_learning.diagnostics_enabled()) {
        logger->info("reduction learning finalize: {}",
                     engine.reduction_learning.summary_string());
    }
    engine.write_learning_summary();
}

template <class TodProc, class Logger>
void finalize_iteration_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    finalize_fruit_loop_iteration(engine, logger);

    logger->info("making index files");
    todproc.make_index_file(engine.redu_dir_name);

    engine.fruit_iter++;
}

}  // namespace citlali::pipeline
