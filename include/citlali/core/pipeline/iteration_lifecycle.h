#pragma once

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
