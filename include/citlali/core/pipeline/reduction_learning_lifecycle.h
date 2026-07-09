#pragma once

#include <citlali/core/config/config_value.h>

namespace citlali::pipeline {

template <class Engine>
bool fruit_loop_learning_source_model_available(const Engine &engine) {
    return engine.ptcproc.run_fruit_loops &&
           (engine.iteration.fruit_iter > 0 ||
            citlali::config::has_config_value(
                engine.ptcproc.fruit_loops_path));
}

template <class Engine>
bool should_log_reduction_learning_diagnostics(const Engine &engine) {
    return engine.reduction_learning.is_enabled() &&
           engine.reduction_learning.diagnostics_enabled();
}

template <class Engine, class Logger>
void log_reduction_learning_iteration_if_needed(Engine &engine,
                                                const Logger &logger,
                                                const char *phase) {
    if (should_log_reduction_learning_diagnostics(engine)) {
        logger->info("reduction learning {}: {}",
                     phase, engine.reduction_learning.summary_string());
    }
}

template <class Engine>
void begin_reduction_learning_iteration(Engine &engine) {
    const bool learning_source_model_available =
        fruit_loop_learning_source_model_available(engine);
    engine.reduction_learning.begin_iteration(
        engine.iteration.fruit_iter, learning_source_model_available,
        engine.typed_config.runtime.reduction_type);
}

template <class Engine>
void finalize_reduction_learning_iteration(Engine &engine) {
    engine.reduction_learning.finalize_iteration(engine.iteration.fruit_iter);
}

}  // namespace citlali::pipeline
