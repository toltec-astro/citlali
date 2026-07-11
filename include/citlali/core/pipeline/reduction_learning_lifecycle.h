#pragma once

#include <citlali/core/config/config_value.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::pipeline {

template <class Engine>
bool fruit_loop_learning_source_model_available(const Engine &engine) {
    return fruit_loops_config(engine).enabled &&
           (engine.iteration.fruit_iter > 0 ||
            citlali::config::has_config_value(
                fruit_loops_config(engine).path));
}

template <class Engine>
bool should_log_reduction_learning_diagnostics(const Engine &engine) {
    return engine.learning.is_enabled() &&
           engine.learning.diagnostics_enabled();
}

template <class Engine, class Logger>
void log_reduction_learning_iteration_if_needed(Engine &engine,
                                                const Logger &logger,
                                                const char *phase) {
    if (should_log_reduction_learning_diagnostics(engine)) {
        logger->info("reduction learning {}: {}",
                     phase, engine.learning.summary_string());
    }
}

template <class Engine>
void begin_reduction_learning_iteration(Engine &engine) {
    const bool learning_source_model_available =
        fruit_loop_learning_source_model_available(engine);
    engine.learning.begin_iteration(
        engine.iteration.fruit_iter, learning_source_model_available,
        runtime_reduction_type(engine));
}

template <class Engine>
void finalize_reduction_learning_iteration(Engine &engine) {
    engine.learning.finalize_iteration(engine.iteration.fruit_iter);
}

}  // namespace citlali::pipeline
