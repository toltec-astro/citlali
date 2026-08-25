#pragma once

#include <citlali/core/pipeline/native_consumer_mode_policy.h>
#include <citlali/core/pipeline/reduction_learning_lifecycle.h>
#include <citlali/core/pipeline/fruit_loop_restart_lifecycle.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/weight_validation_lifecycle.h>
#include <citlali/core/error/error.h>

namespace citlali::pipeline {

template <class Engine>
void reset_fruit_loop_feedback_samples_if_available(Engine &engine) {
    if constexpr (requires {
                      engine.ptcproc.reset_fruit_loop_feedback_samples();
                  }) {
        engine.ptcproc.reset_fruit_loop_feedback_samples();
    }
    if constexpr (requires {
                      engine.ptcproc
                          .reset_fruit_loop_injected_source_samples();
                  }) {
        engine.ptcproc.reset_fruit_loop_injected_source_samples();
    }
}

template <class Engine, class Logger>
void require_realized_fruit_loop_feedback_if_available(
    const Engine &engine, const Logger &logger) {
    if constexpr (requires {
                      engine.ptcproc.current_fruit_loop_feedback_samples();
                  }) {
        const bool model_available =
            fruit_loop_learning_source_model_available(engine);
        const auto feedback_samples =
            engine.ptcproc.current_fruit_loop_feedback_samples();
        if (model_available) {
            logger->info(
                "fruit-loop realized feedback: iteration={} detector_samples={}",
                engine.iteration.fruit_iter, feedback_samples);
        }
        if (model_available &&
            runtime_reduction_type(engine) !=
                citlali::config::ReductionType::beammap &&
            feedback_samples == 0) {
            throw citlali::error::runtime(
                "fruit-loop source model selected zero detector-samples; refusing to continue no-op feedback iterations");
        }
    }
    if constexpr (requires {
                      engine.ptcproc
                          .current_fruit_loop_injected_source_samples();
                  }) {
        const auto &injection =
            fruit_loops_config(engine).injected_source_test;
        if (injection.enabled &&
            engine.iteration.fruit_iter >= injection.start_iteration) {
            const auto injected_samples =
                engine.ptcproc
                    .current_fruit_loop_injected_source_samples();
            logger->info(
                "fruit-loop injected-source test realized: iteration={} "
                "projected_samples={}",
                engine.iteration.fruit_iter, injected_samples);
            if (injected_samples == 0) {
                throw citlali::error::runtime(
                    "fruit-loop injected-source test selected zero projected "
                    "kernel samples");
            }
        }
    }
}

template <class Engine>
bool should_log_fruit_loop_iteration_start(const Engine &engine) {
    return fruit_loops_config(engine).enabled;
}

template <class Engine, class Logger>
void begin_fruit_loop_iteration(Engine &engine, const Logger &logger) {
    reset_fruit_loop_feedback_samples_if_available(engine);
    if (should_log_fruit_loop_iteration_start(engine)) {
        logger->info("starting fruit loops iteration {}", engine.iteration.fruit_iter);
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

    if constexpr (has_raw_timestream_plan_v<decltype(engine)>) {
        const auto &raw_plan = raw_timestream_plan(engine);
        if (raw_plan.observation &&
            raw_plan.observation->native_consumer_route ==
                NativeConsumerRoute::native_required) {
            // Native publication is fail-closed. The canonical reduction
            // index is written by the session boundary only after every
            // required reduction sidecar has committed and validated.
            logger->info(
                "deferring native reduction index until canonical publication");
            return;
        }
    }

    logger->info("making index files");
    todproc.make_index_file(engine.output_paths.redu_dir_name);
}

template <class Engine>
void advance_fruit_loop_iteration(Engine &engine) {
    engine.iteration.fruit_iter++;
}

template <class TodProc, class Logger>
void finalize_iteration_outputs(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    finalize_fruit_loop_iteration(engine, logger);

    make_reduction_iteration_index_file(todproc, logger);
    write_iteration_restart_checkpoint_if_needed(engine, logger);
    advance_fruit_loop_iteration(engine);
}

}  // namespace citlali::pipeline
