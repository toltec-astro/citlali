#pragma once

#include <citlali/core/config/config_value.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/obsnum_format.h>
#include <citlali/core/pipeline/processed_timestream_execution_plan.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/reduction_restart_checkpoint.h>

#include <string>
#include <vector>

namespace citlali::pipeline {

template <class Engine>
std::vector<std::string> reduction_restart_observation_ids(
    const Engine &engine) {
    if constexpr (has_astrometry_plan_v<Engine>) {
        const auto &plan = astrometry_plan(engine);
        if (!plan.initialized || plan.observations.empty() ||
            plan.observations.size() != plan.expected_observation_count) {
            throw citlali::error::runtime(
                "fruit-loop restart requires the complete reduction observation identity plan");
        }
        std::vector<std::string> result;
        result.reserve(plan.observations.size());
        for (const auto &observation : plan.observations) {
            result.push_back(format_obsnum(observation.obsnum));
        }
        return result;
    }
    else if constexpr (requires {
                           engine.cmb.obsnums;
                           engine.omb.obsnums;
                       }) {
        if (!engine.cmb.obsnums.empty()) {
            return engine.cmb.obsnums;
        }
        return engine.omb.obsnums;
    }
    else {
        return engine.omb.obsnums;
    }
}

template <class Engine>
bool fruit_loop_restart_requested(const Engine &engine) {
    return fruit_loops_config(engine).enabled &&
           citlali::config::has_nonempty_config_value(
               fruit_loops_config(engine).restart_path);
}

template <class Engine>
const auto *fruit_loop_restart_resolution(const Engine &engine) {
    using Resolution = ProcessedTimestreamEffectiveResolutionRecord::
        FruitLoopRestartResolution;
    if constexpr (has_processed_timestream_plan_v<Engine>) {
        const auto &restart = processed_timestream_plan(engine)
                                  .effective_resolutions.fruit_loop_restart;
        return restart ? &*restart : static_cast<const Resolution *>(nullptr);
    }
    else {
        return static_cast<const Resolution *>(nullptr);
    }
}

template <class Engine>
bool first_restarted_iteration(const Engine &engine) {
    const auto *restart = fruit_loop_restart_resolution(engine);
    return restart != nullptr &&
           engine.iteration.fruit_iter == restart->next_iteration;
}

template <class Engine, class IterationState, class Logger>
void initialize_fruit_loop_restart_if_requested(
    Engine &engine, IterationState &iteration_state, const Logger &logger) {
    if (!fruit_loop_restart_requested(engine)) {
        return;
    }
    if constexpr (!has_processed_timestream_plan_v<Engine> ||
                  !requires {
                      reduction_restart_observation_ids(engine);
                      load_reduction_restart_checkpoint(
                          fruit_loops_config(engine).restart_path,
                          fruit_loops_config(engine).type,
                          reduction_restart_observation_ids(engine),
                          learning_config(engine),
                          engine.learning);
                  }) {
        throw citlali::error::runtime(
            "exact fruit-loop restart is unavailable for this reduction engine");
    }
    else {
        const auto observation_ids =
            reduction_restart_observation_ids(engine);
        auto summary = load_reduction_restart_checkpoint(
            fruit_loops_config(engine).restart_path,
            fruit_loops_config(engine).type, observation_ids,
            learning_config(engine), engine.learning);
        if (summary.next_iteration >= fruit_loops_config(engine).max_iters) {
            throw citlali::error::runtime(
                "fruit-loop restart checkpoint next iteration " +
                std::to_string(summary.next_iteration) +
                " is not below configured max_iters " +
                std::to_string(fruit_loops_config(engine).max_iters));
        }
        engine.iteration.fruit_iter = summary.next_iteration;
        iteration_state.start_iteration = summary.next_iteration;
        iteration_state.restarted = true;
        processed_timestream_plan(engine)
            .effective_resolutions.fruit_loop_restart =
            ProcessedTimestreamEffectiveResolutionRecord::
                FruitLoopRestartResolution{
                    summary.source_reduction_dir.string(),
                    summary.checkpoint_path.string(),
                    summary.creator_version,
                    summary.completed_iteration,
                    summary.next_iteration,
                    summary.effective_sample_mask_intervals,
                    summary.effective_detector_penalties,
                };
        logger->info(
            "loaded exact fruit-loop restart: checkpoint={} completed_iteration={} next_iteration={} effective_sample_mask_intervals={} effective_detector_penalties={} creator_version={}",
            summary.checkpoint_path.string(), summary.completed_iteration,
            summary.next_iteration,
            summary.effective_sample_mask_intervals,
            summary.effective_detector_penalties, summary.creator_version);
    }
}

template <class Engine, class Logger>
void write_iteration_restart_checkpoint_if_needed(
    const Engine &engine, const Logger &logger) {
    if (!fruit_loops_config(engine).enabled) {
        return;
    }
    if constexpr (requires {
                      reduction_restart_observation_ids(engine);
                      write_reduction_restart_checkpoint(
                          engine.output_paths.redu_dir_name,
                          engine.iteration.fruit_iter,
                          fruit_loops_config(engine).type,
                          reduction_restart_observation_ids(engine),
                          learning_config(engine),
                          engine.learning);
                  }) {
        const auto observation_ids =
            reduction_restart_observation_ids(engine);
        write_reduction_restart_checkpoint(
            engine.output_paths.redu_dir_name, engine.iteration.fruit_iter,
            fruit_loops_config(engine).type, observation_ids,
            learning_config(engine), engine.learning);
        logger->info(
            "fruit-loop restart checkpoint: {}",
            reduction_restart_checkpoint_path(
                engine.output_paths.redu_dir_name)
                .string());
    }
}

}  // namespace citlali::pipeline
