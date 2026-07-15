#pragma once

#include <citlali/core/pipeline/beammap_execution_plan.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>
#include <citlali/core/pipeline/post_processing_execution_plan.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace citlali::pipeline {

template <class Count>
std::size_t checked_beammap_count(Count count, const char *label) {
    if constexpr (std::is_signed_v<Count>) {
        if (count <= 0) {
            throw std::logic_error(
                std::string{"beammap "} + label + " must be positive");
        }
    }
    else if (count == 0) {
        throw std::logic_error(
            std::string{"beammap "} + label + " must be positive");
    }
    return static_cast<std::size_t>(count);
}

template <class Count>
std::size_t checked_beammap_nonnegative_count(
    Count count, const char *label) {
    if constexpr (std::is_signed_v<Count>) {
        if (count < 0) {
            throw std::logic_error(
                std::string{"beammap "} + label +
                " must be nonnegative");
        }
    }
    return static_cast<std::size_t>(count);
}

inline BeammapIterationPhase resolve_beammap_iteration_phase(
    const BeammapExecutionPlan &plan, bool locator_iteration,
    bool measurement_iteration, bool first_measurement_iteration) {
    if (plan.resolution().legacy_phase_behavior) {
        return BeammapIterationPhase::legacy;
    }
    if (locator_iteration) {
        return BeammapIterationPhase::locator;
    }
    if (first_measurement_iteration) {
        return BeammapIterationPhase::measurement_start;
    }
    if (measurement_iteration) {
        return BeammapIterationPhase::measurement;
    }
    return BeammapIterationPhase::pre_measurement;
}

template <class Engine>
void begin_beammap_run_if_available(Engine &engine) {
    if constexpr (has_beammap_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (plan.initialized()) {
            plan.begin_iteration();
        }
    }
}

template <class Engine, class DetectorCount, class ScanCount>
void begin_beammap_observation_if_available(
    Engine &engine, DetectorCount detector_count, ScanCount scan_count) {
    if constexpr (has_beammap_plan_v<Engine> &&
                  has_mapmaking_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (!plan.initialized() ||
            !plan.resolution().mapmaking_enabled) {
            return;
        }
        const auto &maps = mapmaking_plan(engine);
        if (maps.observations.empty()) {
            throw std::logic_error(
                "beammap observation requires mapmaking observation state");
        }
        const auto &observation = maps.observations.back();
        plan.begin_observation(
            observation.observation_index, observation.obsnum,
            beammap_photometry_config(engine),
            checked_beammap_count(detector_count, "detector count"),
            observation.map_count,
            checked_beammap_count(scan_count, "scan count"));
    }
}

template <class Engine, class IterationIndex, class ActiveMapCount>
void begin_beammap_internal_iteration_if_available(
    Engine &engine, IterationIndex iteration_index,
    bool locator_iteration, bool measurement_iteration,
    bool first_measurement_iteration, ActiveMapCount active_map_count) {
    if constexpr (has_beammap_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (!plan.initialized() ||
            !plan.resolution().mapmaking_enabled) {
            return;
        }
        plan.begin_internal_iteration(
            checked_beammap_nonnegative_count(
                iteration_index, "iteration index"),
            resolve_beammap_iteration_phase(
                plan, locator_iteration, measurement_iteration,
                first_measurement_iteration),
            checked_beammap_count(active_map_count, "active map count"));
    }
}

template <class Engine>
void record_beammap_source_aware_rtc_if_available(
    Engine &engine, bool rerun) {
    if constexpr (has_beammap_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (plan.initialized() &&
            plan.resolution().mapmaking_enabled) {
            plan.record_source_aware_rtc_rerun(rerun);
        }
    }
}

template <class Engine>
void record_beammap_mapmaking_pass_completed_if_available(
    Engine &engine) {
    if constexpr (has_beammap_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (plan.initialized() &&
            plan.resolution().mapmaking_enabled) {
            plan.record_mapmaking_pass_completed();
        }
    }
}

template <class Engine>
void record_beammap_fitting_completed_if_available(Engine &engine) {
    if constexpr (has_beammap_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (plan.initialized() &&
            plan.resolution().mapmaking_enabled) {
            plan.record_fitting_completed();
        }
    }
}

template <class Engine, class OutputIteration, class DetectorCount,
          class SlotCount, class MaximumSampleCount>
void record_beammap_detector_tod_written_if_available(
    Engine &engine, OutputIteration output_iteration,
    DetectorCount detector_count, SlotCount slot_count,
    MaximumSampleCount maximum_sample_count) {
    if constexpr (has_beammap_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (plan.initialized() &&
            plan.resolution().mapmaking_enabled) {
            plan.record_detector_tod_written(
                checked_beammap_nonnegative_count(
                    output_iteration, "detector TOD output iteration"),
                checked_beammap_count(
                    detector_count, "detector TOD detector count"),
                checked_beammap_count(
                    slot_count, "detector TOD slot count"),
                checked_beammap_count(
                    maximum_sample_count,
                    "detector TOD maximum sample count"));
        }
    }
}

template <class Engine, class ConvergedCount>
void complete_beammap_internal_iteration_if_available(
    Engine &engine, ConvergedCount converged_count,
    BeammapTerminationReason termination_reason) {
    if constexpr (has_beammap_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (plan.initialized() &&
            plan.resolution().mapmaking_enabled) {
            plan.complete_internal_iteration(
                checked_beammap_nonnegative_count(
                    converged_count, "converged map count"),
                termination_reason);
        }
    }
}

template <class Engine>
void complete_beammap_observation_if_available(Engine &engine) {
    if constexpr (has_beammap_plan_v<Engine>) {
        auto &plan = beammap_plan(engine);
        if (plan.initialized() &&
            plan.resolution().mapmaking_enabled) {
            plan.complete_observation();
        }
    }
}

inline void record_beammap_run_completed(
    BeammapExecutionPlan &plan,
    const MapmakingExecutionPlan &mapmaking_plan,
    const PostProcessingExecutionPlan &post_processing_plan) {
    if (!plan.initialized()) {
        throw std::logic_error("beammap plan is not initialized");
    }
    if (!mapmaking_plan.initialized ||
        !mapmaking_plan.realized.reduction_completed) {
        throw std::logic_error(
            "beammap completion requires completed mapmaking");
    }
    if (!post_processing_plan.initialized ||
        !post_processing_plan.realized.reduction_completed ||
        !post_processing_plan.realized.outputs_completed) {
        throw std::logic_error(
            "beammap completion requires completed post-processing");
    }
    if (!citlali::config::is_beammap_reduction_type(
            mapmaking_plan.effective_resolution.reduction_type) ||
        !citlali::config::is_beammap_reduction_type(
            post_processing_plan.effective_resolution.reduction_type)) {
        throw std::logic_error(
            "beammap completion requires beammap domain plans");
    }
    if (plan.resolution().mapmaking_enabled !=
            mapmaking_plan.effective.enabled ||
        plan.resolution().mapmaking_enabled !=
            post_processing_plan.effective_resolution.mapmaking_enabled) {
        throw std::logic_error(
            "beammap and dependent mapmaking policy differ");
    }

    const auto &observations = plan.observations();
    const auto beammap_executed =
        plan.resolution().mapmaking_enabled;
    if (!beammap_executed) {
        if (!observations.empty() ||
            post_processing_plan.realized.beammap_fits.context_count != 0) {
            throw std::logic_error(
                "disabled beammap execution recorded realized state");
        }
        plan.complete_reduction(false);
        return;
    }

    if (observations.size() != mapmaking_plan.observations.size() ||
        observations.empty()) {
        throw std::logic_error(
            "beammap and mapmaking observation counts differ");
    }

    std::size_t completed_iteration_count = 0;
    for (std::size_t index = 0; index < observations.size(); ++index) {
        const auto &beammap = observations[index];
        const auto &mapmaking = mapmaking_plan.observations[index];
        if (!beammap.outputs_completed ||
            !beammap.terminal_iteration.has_value() ||
            beammap.termination_reason == BeammapTerminationReason::none ||
            !mapmaking.outputs_completed ||
            beammap.observation_index != mapmaking.observation_index ||
            beammap.obsnum != mapmaking.obsnum ||
            beammap.map_count != mapmaking.map_count) {
            throw std::logic_error(
                "beammap and mapmaking observation state differs");
        }
        for (const auto &iteration : beammap.iterations) {
            if (!iteration.completed) {
                throw std::logic_error(
                    "beammap iteration cardinality is incomplete");
            }
        }
        completed_iteration_count += beammap.iterations.size();
    }

    const auto &realized = plan.realized();
    if (!realized.completed_observation_count.has_value() ||
        *realized.completed_observation_count != observations.size() ||
        realized.completed_iteration_count != completed_iteration_count) {
        throw std::logic_error(
            "beammap realized cardinality is incomplete");
    }
    if (post_processing_plan.realized.beammap_fits.context_count !=
        completed_iteration_count) {
        throw std::logic_error(
            "beammap and post-processing fit context counts differ");
    }
    plan.complete_reduction(true);
}

}  // namespace citlali::pipeline
