#pragma once

#include <citlali/core/pipeline/mapmaking_execution_plan.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

inline std::size_t checked_mapmaking_map_count(Eigen::Index count) {
    if (count <= 0) {
        throw std::logic_error("mapmaking map count must be positive");
    }
    return static_cast<std::size_t>(count);
}

inline std::size_t required_mapmaking_write_count(
    std::size_t map_count, std::size_t output_stage_count) {
    if (map_count == 0 || output_stage_count == 0) {
        throw std::logic_error(
            "mapmaking write cardinality requires positive counts");
    }
    if (map_count >
        std::numeric_limits<std::size_t>::max() / output_stage_count) {
        throw std::overflow_error("mapmaking write count overflow");
    }
    return map_count * output_stage_count;
}

template <class Engine>
std::size_t observation_map_output_stage_count(const Engine &engine) {
    return std::size_t{1} +
        ((!coadd_outputs_enabled(engine) &&
          should_write_filtered_outputs(engine))
             ? std::size_t{1}
             : std::size_t{0});
}

template <class Engine>
std::size_t coadd_map_output_stage_count(const Engine &engine) {
    return std::size_t{1} +
        (should_write_filtered_outputs(engine) ? std::size_t{1}
                                               : std::size_t{0});
}

template <class Engine>
void begin_mapmaking_iteration_if_available(Engine &engine) {
    if constexpr (has_mapmaking_plan_v<Engine>) {
        auto &plan = mapmaking_plan(engine);
        if (plan.initialized) {
            plan.begin_iteration();
        }
    }
}

template <class Engine>
void begin_mapmaking_observation_if_available(
    Engine &engine, std::size_t observation_index) {
    if constexpr (has_mapmaking_plan_v<Engine>) {
        auto &plan = mapmaking_plan(engine);
        if (!plan.initialized || !plan.effective.enabled) {
            return;
        }
        if (!std::isfinite(engine.omb.pixel_size_rad) ||
            engine.omb.pixel_size_rad <= 0.0) {
            throw std::logic_error(
                "effective map pixel size must be finite and positive");
        }
        const auto map_count =
            checked_mapmaking_map_count(engine.map_indices.n_maps);
        plan.begin_observation(
            observation_index, engine.observation_identity.obsnum,
            map_count, engine.omb.pixel_size_rad,
            required_mapmaking_write_count(
                map_count, observation_map_output_stage_count(engine)));
    }
}

inline void complete_mapmaking_observation(
    MapmakingExecutionPlan &plan) {
    if (plan.observations.empty()) {
        throw std::logic_error(
            "cannot complete mapmaking observation before it begins");
    }
    auto &observation = plan.observations.back();
    if (observation.outputs_completed) {
        throw std::logic_error(
            "mapmaking observation outputs already completed");
    }
    observation.outputs_completed = true;
    plan.realized.completed_observation_count =
        plan.realized.completed_observation_count.value_or(0) + 1;
}

template <class Engine>
void complete_mapmaking_observation_if_available(Engine &engine) {
    if constexpr (has_mapmaking_plan_v<Engine>) {
        auto &plan = mapmaking_plan(engine);
        if (plan.initialized && plan.effective.enabled) {
            complete_mapmaking_observation(plan);
        }
    }
}

template <class Engine>
void begin_mapmaking_coadd_if_available(Engine &engine) {
    if constexpr (has_mapmaking_plan_v<Engine>) {
        auto &plan = mapmaking_plan(engine);
        if (!plan.initialized || !plan.effective.enabled) {
            return;
        }
        const auto map_count =
            checked_mapmaking_map_count(engine.map_indices.n_maps);
        plan.begin_coadd(
            map_count,
            required_mapmaking_write_count(
                map_count, coadd_map_output_stage_count(engine)));
    }
}

inline void complete_mapmaking_coadd(MapmakingExecutionPlan &plan) {
    if (!plan.coadd.has_value()) {
        throw std::logic_error(
            "cannot complete mapmaking coadd before it begins");
    }
    if (plan.coadd->outputs_completed) {
        throw std::logic_error("mapmaking coadd outputs already completed");
    }
    plan.coadd->outputs_completed = true;
    plan.realized.completed_coadd_count =
        plan.realized.completed_coadd_count.value_or(0) + 1;
}

template <class Engine>
void complete_mapmaking_coadd_if_available(Engine &engine) {
    if constexpr (has_mapmaking_plan_v<Engine>) {
        auto &plan = mapmaking_plan(engine);
        if (plan.initialized && plan.effective.enabled) {
            complete_mapmaking_coadd(plan);
        }
    }
}

}  // namespace citlali::pipeline
