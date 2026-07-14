#pragma once

#include <citlali/core/pipeline/pointing_execution_plan.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <stdexcept>

namespace citlali::pipeline {

template <class Engine>
void begin_pointing_iteration_if_available(Engine &engine) {
    if constexpr (has_pointing_plan_v<Engine>) {
        auto &plan = pointing_plan(engine);
        if (plan.initialized) {
            plan.begin_iteration();
        }
    }
}

template <class Engine>
void begin_pointing_observation_if_available(Engine &engine) {
    if constexpr (has_pointing_plan_v<Engine> &&
                  has_mapmaking_plan_v<Engine>) {
        auto &plan = pointing_plan(engine);
        auto &maps = mapmaking_plan(engine);
        if (!plan.initialized ||
            !plan.effective_resolution.mapmaking_enabled) {
            return;
        }
        if (maps.observations.empty()) {
            throw std::logic_error(
                "pointing observation requires mapmaking observation state");
        }
        const auto &observation = maps.observations.back();
        plan.begin_observation(
            observation.observation_index, observation.obsnum,
            observation.map_count);
    }
}

template <class Engine>
void complete_pointing_observation_if_available(Engine &engine) {
    if constexpr (has_pointing_plan_v<Engine>) {
        auto &plan = pointing_plan(engine);
        if (plan.initialized &&
            plan.effective_resolution.mapmaking_enabled) {
            complete_pointing_observation(plan);
        }
    }
}

}  // namespace citlali::pipeline
