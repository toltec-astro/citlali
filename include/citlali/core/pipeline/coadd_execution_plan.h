#pragma once

#include <citlali/core/config/coadd_config.h>
#include <citlali/core/pipeline/mapmaking_execution_plan.h>

#include <cstddef>
#include <optional>
#include <stdexcept>

namespace citlali::pipeline {

struct CoaddEffectiveResolutionRecord {
    bool mapmaking_enabled = false;
    bool requested_enabled = false;
    bool effective_enabled = false;
    bool disabled_by_mapmaking = false;
};

struct CoaddRealizedState {
    bool reduction_completed = false;
    bool coadd_executed = false;
    std::optional<std::size_t> map_count;
    std::optional<std::size_t> required_map_write_count;
    bool outputs_completed = false;
};

struct CoaddExecutionPlan {
    bool initialized = false;
    citlali::config::CoaddConfig requested;
    citlali::config::CoaddConfig effective;
    CoaddEffectiveResolutionRecord effective_resolution;
    CoaddRealizedState realized;

    void reset_from_request(
        const citlali::config::CoaddConfig &request,
        bool mapmaking_enabled) {
        initialized = true;
        requested = request;
        effective = request;
        if (!mapmaking_enabled) {
            effective.enabled = false;
        }
        effective_resolution = CoaddEffectiveResolutionRecord{
            mapmaking_enabled,
            request.enabled,
            effective.enabled,
            request.enabled && !mapmaking_enabled,
        };
        realized = {};
    }
};

inline void record_coadd_run_completed(
    CoaddExecutionPlan &plan,
    const MapmakingExecutionPlan &mapmaking_plan) {
    if (!plan.initialized) {
        throw std::logic_error("coadd plan is not initialized");
    }
    if (!mapmaking_plan.initialized ||
        !mapmaking_plan.realized.reduction_completed) {
        throw std::logic_error(
            "coadd completion requires completed mapmaking provenance");
    }

    const bool coadd_available = mapmaking_plan.coadd.has_value();
    if (plan.effective.enabled != coadd_available) {
        throw std::logic_error(
            "effective coadd policy differs from realized output state");
    }

    plan.realized = {};
    if (coadd_available) {
        const auto &coadd = *mapmaking_plan.coadd;
        if (!coadd.outputs_completed || coadd.map_count == 0 ||
            coadd.required_map_write_count < coadd.map_count) {
            throw std::logic_error(
                "coadd output cardinality is incomplete");
        }
        plan.realized.coadd_executed = true;
        plan.realized.map_count = coadd.map_count;
        plan.realized.required_map_write_count =
            coadd.required_map_write_count;
        plan.realized.outputs_completed = true;
    }
    plan.realized.reduction_completed = true;
}

}  // namespace citlali::pipeline
