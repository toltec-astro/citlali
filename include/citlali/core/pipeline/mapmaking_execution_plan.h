#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/mapmaking_resolution.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct MapmakingEffectiveResolutionRecord {
    citlali::config::ReductionType reduction_type =
        citlali::config::ReductionType::science;
    citlali::config::MapGrouping requested_grouping =
        citlali::config::MapGrouping::automatic;
    citlali::config::MapGrouping effective_grouping =
        citlali::config::MapGrouping::array;
    bool automatic_grouping_resolved = false;
    bool detector_grouping_fell_back_to_array = false;
    std::string requested_unit = "mJy/beam";
    std::string effective_unit = "mJy/beam";
    bool uncalibrated_unit_substituted = false;
};

inline MapmakingEffectiveResolutionRecord resolve_mapmaking_request(
    const citlali::config::MapmakingConfig &request,
    citlali::config::ReductionType reduction_type,
    bool flux_calibration_enabled = true,
    citlali::config::TodType tod_type = citlali::config::TodType::xs) {
    const auto effective_grouping = effective_map_grouping_for_reduction(
        reduction_type, request.grouping);
    const std::string effective_unit = flux_calibration_enabled
        ? request.unit
        : std::string{citlali::config::to_string(tod_type)};
    return MapmakingEffectiveResolutionRecord{
        reduction_type,
        request.grouping,
        effective_grouping,
        citlali::config::is_automatic_map_grouping(request.grouping),
        citlali::config::is_detector_map_grouping(request.grouping) &&
            effective_grouping == citlali::config::MapGrouping::array,
        request.unit,
        effective_unit,
        !flux_calibration_enabled,
    };
}

struct MapmakingObservationState {
    std::size_t observation_index = 0;
    std::string obsnum;
    std::size_t map_count = 0;
    double effective_pixel_size_rad = 0.0;
    std::size_t required_map_write_count = 0;
    bool outputs_completed = false;
};

struct MapmakingCoaddState {
    std::size_t map_count = 0;
    std::size_t required_map_write_count = 0;
    bool outputs_completed = false;
};

struct MapmakingRealizedState {
    bool reduction_completed = false;
    bool mapmaking_executed = false;
    std::optional<std::size_t> completed_observation_count;
    std::optional<std::size_t> completed_coadd_count;
};

struct MapmakingExecutionPlan {
    bool initialized = false;
    citlali::config::MapmakingConfig requested;
    citlali::config::MapmakingConfig effective;
    MapmakingEffectiveResolutionRecord effective_resolution;
    std::vector<MapmakingObservationState> observations;
    std::optional<MapmakingCoaddState> coadd;
    MapmakingRealizedState realized;

    void reset_from_request(
        const citlali::config::MapmakingConfig &request,
        citlali::config::ReductionType reduction_type,
        bool flux_calibration_enabled = true,
        citlali::config::TodType tod_type = citlali::config::TodType::xs) {
        initialized = true;
        requested = request;
        effective = request;
        effective_resolution =
            resolve_mapmaking_request(
                request, reduction_type, flux_calibration_enabled,
                tod_type);
        effective.grouping = effective_resolution.effective_grouping;
        effective.unit = effective_resolution.effective_unit;
        observations.clear();
        coadd.reset();
        realized = {};
    }

    void begin_iteration() {
        if (!initialized) {
            throw std::logic_error("mapmaking plan is not initialized");
        }
        observations.clear();
        coadd.reset();
        realized = {};
        realized.completed_observation_count = std::size_t{0};
        realized.completed_coadd_count = std::size_t{0};
    }

    MapmakingObservationState &begin_observation(
        std::size_t observation_index, std::string obsnum,
        std::size_t map_count, double effective_pixel_size_rad,
        std::size_t required_map_write_count) {
        if (!initialized) {
            throw std::logic_error("mapmaking plan is not initialized");
        }
        observations.push_back(MapmakingObservationState{
            observation_index, std::move(obsnum), map_count,
            effective_pixel_size_rad, required_map_write_count, false});
        return observations.back();
    }

    MapmakingCoaddState &begin_coadd(
        std::size_t map_count,
        std::size_t required_map_write_count) {
        if (!initialized) {
            throw std::logic_error("mapmaking plan is not initialized");
        }
        coadd.emplace(MapmakingCoaddState{
            map_count, required_map_write_count, false});
        return *coadd;
    }
};

inline void record_mapmaking_run_completed(MapmakingExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error("mapmaking plan is not initialized");
    }
    if (!plan.realized.completed_observation_count.has_value() ||
        !plan.realized.completed_coadd_count.has_value()) {
        throw std::logic_error(
            "mapmaking iteration cardinality was not initialized");
    }
    const auto completed_observations = static_cast<std::size_t>(
        std::count_if(
            plan.observations.begin(), plan.observations.end(),
            [](const auto &observation) {
                return observation.outputs_completed;
            }));
    if (*plan.realized.completed_observation_count !=
            completed_observations ||
        completed_observations != plan.observations.size()) {
        throw std::logic_error(
            "mapmaking observation cardinality is incomplete");
    }
    const std::size_t completed_coadds =
        plan.coadd && plan.coadd->outputs_completed ? 1 : 0;
    if (*plan.realized.completed_coadd_count != completed_coadds ||
        (plan.coadd && !plan.coadd->outputs_completed)) {
        throw std::logic_error("mapmaking coadd cardinality is incomplete");
    }
    if (plan.effective.enabled && plan.observations.empty()) {
        throw std::logic_error(
            "mapmaking completed without observation products");
    }
    if (!plan.effective.enabled &&
        (!plan.observations.empty() || plan.coadd.has_value())) {
        throw std::logic_error(
            "disabled mapmaking recorded product cardinality");
    }
    plan.realized.reduction_completed = true;
    plan.realized.mapmaking_executed = plan.effective.enabled;
}

}  // namespace citlali::pipeline
