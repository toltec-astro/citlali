#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/mapmaking_resolution.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

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
    std::optional<int> map_count;
    std::optional<double> effective_pixel_size_rad;
    std::optional<std::size_t> required_map_write_count;
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
    std::optional<MapmakingObservationState> observation;
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
        observation.reset();
        realized = {};
    }

    MapmakingObservationState &begin_observation() {
        if (!initialized) {
            throw std::logic_error("mapmaking plan is not initialized");
        }
        observation.emplace();
        return *observation;
    }
};

inline void record_mapmaking_run_completed(MapmakingExecutionPlan &plan) {
    if (!plan.initialized) {
        throw std::logic_error("mapmaking plan is not initialized");
    }
    plan.realized.reduction_completed = true;
    plan.realized.mapmaking_executed = plan.effective.enabled;
}

}  // namespace citlali::pipeline
