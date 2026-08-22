#pragma once

#include <citlali/core/pipeline/observation_buffer_allocation.h>
#include <citlali/core/pipeline/observation_map_access.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/native_cohort_product_provenance_v2.h>
#include <citlali/core/pipeline/native_consumer_execution_policy.h>
#include <citlali/core/pipeline/native_consumer_mode_policy.h>
#include <citlali/core/pipeline/pointing_provenance_lifecycle.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/runtime_policy.h>

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace citlali::pipeline {

template <bool IsBeammap, class Engine>
void begin_native_consumer_observation_if_available(
    Engine &engine, std::size_t observation_index) {
    if constexpr (!has_raw_timestream_plan_v<Engine>) {
        return;
    }
    else {
        auto &plan = raw_timestream_plan(engine);
        if (!plan.initialized || !plan.observation) {
            throw std::logic_error(
                "native consumer routing requires an active raw observation");
        }
        if (plan.realized.execution_completed ||
            plan.realized.native_cohort_provenance ||
            plan.observation->native_cohort_lineage) {
            throw std::logic_error(
                "native consumer routing is already active or realized");
        }
        const auto reduction_type = runtime_reduction_type(engine);
        if (IsBeammap !=
            (reduction_type == citlali::config::ReductionType::beammap)) {
            throw std::logic_error(
                "compile-time and typed native consumer modes disagree");
        }
        const auto relation =
            engine.calib.apt_detector_relation_v2_handle();
        const auto carriers = engine.alignment.native_carriers;
        const auto route = resolve_native_consumer_route({
            reduction_type, mapmaking_config(engine).grouping,
            static_cast<bool>(relation), static_cast<bool>(carriers)});

        std::shared_ptr<NativeCohortObservationLineageV2> lineage;
        if (route == NativeConsumerRoute::native_required) {
            require_supported_native_consumer_observation(engine);
            const auto scan_count = engine.telescope.scan_indices.cols();
            if (scan_count <= 0) {
                throw std::logic_error(
                    "native consumer requires positive scan cardinality");
            }
            lineage = NativeCohortObservationLineageV2::create(
                make_native_cohort_observation_binding_v2(
                    observation_index, *relation, carriers),
                static_cast<std::size_t>(scan_count));
        }
        plan.observation->native_consumer_route = route;
        plan.observation->native_cohort_lineage = std::move(lineage);
    }
}

template <class TodProc, class MapExtents, class MapCoords, class Logger>
void allocate_observation_map_buffers_if_needed(
    TodProc &todproc, MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, const Logger &logger) {
    auto &engine = todproc.engine();

    if (!should_allocate_observation_map_buffers(engine)) {
        return;
    }

    allocate_observation_map_buffers(
        todproc, observation_map_extent_at(map_extents, observation_index),
        observation_map_coord_at(map_coords, observation_index),
        logger);
    begin_mapmaking_observation_if_available(engine, observation_index);
    begin_pointing_observation_if_available(engine);
}

template <class TodProc, class MapExtents, class MapCoords, class Logger>
void allocate_reduction_observation_map_buffers_if_needed(
    TodProc &todproc, MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, const Logger &logger) {
    allocate_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, observation_index, logger);
}

}  // namespace citlali::pipeline
