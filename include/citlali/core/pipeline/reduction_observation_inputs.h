#pragma once

#include <citlali/core/pipeline/flux_calibration.h>
#include <citlali/core/pipeline/hwpr_loading.h>
#include <citlali/core/pipeline/observation_buffers.h>
#include <citlali/core/pipeline/observation_detector_diagnostics.h>
#include <citlali/core/pipeline/observation_date.h>
#include <citlali/core/pipeline/reduction_observation_date.h>
#include <citlali/core/pipeline/observation_exposure_time.h>
#include <citlali/core/pipeline/observation_sample_rate.h>
#include <citlali/core/pipeline/raw_timestream_observation_shadow.h>
#include <citlali/core/pipeline/reduction_observation_calibration.h>
#include <citlali/core/pipeline/runtime_policy.h>
#include <citlali/core/pipeline/rawobs_observation_output_layout.h>
#include <citlali/core/pipeline/scan_indices.h>
#include <citlali/core/pipeline/telescope_pointing.h>
#include <citlali/core/pipeline/timing_gap_output.h>

#include <cstddef>
#include <utility>

namespace citlali::pipeline {

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class Logger>
bool prepare_reduction_observation_calibration_state(
    TodProc &todproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool has_multiple_inputs,
    std::size_t observation_index, const Logger &logger) {
    return configure_reduction_observation_calibration_if_needed<IsBeammap>(
        todproc, rawobs, rawobs_kids_meta, has_multiple_inputs,
        observation_index, logger);
}

template <class Engine, class Logger>
bool prepare_reduction_observation_sample_rate(Engine &engine,
                                               const Logger &logger) {
    if (!configure_effective_sample_rate(engine, logger)) {
        return false;
    }
    if constexpr (has_raw_timestream_plan_v<Engine>) {
        auto &plan = raw_timestream_plan(engine);
        if (plan.initialized) {
            const auto shadow = begin_raw_timestream_observation_shadow(
                plan, runtime_reduction_type(engine), engine.telescope.fsmp,
                engine.telescope.d_fsmp, engine.rtcproc);
            if (!shadow.exact) {
                logger->error(
                    "typed raw observation shadow differs from legacy state: {}",
                    shadow.diagnostic());
                return false;
            }
            if (shadow.edge_guard_deferred) {
                logger->debug(
                    "typed raw observation shadow deferred edge-guard parity for frequency-derived downsample factor");
            }
        }
    }
    return true;
}

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class DateObsFactory, class Logger>
bool prepare_reduction_observation_inputs(
    TodProc &todproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool has_multiple_inputs,
    MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, DateObsFactory &&date_obs_factory,
    const Logger &logger) {
    auto &engine = todproc.engine();

    if (!prepare_reduction_observation_calibration_state<IsBeammap>(
            todproc, rawobs, rawobs_kids_meta, has_multiple_inputs,
            observation_index, logger)) {
        return false;
    }

    if (!prepare_reduction_observation_sample_rate(engine, logger)) {
        return false;
    }

    load_reduction_observation_detector_diagnostics(todproc, rawobs, logger);
    prepare_reduction_observation_output_layout(
        engine, rawobs_kids_meta, logger);
    load_reduction_observation_hwpr_data_if_requested(engine, rawobs, logger);
    calculate_reduction_observation_flux_calibration(engine, logger);
    load_and_point_reduction_observation_telescope_data_if_needed(
        todproc, rawobs, has_multiple_inputs, logger);
    append_reduction_observation_date(
        engine, make_reduction_observation_date_obs(
                    std::forward<DateObsFactory>(date_obs_factory), engine));
    record_reduction_observation_timing_gaps_if_needed(engine, logger);
    calculate_reduction_observation_scan_indices_if_needed(
        engine, has_multiple_inputs, logger);
    begin_native_consumer_observation_if_available<IsBeammap>(
        engine, observation_index);
    allocate_reduction_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, observation_index, logger);
    update_reduction_observation_exposure_time(engine);
    return true;
}

}  // namespace citlali::pipeline
