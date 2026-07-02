#pragma once

#include <citlali/core/pipeline/flux_calibration.h>
#include <citlali/core/pipeline/hwpr_loading.h>
#include <citlali/core/pipeline/observation_buffers.h>
#include <citlali/core/pipeline/observation_detector_diagnostics.h>
#include <citlali/core/pipeline/observation_date.h>
#include <citlali/core/pipeline/observation_exposure_time.h>
#include <citlali/core/pipeline/observation_sample_rate.h>
#include <citlali/core/pipeline/reduction_observation_calibration.h>
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
    const Logger &logger) {
    return configure_reduction_observation_calibration_if_needed<IsBeammap>(
        todproc, rawobs, rawobs_kids_meta, has_multiple_inputs, logger);
}

template <class Engine, class Logger>
bool prepare_reduction_observation_sample_rate(Engine &engine,
                                               const Logger &logger) {
    return configure_effective_sample_rate(engine, logger);
}

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class DateObs, class Logger>
bool prepare_reduction_observation_inputs(
    TodProc &todproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool has_multiple_inputs,
    MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, DateObs &&date_obs, const Logger &logger) {
    auto &engine = todproc.engine();

    if (!prepare_reduction_observation_calibration_state<IsBeammap>(
            todproc, rawobs, rawobs_kids_meta, has_multiple_inputs, logger)) {
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
        engine, std::forward<DateObs>(date_obs));
    record_reduction_observation_timing_gaps_if_needed(engine, logger);
    calculate_reduction_observation_scan_indices_if_needed(
        engine, has_multiple_inputs, logger);
    allocate_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, observation_index, logger);
    update_observation_exposure_time(engine);
    return true;
}

}  // namespace citlali::pipeline
