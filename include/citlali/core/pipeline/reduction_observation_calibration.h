#pragma once

#include <citlali/core/pipeline/kids_metadata.h>
#include <citlali/core/pipeline/observation_calibration_config.h>

#include <cstddef>

namespace citlali::pipeline {

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class Logger>
bool configure_reduction_observation_calibration_if_needed(
    TodProc &todproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool should_configure,
    std::size_t observation_index, const Logger &logger) {
    if (!should_configure) {
        return true;
    }

    auto &engine = todproc.engine();
    configure_observation_calibration_with_context<IsBeammap>(
        todproc, rawobs, rawobs_kids_meta, observation_index, logger);

    update_sample_rate_from_rawobs_meta(engine, rawobs_kids_meta, logger);
    return true;
}

}  // namespace citlali::pipeline
