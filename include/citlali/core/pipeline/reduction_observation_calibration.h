#pragma once

#include <citlali/core/pipeline/flxscale_correction.h>
#include <citlali/core/pipeline/kids_metadata.h>
#include <citlali/core/pipeline/observation_calibration_config.h>

namespace citlali::pipeline {

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class Logger>
bool configure_reduction_observation_calibration_if_needed(
    TodProc &todproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool should_configure,
    const Logger &logger) {
    if (!should_configure) {
        return true;
    }

    auto &engine = todproc.engine();
    configure_observation_calibration<IsBeammap>(todproc, rawobs, logger);
    if (!apply_flxscale_correction(engine, rawobs, logger)) {
        return false;
    }

    update_sample_rate_from_rawobs_meta(engine, rawobs_kids_meta, logger);
    return true;
}

}  // namespace citlali::pipeline
