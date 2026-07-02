#pragma once

#include <citlali/core/pipeline/telescope_data_loading.h>
#include <citlali/core/pipeline/telescope_pointing_operations.h>

namespace citlali::pipeline {

template <class TodProc, class RawObs, class Logger>
void load_and_point_telescope_data_if_needed(TodProc &todproc,
                                             const RawObs &rawobs,
                                             bool should_load,
                                             const Logger &logger) {
    if (!should_load) {
        return;
    }

    load_and_align_telescope_data(todproc, rawobs, logger);
    calculate_telescope_pointing(todproc, logger);
}

template <class TodProc, class RawObs, class Logger>
void load_and_point_reduction_observation_telescope_data_if_needed(
    TodProc &todproc, const RawObs &rawobs, bool should_load,
    const Logger &logger) {
    load_and_point_telescope_data_if_needed(
        todproc, rawobs, should_load, logger);
}

}  // namespace citlali::pipeline
