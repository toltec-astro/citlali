#pragma once

#include <citlali/core/pipeline/telescope_data_loading.h>
#include <citlali/core/pipeline/telescope_pointing_operations.h>
#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/utils/constants.h>

#include <cstddef>

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

template <class TodProc, class RawObs>
void capture_reduction_observation_rtc_sampling_source_motion(
    TodProc &todproc, const RawObs &rawobs, std::size_t observation_index) {
    auto &engine = todproc.engine();
    const auto telescope_source = telescope_data_filepath(rawobs);
    engine.alignment.rtc_sampling_source_motion =
        capture_rtc_sampling_source_motion(engine.telescope.tel_data,
                                           RAD_TO_ASEC);
    bind_rtc_sampling_source_observation_identity(
        engine.alignment.rtc_sampling_source_motion, observation_index,
        engine.observation_identity.obsnum, telescope_source);
}

}  // namespace citlali::pipeline
