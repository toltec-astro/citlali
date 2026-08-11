#pragma once

#include <citlali/core/pipeline/flxscale_correction.h>
#include <citlali/core/pipeline/initial_observation_map_dimensions.h>
#include <citlali/core/pipeline/kids_metadata.h>
#include <citlali/core/pipeline/observation_calibration_config.h>
#include <citlali/core/pipeline/observation_input_checks.h>
#include <citlali/core/pipeline/scan_indices.h>
#include <citlali/core/pipeline/telescope_data_loading.h>
#include <citlali/core/pipeline/telescope_pointing.h>
#include <citlali/core/pipeline/timestream_alignment_helpers.h>

#include <cstddef>
#include <citlali/core/utils/constants.h>

namespace citlali::pipeline {

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class Logger>
bool prepare_initial_observation_setup(TodProc &todproc, const RawObs &rawobs,
                                       const RawObsKidsMeta &rawobs_kids_meta,
                                       MapExtents &map_extents,
                                       MapCoords &map_coords,
                                       std::size_t observation_index,
                                       const Logger &logger) {
    auto &engine = todproc.engine();
    configure_observation_calibration_with_context<IsBeammap>(
        todproc, rawobs, rawobs_kids_meta, observation_index, logger);
    if (!apply_flxscale_correction(engine, rawobs, logger)) {
        return false;
    }

    check_observation_inputs(todproc, rawobs, logger);
    update_sample_rate_from_rawobs_meta(engine, rawobs_kids_meta, logger);
    const auto tel_path = telescope_data_filepath(rawobs);
    load_telescope_data_file(engine, tel_path, logger);
    overwrite_map_center_if_configured(engine, logger);
    if (should_align_telescope_timestreams(engine)) {
        align_telescope_timestreams(todproc, rawobs, logger);
    }
    else {
        reset_simulated_telescope_indices(engine, rawobs);
    }
    calculate_telescope_pointing(todproc, logger);
    calculate_scan_indices(engine, logger);
    calculate_initial_observation_map_dimensions(
        todproc, map_extents, map_coords, logger);
    return true;
}

}  // namespace citlali::pipeline
