#pragma once

#include <citlali/core/pipeline/map_center_override.h>
#include <citlali/core/pipeline/telescope_data_source.h>
#include <citlali/core/pipeline/telescope_timestream_alignment.h>

namespace citlali::pipeline {

template <class TodProc, class RawObs, class Logger>
void load_and_align_telescope_data(TodProc &todproc, const RawObs &rawobs,
                                   const Logger &logger) {
    auto &engine = todproc.engine();

    auto tel_path = telescope_data_filepath(rawobs);
    load_telescope_data_file(engine, tel_path, logger);

    overwrite_map_center_if_configured(engine, logger);

    if (should_align_telescope_timestreams(engine)) {
        align_telescope_timestreams(todproc, rawobs, logger);
    }
    else {
        reset_simulated_telescope_indices(engine, rawobs);
    }
}

}  // namespace citlali::pipeline
