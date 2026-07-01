#pragma once

#include <citlali/core/pipeline/filtered_observation_outputs.h>
#include <citlali/core/pipeline/observation_coadd_accumulation.h>
#include <citlali/core/pipeline/raw_observation_outputs.h>

namespace citlali::pipeline {

template <auto RawObsMap, auto FilteredObsMap, bool FitMaps, class TodProc,
          class Logger>
void write_observation_outputs_and_accumulate(TodProc &todproc,
                                              const Logger &logger) {
    auto &engine = todproc.engine();

    write_raw_observation_outputs<RawObsMap>(todproc, logger);

    if (engine.run_coadd) {
        coadd_observation(todproc, logger);
    }
    else {
        write_filtered_observation_outputs_if_needed<FilteredObsMap, FitMaps>(
            todproc, logger);
    }
}

}  // namespace citlali::pipeline
