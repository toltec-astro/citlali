#pragma once

#include <citlali/core/pipeline/filtered_observation_outputs.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/raw_observation_outputs.h>

namespace citlali::pipeline {

template <class TodProc, class Logger>
void coadd_observation(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("coadding");
    if (!engine.rtcproc.run_polarization) {
        todproc.coadd();
    }
}

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
