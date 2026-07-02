#pragma once

#include <citlali/core/pipeline/filtered_observation_outputs.h>
#include <citlali/core/pipeline/observation_coadd_accumulation.h>
#include <citlali/core/pipeline/raw_observation_outputs.h>

namespace citlali::pipeline {

template <class Engine>
bool should_accumulate_observation_coadd(const Engine &engine) {
    return engine.run_coadd;
}

template <class TodProc, class Logger>
void write_coadded_observation_outputs(TodProc &todproc,
                                       const Logger &logger) {
    coadd_observation(todproc, logger);
}

template <auto FilteredObsMap, bool FitMaps, class TodProc, class Logger>
void write_noncoadded_observation_outputs(TodProc &todproc,
                                          const Logger &logger) {
    write_filtered_observation_outputs_if_needed<FilteredObsMap, FitMaps>(
        todproc, logger);
}

template <auto RawObsMap, auto FilteredObsMap, bool FitMaps, class TodProc,
          class Logger>
void write_observation_outputs_and_accumulate(TodProc &todproc,
                                              const Logger &logger) {
    auto &engine = todproc.engine();

    write_raw_observation_outputs<RawObsMap>(todproc, logger);

    if (should_accumulate_observation_coadd(engine)) {
        write_coadded_observation_outputs(todproc, logger);
    }
    else {
        write_noncoadded_observation_outputs<FilteredObsMap, FitMaps>(
            todproc, logger);
    }
}

}  // namespace citlali::pipeline
