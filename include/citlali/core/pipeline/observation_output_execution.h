#pragma once

#include <citlali/core/pipeline/beammap_provenance_lifecycle.h>
#include <citlali/core/pipeline/filtered_observation_outputs.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/observation_coadd_accumulation.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/pointing_provenance_lifecycle.h>
#include <citlali/core/pipeline/raw_observation_outputs.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class Engine>
bool should_accumulate_observation_coadd(const Engine &engine) {
    return coadd_outputs_enabled(engine);
}

template <class TodProc, class Logger>
void write_coadded_observation_outputs(TodProc &todproc,
                                       StageProfileCollector &stage_profile,
                                       const Logger &logger) {
    coadd_observation(todproc, stage_profile, logger);
}

template <auto FilteredObsMap, bool FitMaps, class TodProc, class Logger>
void write_noncoadded_observation_outputs(TodProc &todproc,
                                          StageProfileCollector &stage_profile,
                                          const Logger &logger) {
    write_filtered_observation_outputs_if_needed<FilteredObsMap, FitMaps>(
        todproc, stage_profile, logger);
}

template <auto RawObsMap, auto FilteredObsMap, bool FitMaps, class TodProc,
          class Logger>
void write_observation_outputs_and_accumulate(TodProc &todproc,
                                              StageProfileCollector &stage_profile,
                                              const Logger &logger) {
    auto &engine = todproc.engine();
    (void)stage_profile;
    const auto profile_scope =
        profile_stage(stage_profile, "observation.outputs_and_accumulation", logger);

    write_raw_observation_outputs<RawObsMap>(
        todproc, stage_profile, logger);

    if (should_accumulate_observation_coadd(engine)) {
        write_coadded_observation_outputs(todproc, stage_profile, logger);
    }
    else {
        write_noncoadded_observation_outputs<FilteredObsMap, FitMaps>(
            todproc, stage_profile, logger);
    }
    complete_mapmaking_observation_if_available(engine);
    complete_pointing_observation_if_available(engine);
    complete_beammap_observation_if_available(engine);
}

}  // namespace citlali::pipeline
