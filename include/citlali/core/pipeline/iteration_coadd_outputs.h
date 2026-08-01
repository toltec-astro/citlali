#pragma once

#include <citlali/core/pipeline/filtered_coadd_outputs.h>
#include <citlali/core/pipeline/coadd_provenance_lifecycle.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/raw_coadd_outputs.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <auto RawCoaddMap, auto FilteredCoaddMap, class TodProc,
          class Logger>
void write_iteration_coadd_outputs_if_needed(TodProc &todproc,
                                             StageProfileCollector &stage_profile,
                                             const Logger &logger) {
    auto &engine = todproc.engine();

    if (!should_write_iteration_coadd_outputs(engine)) {
        return;
    }

    const auto profile_scope = profile_stage(stage_profile, "iteration.coadd_outputs", logger);
    begin_mapmaking_coadd_if_available(engine);
    write_raw_coadd_outputs<RawCoaddMap>(todproc, stage_profile, logger);
    write_filtered_coadd_outputs_if_needed<FilteredCoaddMap>(
        todproc, stage_profile, logger);
    record_coadd_realized_maps_if_available(engine);
    complete_mapmaking_coadd_if_available(engine);
}

}  // namespace citlali::pipeline
