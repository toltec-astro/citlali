#pragma once

#include <citlali/core/pipeline/filtered_coadd_outputs.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/raw_coadd_outputs.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <auto RawCoaddMap, auto FilteredCoaddMap, class TodProc,
          class Logger>
void write_iteration_coadd_outputs_if_needed(TodProc &todproc,
                                             const Logger &logger) {
    auto &engine = todproc.engine();

    if (!should_write_iteration_coadd_outputs(engine)) {
        return;
    }

    const auto profile_scope = profile_stage("iteration.coadd_outputs", logger);
    write_raw_coadd_outputs<RawCoaddMap>(todproc, logger);
    write_filtered_coadd_outputs_if_needed<FilteredCoaddMap>(todproc, logger);
}

}  // namespace citlali::pipeline
