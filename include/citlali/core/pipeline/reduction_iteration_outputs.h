#pragma once

#include <citlali/core/pipeline/iteration_lifecycle.h>
#include <citlali/core/pipeline/iteration_coadd_outputs.h>

namespace citlali::pipeline {

template <auto RawCoaddMap, auto FilteredCoaddMap, class TodProc,
          class Logger>
void finish_reduction_iteration(TodProc &todproc, const Logger &logger) {
    write_iteration_coadd_outputs_if_needed<RawCoaddMap, FilteredCoaddMap>(
        todproc, logger);
    finalize_iteration_outputs(todproc, logger);
}

}  // namespace citlali::pipeline
