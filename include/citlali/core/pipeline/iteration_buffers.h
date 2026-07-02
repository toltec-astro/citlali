#pragma once

#include <citlali/core/pipeline/iteration_buffer_allocation.h>
#include <citlali/core/pipeline/iteration_buffer_policy.h>
#include <citlali/core/pipeline/iteration_buffer_state.h>

namespace citlali::pipeline {

template <class TodProc, class Logger>
void prepare_coadd_iteration_buffers(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    allocate_coadd_map_buffer(todproc, logger);
    if (should_allocate_coadd_noise_buffer(engine)) {
        allocate_coadd_noise_buffer(todproc, logger);
    }

    reset_coadd_iteration_accumulators(engine);
}

template <class TodProc, class Logger>
void prepare_iteration_observation_buffers(TodProc &todproc,
                                           const Logger &logger) {
    auto &engine = todproc.engine();

    clear_iteration_observation_dates(engine);
    if (should_prepare_coadd_iteration_buffers(engine)) {
        prepare_coadd_iteration_buffers(todproc, logger);
    }
}

}  // namespace citlali::pipeline
