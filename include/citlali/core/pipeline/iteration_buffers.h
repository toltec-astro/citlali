#pragma once

namespace citlali::pipeline {

template <class Engine>
bool should_allocate_coadd_noise_buffer(const Engine &engine) {
    return engine.run_noise;
}

template <class TodProc, class Logger>
void allocate_coadd_map_buffer(TodProc &todproc, const Logger &logger) {
    logger->info("allocating cmb");
    todproc.allocate_cmb();
}

template <class TodProc, class Logger>
void prepare_coadd_iteration_buffers(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("allocating cmb");
    todproc.allocate_cmb();
    if (should_allocate_coadd_noise_buffer(engine)) {
        logger->info("allocating nmb");
        todproc.allocate_nmb(engine.cmb);
    }

    engine.cmb.obsnums.clear();
    engine.cmb.exposure_time = 0;
}

template <class TodProc, class Logger>
void prepare_iteration_observation_buffers(TodProc &todproc,
                                           const Logger &logger) {
    auto &engine = todproc.engine();

    engine.date_obs.clear();
    if (engine.run_coadd) {
        prepare_coadd_iteration_buffers(todproc, logger);
    }
}

}  // namespace citlali::pipeline
