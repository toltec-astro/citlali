#pragma once

namespace citlali::pipeline {

template <class TodProc, class Logger>
void allocate_coadd_map_buffer(TodProc &todproc, const Logger &logger) {
    logger->info("allocating cmb");
    todproc.allocate_cmb();
}

template <class TodProc, class Logger>
void allocate_coadd_noise_buffer(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("allocating nmb");
    todproc.allocate_nmb(engine.cmb);
}

}  // namespace citlali::pipeline
