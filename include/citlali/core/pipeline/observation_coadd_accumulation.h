#pragma once

namespace citlali::pipeline {

template <class TodProc, class Logger>
void coadd_observation(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("coadding");
    if (!engine.rtcproc.run_polarization) {
        todproc.coadd();
    }
}

}  // namespace citlali::pipeline
