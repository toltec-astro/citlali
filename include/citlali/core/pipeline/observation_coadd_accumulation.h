#pragma once

#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <class Engine>
bool should_run_observation_coadd(const Engine &engine) {
    return !engine.rtcproc.run_polarization;
}

template <class TodProc, class Logger>
void coadd_observation(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("coadding");
    const auto profile_scope = profile_stage("observation.coadd", logger);
    if (should_run_observation_coadd(engine)) {
        todproc.coadd();
    }
}

}  // namespace citlali::pipeline
