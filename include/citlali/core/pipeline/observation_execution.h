#pragma once

namespace citlali::pipeline {

template <class Engine, class KidsProc, class RawObs, class Logger>
void setup_and_run_observation_pipeline(Engine &engine, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        const Logger &logger) {
    logger->info("pipeline setup");
    engine.setup();

    if (engine.run_tod) {
        logger->info("running pipeline");
        engine.pipeline(kidsproc, rawobs);
    }
}

template <class TodProc, class Logger>
void prepare_coadd_iteration_buffers(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("allocating cmb");
    todproc.allocate_cmb();
    if (engine.run_noise) {
        logger->info("allocating nmb");
        todproc.allocate_nmb(engine.cmb);
    }

    engine.cmb.obsnums.clear();
    engine.cmb.exposure_time = 0;
}

}  // namespace citlali::pipeline
