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

}  // namespace citlali::pipeline
