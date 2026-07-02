#pragma once

namespace citlali::pipeline {

template <class Engine>
bool should_run_observation_tod(const Engine &engine) {
    return engine.run_tod;
}

template <class Engine, class Logger>
void setup_observation_pipeline(Engine &engine, const Logger &logger) {
    logger->info("pipeline setup");
    engine.setup();
}

template <class Engine, class KidsProc, class RawObs, class Logger>
void setup_and_run_observation_pipeline(Engine &engine, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        const Logger &logger) {
    setup_observation_pipeline(engine, logger);

    if (should_run_observation_tod(engine)) {
        logger->info("running pipeline");
        engine.pipeline(kidsproc, rawobs);
    }
}

}  // namespace citlali::pipeline
